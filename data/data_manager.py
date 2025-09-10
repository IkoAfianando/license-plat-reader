"""
Data Management System for License Plate Reader
Handles dataset organization, validation, and preprocessing
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Comprehensive data management for LPR system"""
    
    def __init__(self, base_path: str = "data"):
        """Initialize data manager with base directory"""
        self.base_path = Path(base_path)
        self.raw_dir = self.base_path / "raw"
        self.processed_dir = self.base_path / "processed"
        self.annotations_dir = self.base_path / "annotations"
        self.models_dir = self.base_path / "models"
        self.exports_dir = self.base_path / "exports"
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.annotations_dir, 
                        self.models_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.base_path / "dataset_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load dataset metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'created': datetime.now().isoformat(),
                'version': '1.0',
                'datasets': {},
                'statistics': {}
            }
            self.save_metadata()
    
    def save_metadata(self):
        """Save dataset metadata"""
        self.metadata['updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_raw_images(self, 
                      image_paths: List[str], 
                      dataset_name: str = "default",
                      source: str = "unknown") -> Dict:
        """
        Add raw images to the dataset
        
        Args:
            image_paths: List of image file paths
            dataset_name: Name of the dataset
            source: Source of the images (camera_1, manual_collection, etc.)
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'added': 0,
            'skipped': 0,
            'errors': 0,
            'files': []
        }
        
        dataset_dir = self.raw_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        for img_path in image_paths:
            try:
                if not os.path.exists(img_path):
                    results['errors'] += 1
                    continue
                
                # Generate unique filename with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                original_name = Path(img_path).stem
                extension = Path(img_path).suffix
                
                # Calculate file hash for duplicate detection
                file_hash = self._calculate_file_hash(img_path)
                
                # Check for duplicates
                if self._is_duplicate(file_hash):
                    results['skipped'] += 1
                    continue
                
                # Copy file with standardized naming
                new_filename = f"{timestamp}_{source}_{original_name}{extension}"
                dst_path = dataset_dir / new_filename
                
                shutil.copy2(img_path, dst_path)
                
                # Store file metadata
                file_info = {
                    'original_path': img_path,
                    'new_path': str(dst_path),
                    'hash': file_hash,
                    'source': source,
                    'timestamp': timestamp,
                    'size': os.path.getsize(img_path)
                }
                
                results['files'].append(file_info)
                results['added'] += 1
                
                # Update metadata
                if dataset_name not in self.metadata['datasets']:
                    self.metadata['datasets'][dataset_name] = {
                        'created': datetime.now().isoformat(),
                        'files': []
                    }
                
                self.metadata['datasets'][dataset_name]['files'].append(file_info)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results['errors'] += 1
        
        self.save_metadata()
        logger.info(f"Added {results['added']} images to dataset '{dataset_name}'")
        
        return results
    
    def create_annotations_template(self, 
                                  dataset_name: str, 
                                  annotation_format: str = "yolo") -> str:
        """
        Create annotation template files for a dataset
        
        Args:
            dataset_name: Name of the dataset
            annotation_format: Format for annotations (yolo, coco, pascal)
            
        Returns:
            Path to annotation directory
        """
        annotation_dir = self.annotations_dir / dataset_name / annotation_format
        annotation_dir.mkdir(parents=True, exist_ok=True)
        
        # Get images from dataset
        dataset_dir = self.raw_dir / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
        
        if annotation_format == "yolo":
            # Create empty YOLO label files
            for img_path in images:
                label_path = annotation_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    label_path.touch()  # Create empty file
            
            # Create classes.txt file
            classes_file = annotation_dir / "classes.txt"
            with open(classes_file, 'w') as f:
                f.write("license_plate\n")
                
        elif annotation_format == "coco":
            # Create COCO format annotation template
            coco_template = {
                "info": {
                    "description": f"License Plate Dataset - {dataset_name}",
                    "version": "1.0",
                    "year": datetime.now().year
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "license_plate", "supercategory": "object"}
                ]
            }
            
            # Add image entries
            for i, img_path in enumerate(images):
                img = cv2.imread(str(img_path))
                height, width = img.shape[:2]
                
                coco_template["images"].append({
                    "id": i + 1,
                    "width": width,
                    "height": height,
                    "file_name": img_path.name
                })
            
            with open(annotation_dir / "annotations.json", 'w') as f:
                json.dump(coco_template, f, indent=2)
        
        logger.info(f"Created {annotation_format} annotation templates in {annotation_dir}")
        return str(annotation_dir)
    
    def validate_dataset(self, dataset_name: str) -> Dict:
        """
        Validate dataset integrity and completeness
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'images': {'total': 0, 'valid': 0, 'corrupted': []},
            'annotations': {'total': 0, 'valid': 0, 'missing': []},
            'statistics': {}
        }
        
        dataset_dir = self.raw_dir / dataset_name
        if not dataset_dir.exists():
            results['valid'] = False
            results['error'] = f"Dataset '{dataset_name}' not found"
            return results
        
        # Validate images
        image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
        results['images']['total'] = len(image_files)
        
        image_sizes = []
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    results['images']['valid'] += 1
                    height, width = img.shape[:2]
                    image_sizes.append((width, height))
                else:
                    results['images']['corrupted'].append(str(img_path))
            except Exception as e:
                results['images']['corrupted'].append(str(img_path))
        
        # Check annotations (YOLO format)
        annotation_dir = self.annotations_dir / dataset_name / "yolo"
        if annotation_dir.exists():
            for img_path in image_files:
                label_path = annotation_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    results['annotations']['valid'] += 1
                else:
                    results['annotations']['missing'].append(str(img_path))
            results['annotations']['total'] = len(image_files)
        
        # Calculate statistics
        if image_sizes:
            widths, heights = zip(*image_sizes)
            results['statistics'] = {
                'avg_width': sum(widths) / len(widths),
                'avg_height': sum(heights) / len(heights),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights)
            }
        
        # Overall validation
        if results['images']['corrupted'] or results['annotations']['missing']:
            results['valid'] = False
        
        return results
    
    def export_dataset(self, 
                      dataset_name: str, 
                      export_format: str = "yolo",
                      train_split: float = 0.8,
                      val_split: float = 0.1,
                      test_split: float = 0.1) -> str:
        """
        Export dataset in specified format with train/val/test splits
        
        Args:
            dataset_name: Name of the dataset
            export_format: Export format (yolo, coco, pascal)
            train_split: Training set ratio
            val_split: Validation set ratio  
            test_split: Test set ratio
            
        Returns:
            Path to exported dataset
        """
        if not abs(train_split + val_split + test_split - 1.0) < 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        export_dir = self.exports_dir / f"{dataset_name}_{export_format}"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dataset files
        dataset_dir = self.raw_dir / dataset_name
        annotation_dir = self.annotations_dir / dataset_name / export_format
        
        images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
        
        # Shuffle and split dataset
        np.random.shuffle(images)
        n_images = len(images)
        
        train_end = int(n_images * train_split)
        val_end = train_end + int(n_images * val_split)
        
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        # Create export structure
        for split_name, split_images in splits.items():
            split_dir = export_dir / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
            
            # Copy images and labels
            for img_path in split_images:
                # Copy image
                dst_img = split_dir / "images" / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Copy label if exists
                if export_format == "yolo":
                    label_path = annotation_dir / (img_path.stem + ".txt")
                    if label_path.exists():
                        dst_label = split_dir / "labels" / (img_path.stem + ".txt")
                        shutil.copy2(label_path, dst_label)
        
        # Create dataset.yaml for YOLOv8
        if export_format == "yolo":
            dataset_yaml = {
                'path': str(export_dir),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'names': ['license_plate'],
                'nc': 1
            }
            
            with open(export_dir / "dataset.yaml", 'w') as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        # Save export metadata
        export_metadata = {
            'dataset_name': dataset_name,
            'export_format': export_format,
            'created': datetime.now().isoformat(),
            'splits': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            },
            'total_images': n_images
        }
        
        with open(export_dir / "export_info.json", 'w') as f:
            json.dump(export_metadata, f, indent=2)
        
        logger.info(f"Exported dataset '{dataset_name}' to {export_dir}")
        return str(export_dir)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_duplicate(self, file_hash: str) -> bool:
        """Check if file hash already exists in metadata"""
        for dataset in self.metadata['datasets'].values():
            for file_info in dataset['files']:
                if file_info.get('hash') == file_hash:
                    return True
        return False
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_datasets': len(self.metadata['datasets']),
            'total_images': 0,
            'datasets': {}
        }
        
        for name, dataset in self.metadata['datasets'].items():
            dataset_stats = {
                'images': len(dataset['files']),
                'total_size': sum(f.get('size', 0) for f in dataset['files']),
                'sources': list(set(f.get('source', 'unknown') for f in dataset['files']))
            }
            stats['datasets'][name] = dataset_stats
            stats['total_images'] += dataset_stats['images']
        
        return stats


def main():
    """Demo script for data management"""
    print("📁 License Plate Reader Data Management Demo")
    print("=" * 50)
    
    # Initialize data manager
    dm = DataManager()
    
    # Display current statistics
    stats = dm.get_dataset_statistics()
    print(f"📊 Current Statistics:")
    print(f"   Total Datasets: {stats['total_datasets']}")
    print(f"   Total Images: {stats['total_images']}")
    
    if stats['datasets']:
        print(f"   Datasets:")
        for name, dataset_stats in stats['datasets'].items():
            size_mb = dataset_stats['total_size'] / (1024 * 1024)
            print(f"     {name}: {dataset_stats['images']} images ({size_mb:.1f} MB)")
    
    print(f"\n✅ Data management system ready!")
    print(f"   Data directory: {dm.base_path}")
    print(f"   Use DataManager class to manage datasets")


if __name__ == "__main__":
    main()