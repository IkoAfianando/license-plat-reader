"""
Model Management System for License Plate Reader
Handles model downloading, loading, training, and versioning
"""

import os
import json
import shutil
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import yaml

try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.error("PyTorch/Ultralytics not available. Install with: pip install torch ultralytics")

logger = logging.getLogger(__name__)

class ModelManager:
    """Comprehensive model management for LPR system"""
    
    def __init__(self, base_path: str = "models"):
        """Initialize model manager with base directory"""
        self.base_path = Path(base_path)
        self.pretrained_dir = self.base_path / "pretrained"
        self.custom_dir = self.base_path / "custom"
        self.exports_dir = self.base_path / "exports"
        
        # Create directories
        for dir_path in [self.pretrained_dir, self.custom_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.base_path / "model_registry.json"
        self.load_registry()
        
        # Available pretrained models
        self.available_models = {
            'yolov8n.pt': {
                'size': '6.2MB',
                'speed': 'fastest',
                'accuracy': 'good',
                'description': 'YOLOv8 Nano - fastest inference',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
            },
            'yolov8s.pt': {
                'size': '21.5MB', 
                'speed': 'fast',
                'accuracy': 'better',
                'description': 'YOLOv8 Small - balanced speed/accuracy',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt'
            },
            'yolov8m.pt': {
                'size': '49.7MB',
                'speed': 'medium',
                'accuracy': 'high',
                'description': 'YOLOv8 Medium - high accuracy',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt'
            },
            'yolov8l.pt': {
                'size': '83.7MB',
                'speed': 'slow',
                'accuracy': 'higher',
                'description': 'YOLOv8 Large - higher accuracy',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt'
            },
            'yolov8x.pt': {
                'size': '136.7MB',
                'speed': 'slowest', 
                'accuracy': 'highest',
                'description': 'YOLOv8 Extra Large - highest accuracy',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
            }
        }
    
    def load_registry(self):
        """Load model registry metadata"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'created': datetime.now().isoformat(),
                'version': '1.0',
                'models': {}
            }
            self.save_registry()
    
    def save_registry(self):
        """Save model registry metadata"""
        self.registry['updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def download_model(self, 
                      model_name: str, 
                      force_download: bool = False) -> Dict[str, Any]:
        """
        Download pretrained model
        
        Args:
            model_name: Name of the model to download
            force_download: Force redownload even if exists
            
        Returns:
            Download result dictionary
        """
        if model_name not in self.available_models:
            available = ', '.join(self.available_models.keys())
            raise ValueError(f"Model '{model_name}' not available. Available: {available}")
        
        model_path = self.pretrained_dir / model_name
        model_info = self.available_models[model_name]
        
        # Check if already exists
        if model_path.exists() and not force_download:
            logger.info(f"Model {model_name} already exists")
            return {
                'status': 'exists',
                'path': str(model_path),
                'size': model_path.stat().st_size
            }
        
        try:
            logger.info(f"Downloading {model_name} ({model_info['size']})...")
            
            # Download with progress
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Progress update every 1MB
                        if downloaded_size % (1024 * 1024) == 0:
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"  Progress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f}MB)")
            
            # Verify download
            actual_size = model_path.stat().st_size
            file_hash = self._calculate_file_hash(str(model_path))
            
            # Update registry
            self.registry['models'][model_name] = {
                'type': 'pretrained',
                'path': str(model_path),
                'size': actual_size,
                'hash': file_hash,
                'downloaded': datetime.now().isoformat(),
                'info': model_info
            }
            self.save_registry()
            
            logger.info(f"Successfully downloaded {model_name}")
            
            return {
                'status': 'downloaded',
                'path': str(model_path),
                'size': actual_size,
                'hash': file_hash
            }
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            raise
    
    def load_model(self, 
                   model_path: str, 
                   device: str = 'auto') -> Tuple[Any, Dict[str, Any]]:
        """
        Load YOLO model from path
        
        Args:
            model_path: Path to model file
            device: Device to load model on
            
        Returns:
            Tuple of (model, model_info)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch ultralytics")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            logger.info(f"Loading model: {model_path}")
            
            # Load YOLO model
            model = YOLO(str(model_path))
            
            # Get model info
            model_info = {
                'path': str(model_path),
                'name': model_path.name,
                'size': model_path.stat().st_size,
                'device': device,
                'loaded': datetime.now().isoformat()
            }
            
            # Try to get model details
            if hasattr(model, 'model'):
                try:
                    model_info['parameters'] = sum(p.numel() for p in model.model.parameters())
                    model_info['classes'] = len(model.names) if hasattr(model, 'names') else 'unknown'
                    model_info['input_size'] = getattr(model.model, 'imgsz', [640, 640])
                except:
                    pass
            
            logger.info(f"Model loaded successfully: {model_path.name}")
            
            return model, model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def validate_model(self, 
                      model_path: str, 
                      test_images: List[str]) -> Dict[str, Any]:
        """
        Validate model performance on test images
        
        Args:
            model_path: Path to model file
            test_images: List of test image paths
            
        Returns:
            Validation results
        """
        model, model_info = self.load_model(model_path)
        
        validation_results = {
            'model_path': str(model_path),
            'test_images': len(test_images),
            'results': [],
            'summary': {}
        }
        
        total_inference_time = 0
        total_detections = 0
        successful_inferences = 0
        
        for img_path in test_images:
            try:
                if not os.path.exists(img_path):
                    continue
                
                # Run inference
                import time
                start_time = time.time()
                results = model(img_path, verbose=False)
                inference_time = time.time() - start_time
                
                # Count detections
                detections = 0
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        detections = len(result.boxes)
                
                image_result = {
                    'image_path': img_path,
                    'detections': detections,
                    'inference_time': inference_time,
                    'success': True
                }
                
                validation_results['results'].append(image_result)
                
                total_inference_time += inference_time
                total_detections += detections
                successful_inferences += 1
                
            except Exception as e:
                error_result = {
                    'image_path': img_path,
                    'error': str(e),
                    'success': False
                }
                validation_results['results'].append(error_result)
        
        # Calculate summary statistics
        if successful_inferences > 0:
            validation_results['summary'] = {
                'successful_inferences': successful_inferences,
                'total_detections': total_detections,
                'avg_inference_time': total_inference_time / successful_inferences,
                'avg_detections_per_image': total_detections / successful_inferences,
                'estimated_fps': successful_inferences / total_inference_time if total_inference_time > 0 else 0
            }
        
        return validation_results
    
    def create_custom_model(self, 
                           base_model: str,
                           dataset_path: str,
                           model_name: str,
                           epochs: int = 100,
                           batch_size: int = 16,
                           image_size: int = 640) -> Dict[str, Any]:
        """
        Train custom model on dataset
        
        Args:
            base_model: Base model to start training from
            dataset_path: Path to dataset.yaml file
            model_name: Name for the custom model
            epochs: Number of training epochs
            batch_size: Training batch size
            image_size: Input image size
            
        Returns:
            Training results
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for training")
        
        # Ensure base model is available
        base_model_path = self.pretrained_dir / base_model
        if not base_model_path.exists():
            logger.info(f"Downloading base model: {base_model}")
            self.download_model(base_model)
        
        # Verify dataset exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        try:
            logger.info(f"Starting training: {model_name}")
            logger.info(f"Base model: {base_model}")
            logger.info(f"Dataset: {dataset_path}")
            logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
            
            # Load base model
            model = YOLO(str(base_model_path))
            
            # Create output directory for this training run
            training_dir = self.custom_dir / model_name
            training_dir.mkdir(exist_ok=True)
            
            # Start training
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                name=model_name,
                project=str(self.custom_dir),
                save_period=10,  # Save checkpoint every 10 epochs
                patience=50,     # Early stopping patience
                verbose=True
            )
            
            # Find the best model
            best_model_path = training_dir / "weights" / "best.pt"
            last_model_path = training_dir / "weights" / "last.pt"
            
            custom_model_path = self.custom_dir / f"{model_name}.pt"
            
            # Copy best model to main location
            if best_model_path.exists():
                shutil.copy2(best_model_path, custom_model_path)
                logger.info(f"Custom model saved: {custom_model_path}")
            elif last_model_path.exists():
                shutil.copy2(last_model_path, custom_model_path)
                logger.info(f"Custom model saved (last checkpoint): {custom_model_path}")
            
            # Extract training metrics
            training_results = {
                'model_name': model_name,
                'base_model': base_model,
                'dataset': dataset_path,
                'epochs_completed': epochs,
                'training_directory': str(training_dir),
                'model_path': str(custom_model_path),
                'created': datetime.now().isoformat()
            }
            
            # Try to extract metrics from results
            if hasattr(results, 'results_dict'):
                training_results['final_metrics'] = results.results_dict
            
            # Update registry
            model_hash = self._calculate_file_hash(str(custom_model_path))
            self.registry['models'][model_name] = {
                'type': 'custom',
                'path': str(custom_model_path),
                'size': custom_model_path.stat().st_size,
                'hash': model_hash,
                'training_results': training_results
            }
            self.save_registry()
            
            logger.info(f"Training completed successfully: {model_name}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def export_model(self, 
                    model_path: str, 
                    export_format: str = 'onnx',
                    optimize: bool = True) -> Dict[str, Any]:
        """
        Export model to different formats
        
        Args:
            model_path: Path to model file
            export_format: Export format (onnx, tensorrt, coreml, etc.)
            optimize: Apply optimization during export
            
        Returns:
            Export results
        """
        model, model_info = self.load_model(model_path)
        
        try:
            logger.info(f"Exporting model to {export_format}: {model_path}")
            
            # Export model
            export_path = model.export(
                format=export_format,
                optimize=optimize,
                verbose=True
            )
            
            # Move to exports directory
            export_filename = Path(model_path).stem + f"_{export_format}.{export_format}"
            final_export_path = self.exports_dir / export_filename
            
            if isinstance(export_path, (str, Path)):
                shutil.move(str(export_path), str(final_export_path))
            
            export_results = {
                'original_model': str(model_path),
                'export_format': export_format,
                'export_path': str(final_export_path),
                'optimized': optimize,
                'exported': datetime.now().isoformat()
            }
            
            # Update registry
            export_name = f"{Path(model_path).stem}_{export_format}"
            self.registry['models'][export_name] = {
                'type': 'export',
                'format': export_format,
                'path': str(final_export_path),
                'size': final_export_path.stat().st_size,
                'source_model': str(model_path),
                'export_results': export_results
            }
            self.save_registry()
            
            logger.info(f"Model exported successfully: {final_export_path}")
            
            return export_results
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available models
        
        Args:
            model_type: Filter by model type (pretrained, custom, export)
            
        Returns:
            List of model information
        """
        models = []
        
        for name, info in self.registry['models'].items():
            if model_type and info.get('type') != model_type:
                continue
            
            model_info = {
                'name': name,
                'type': info.get('type', 'unknown'),
                'path': info.get('path'),
                'size_mb': info.get('size', 0) / (1024 * 1024),
                'exists': os.path.exists(info.get('path', ''))
            }
            
            # Add type-specific information
            if info.get('type') == 'custom' and 'training_results' in info:
                training = info['training_results']
                model_info.update({
                    'base_model': training.get('base_model'),
                    'epochs': training.get('epochs_completed'),
                    'created': training.get('created')
                })
            
            elif info.get('type') == 'export':
                model_info.update({
                    'format': info.get('format'),
                    'source_model': info.get('source_model')
                })
            
            models.append(model_info)
        
        return sorted(models, key=lambda x: x['name'])
    
    def delete_model(self, model_name: str, remove_files: bool = False) -> Dict[str, Any]:
        """
        Delete model from registry and optionally remove files
        
        Args:
            model_name: Name of the model to delete
            remove_files: Whether to delete the actual model files
            
        Returns:
            Deletion results
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.registry['models'][model_name]
        model_path = model_info.get('path')
        
        results = {
            'model_name': model_name,
            'registry_removed': False,
            'files_removed': False
        }
        
        # Remove from registry
        del self.registry['models'][model_name]
        self.save_registry()
        results['registry_removed'] = True
        
        # Remove files if requested
        if remove_files and model_path and os.path.exists(model_path):
            try:
                os.remove(model_path)
                results['files_removed'] = True
                logger.info(f"Deleted model files: {model_path}")
            except Exception as e:
                logger.error(f"Failed to delete model files: {e}")
        
        logger.info(f"Model '{model_name}' deleted from registry")
        
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        if model_name not in self.registry['models']:
            # Check if it's an available pretrained model
            if model_name in self.available_models:
                return {
                    'name': model_name,
                    'type': 'available_pretrained',
                    'info': self.available_models[model_name],
                    'downloaded': False
                }
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.registry['models'][model_name]
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about model registry"""
        stats = {
            'total_models': len(self.registry['models']),
            'by_type': {},
            'total_size_mb': 0,
            'available_pretrained': len(self.available_models)
        }
        
        for model_info in self.registry['models'].values():
            model_type = model_info.get('type', 'unknown')
            stats['by_type'][model_type] = stats['by_type'].get(model_type, 0) + 1
            stats['total_size_mb'] += model_info.get('size', 0) / (1024 * 1024)
        
        return stats


def main():
    """Demo script for model management"""
    print("🤖 License Plate Reader Model Management Demo")
    print("=" * 50)
    
    # Initialize model manager
    mm = ModelManager()
    
    # Display available models
    print("📋 Available Pretrained Models:")
    for name, info in mm.available_models.items():
        print(f"  {name}: {info['size']} - {info['description']}")
    
    # Display registry stats
    stats = mm.get_registry_stats()
    print(f"\\n📊 Registry Statistics:")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Available for download: {stats['available_pretrained']}")
    
    if stats['by_type']:
        print(f"  By type:")
        for model_type, count in stats['by_type'].items():
            print(f"    {model_type}: {count}")
    
    # List current models
    models = mm.list_models()
    if models:
        print(f"\\n📁 Current Models:")
        for model in models:
            status = "✅" if model['exists'] else "❌"
            print(f"  {status} {model['name']} ({model['type']}, {model['size_mb']:.1f}MB)")
    
    print(f"\\n✅ Model management system ready!")
    print(f"   Models directory: {mm.base_path}")
    print(f"   Use ModelManager class to manage models")


if __name__ == "__main__":
    main()