"""
Sample Data Generator for License Plate Reader System
Creates synthetic license plate images for testing and development
"""

import cv2
import numpy as np
import random
import string
from pathlib import Path
import json
from typing import List, Tuple, Dict
import argparse

class SampleDataGenerator:
    """Generate synthetic license plate images for testing"""
    
    def __init__(self, output_dir: str = "data/raw/synthetic"):
        """Initialize generator with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # License plate templates for different regions
        self.plate_formats = {
            'US': [
                {'pattern': 'AAA-1111', 'width': 300, 'height': 160},
                {'pattern': '111-AAA', 'width': 300, 'height': 160},
                {'pattern': 'A11-1AA', 'width': 300, 'height': 160},
            ],
            'EU': [
                {'pattern': 'AA-111-AA', 'width': 520, 'height': 110},
                {'pattern': '1-AAA-111', 'width': 520, 'height': 110},
            ]
        }
        
        # Color schemes
        self.color_schemes = [
            {'bg': (255, 255, 255), 'text': (0, 0, 0)},      # White/Black
            {'bg': (255, 255, 0), 'text': (0, 0, 0)},        # Yellow/Black  
            {'bg': (0, 0, 255), 'text': (255, 255, 255)},    # Blue/White
            {'bg': (0, 255, 0), 'text': (0, 0, 0)},          # Green/Black
        ]
        
        # Background vehicle images (placeholders)
        self.vehicle_colors = [
            (128, 128, 128),  # Gray
            (255, 255, 255),  # White
            (0, 0, 0),        # Black
            (0, 0, 255),      # Red
            (255, 0, 0),      # Blue
        ]
    
    def generate_plate_text(self, pattern: str) -> str:
        """Generate random text based on pattern"""
        result = ""
        for char in pattern:
            if char == 'A':
                result += random.choice(string.ascii_uppercase)
            elif char == '1':
                result += random.choice(string.digits)
            else:
                result += char
        return result
    
    def create_license_plate(self, 
                           text: str, 
                           width: int, 
                           height: int,
                           color_scheme: Dict) -> np.ndarray:
        """Create a synthetic license plate image"""
        # Create plate background
        plate = np.full((height, width, 3), color_scheme['bg'], dtype=np.uint8)
        
        # Add border
        cv2.rectangle(plate, (0, 0), (width-1, height-1), (0, 0, 0), 3)
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width / 200, height / 50)  # Scale font to plate size
        thickness = max(1, int(font_scale * 2))
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Center text
        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2
        
        # Add text
        cv2.putText(plate, text, (x, y), font, font_scale, 
                   color_scheme['text'], thickness, cv2.LINE_AA)
        
        return plate
    
    def create_vehicle_scene(self, 
                           plate_img: np.ndarray,
                           scene_width: int = 640,
                           scene_height: int = 480) -> Tuple[np.ndarray, List[int]]:
        """Create a synthetic vehicle scene with license plate"""
        # Create vehicle background
        vehicle_color = random.choice(self.vehicle_colors)
        scene = np.full((scene_height, scene_width, 3), (100, 150, 100), dtype=np.uint8)  # Road color
        
        # Draw simple vehicle rectangle (rear view)
        vehicle_width = random.randint(200, 300)
        vehicle_height = random.randint(120, 180)
        vehicle_x = (scene_width - vehicle_width) // 2
        vehicle_y = scene_height - vehicle_height - 50
        
        cv2.rectangle(scene, 
                     (vehicle_x, vehicle_y), 
                     (vehicle_x + vehicle_width, vehicle_y + vehicle_height),
                     vehicle_color, -1)
        
        # Add some vehicle details (windows, lights)
        # Rear window
        window_y = vehicle_y + 10
        window_height = vehicle_height // 3
        cv2.rectangle(scene,
                     (vehicle_x + 20, window_y),
                     (vehicle_x + vehicle_width - 20, window_y + window_height),
                     (50, 50, 150), -1)
        
        # Tail lights
        light_size = 15
        cv2.circle(scene, (vehicle_x + 20, vehicle_y + vehicle_height - 30), 
                  light_size, (0, 0, 255), -1)
        cv2.circle(scene, (vehicle_x + vehicle_width - 20, vehicle_y + vehicle_height - 30), 
                  light_size, (0, 0, 255), -1)
        
        # Position license plate
        plate_h, plate_w = plate_img.shape[:2]
        
        # Scale plate to fit vehicle
        scale_factor = min(vehicle_width * 0.6 / plate_w, 40 / plate_h)
        new_plate_w = int(plate_w * scale_factor)
        new_plate_h = int(plate_h * scale_factor)
        
        plate_scaled = cv2.resize(plate_img, (new_plate_w, new_plate_h))
        
        # Position plate at bottom center of vehicle
        plate_x = vehicle_x + (vehicle_width - new_plate_w) // 2
        plate_y = vehicle_y + vehicle_height - new_plate_h - 10
        
        # Add plate to scene
        scene[plate_y:plate_y + new_plate_h, plate_x:plate_x + new_plate_w] = plate_scaled
        
        # Return bounding box in YOLO format (center_x, center_y, width, height) normalized
        center_x = (plate_x + new_plate_w / 2) / scene_width
        center_y = (plate_y + new_plate_h / 2) / scene_height
        norm_width = new_plate_w / scene_width
        norm_height = new_plate_h / scene_height
        
        bbox = [center_x, center_y, norm_width, norm_height]
        
        return scene, bbox
    
    def add_noise_and_distortion(self, image: np.ndarray) -> np.ndarray:
        """Add realistic noise and distortions to image"""
        
        brightness = random.randint(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
            
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
                
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        if random.random() < 0.2:
            h, w = image.shape[:2]    
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = random.randint(-10, 10)
            pts2 = np.float32([[offset, 0], [w-offset, 0], [0, h], [w, h]])
            
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (w, h))
        
        return image
    
    def generate_dataset(self, 
                        num_images: int = 100,
                        region: str = 'US',
                        dataset_name: str = 'synthetic') -> Dict:
        """
        Generate a complete synthetic dataset
        
        Args:
            num_images: Number of images to generate
            region: License plate region (US, EU)
            dataset_name: Name of the dataset
            
        Returns:
            Generation statistics
        """
        if region not in self.plate_formats:
            raise ValueError(f"Region '{region}' not supported. Available: {list(self.plate_formats.keys())}")
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Create labels directory for YOLO format
        labels_dir = Path(f"data/annotations/{dataset_name}/yolo")
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'generated': 0,
            'errors': 0,
            'region': region,
            'dataset_name': dataset_name,
            'files': []
        }
        
        print(f"🎨 Generating {num_images} synthetic {region} license plate images...")
        
        for i in range(num_images):
            try:
                # Choose random plate format
                plate_format = random.choice(self.plate_formats[region])
                
                # Generate random plate text
                plate_text = self.generate_plate_text(plate_format['pattern'])
                
                # Choose random color scheme
                color_scheme = random.choice(self.color_schemes)
                
                # Create license plate
                plate_img = self.create_license_plate(
                    plate_text, 
                    plate_format['width'],
                    plate_format['height'],
                    color_scheme
                )
                
                # Create vehicle scene
                scene_img, bbox = self.create_vehicle_scene(plate_img)
                
                # Add noise and distortions
                scene_img = self.add_noise_and_distortion(scene_img)
                
                # Save image
                filename = f"synthetic_{region.lower()}_{i:04d}.jpg"
                img_path = dataset_dir / filename
                cv2.imwrite(str(img_path), scene_img)
                
                # Save YOLO label
                label_filename = filename.replace('.jpg', '.txt')
                label_path = labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    # Class 0 (license_plate) + bounding box
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\\n")
                
                # Store metadata
                file_info = {
                    'filename': filename,
                    'plate_text': plate_text,
                    'region': region,
                    'format': plate_format['pattern'],
                    'color_scheme': color_scheme,
                    'bbox': bbox
                }
                
                results['files'].append(file_info)
                results['generated'] += 1
                
                if (i + 1) % 10 == 0:
                    print(f"   Generated {i + 1}/{num_images} images...")
                    
            except Exception as e:
                print(f"   Error generating image {i}: {e}")
                results['errors'] += 1
        
        # Save dataset metadata
        metadata_path = dataset_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create classes.txt for YOLO
        classes_path = labels_dir / "classes.txt"
        with open(classes_path, 'w') as f:
            f.write("license_plate\\n")
        
        print(f"✅ Dataset generation complete!")
        print(f"   Generated: {results['generated']} images")
        print(f"   Errors: {results['errors']}")
        print(f"   Saved to: {dataset_dir}")
        print(f"   Labels: {labels_dir}")
        
        return results


def main():
    """Command line interface for sample data generation"""
    parser = argparse.ArgumentParser(description="Generate synthetic license plate data")
    parser.add_argument('--num_images', type=int, default=50, 
                       help='Number of images to generate (default: 50)')
    parser.add_argument('--region', choices=['US', 'EU'], default='US',
                       help='License plate region (default: US)')
    parser.add_argument('--dataset_name', type=str, default='synthetic',
                       help='Dataset name (default: synthetic)')
    parser.add_argument('--output_dir', type=str, default='data/raw/synthetic',
                       help='Output directory (default: data/raw/synthetic)')
    
    args = parser.parse_args()
    
    print("🚗 License Plate Sample Data Generator")
    print("=" * 40)
    
    # Initialize generator
    generator = SampleDataGenerator(args.output_dir)
    
    # Generate dataset
    results = generator.generate_dataset(
        num_images=args.num_images,
        region=args.region,
        dataset_name=args.dataset_name
    )
    
    print("\\n📊 Generation Summary:")
    print(f"   Region: {results['region']}")
    print(f"   Dataset: {results['dataset_name']}")
    print(f"   Success: {results['generated']}/{args.num_images}")
    
    if results['generated'] > 0:
        print("\\n💡 Next Steps:")
        print("   1. Review generated images for quality")
        print("   2. Use data_manager.py to organize dataset")
        print("   3. Train custom model with this data")
        print("   4. Test detection accuracy")


if __name__ == "__main__":
    main()