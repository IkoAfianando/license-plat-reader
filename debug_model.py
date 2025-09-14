#!/usr/bin/env python3
"""
Debug script to analyze why the model does not detect license plates
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

def debug_model_predictions():
    """Debug model predictions with various confidence levels"""
    
    # Paths
    dataset_root = Path("/home/ikoafian/COMPANY/ieko-media/dataset")
    images_dir = dataset_root / "images"
    models_dir = Path("models")
    
    # Get test images
    test_images = list(images_dir.glob("*.png"))[:3]  # 3 images for debugging
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"🔍 Debug Model Predictions")
    print("=" * 50)
    
    # Test different models
    models_to_test = [
        ("Pre-trained YOLOv8n", "yolov8n.pt"),
        ("Quick Fine-tuned", models_dir / "quick_license_detector" / "weights" / "best.pt"),
        ("Quick Fine-tuned Last", models_dir / "quick_license_detector" / "weights" / "last.pt"),
        ("Exported Model", models_dir / "license_plate_detector_quick.pt"),
    ]
    
    # Test different confidence levels
    confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    for model_name, model_path in models_to_test:
        print(f"\n🤖 Testing: {model_name}")
        print(f"📁 Path: {model_path}")
        
        # Check if model exists
        if isinstance(model_path, Path) and not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            continue
            
        try:
            # Load model
            model = YOLO(str(model_path))
            print(f"✅ Model loaded successfully")
            
            # Test on first image
            test_img = test_images[0]
            print(f"📸 Testing on: {test_img.name}")
            
            # Try different confidence levels
            for conf in confidence_levels:
                results = model.predict(str(test_img), conf=conf, verbose=False)
                
                if results[0].boxes is not None:
                    num_detections = len(results[0].boxes)
                    if num_detections > 0:
                        print(f"  ✅ Confidence {conf}: {num_detections} detections")
                        
                        # Show first detection details
                        box = results[0].boxes[0]
                        detected_conf = float(box.conf[0])
                        detected_cls = int(box.cls[0])
                        class_name = model.names[detected_cls]
                        coords = box.xyxy[0].cpu().numpy()
                        
                        print(f"     First detection: {class_name} ({detected_conf:.3f}) at {coords}")
                        break
                else:
                    print(f"  ❌ Confidence {conf}: 0 detections")
            
            # Test with very low confidence to see what model detects
            print(f"  🔍 Ultra-low confidence test (0.001):")
            results = model.predict(str(test_img), conf=0.001, verbose=False)
            if results[0].boxes is not None:
                boxes = results[0].boxes
                print(f"     Total objects detected: {len(boxes)}")
                for i, box in enumerate(boxes[:5]):  # Show first 5
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    print(f"     {i+1}. {class_name}: {conf:.4f}")
            else:
                print("     No objects detected even at 0.001 confidence")
                
        except Exception as e:
            print(f"❌ Error loading/testing model: {e}")
    
    return test_images[0] if test_images else None

def visualize_ground_truth_vs_prediction(img_path):
    """Compare ground truth annotations vs model predictions"""
    
    print(f"\n📊 Ground Truth vs Prediction Analysis")
    print("=" * 50)
    
    # Paths
    annotations_dir = Path("/home/ikoafian/COMPANY/ieko-media/dataset/annotations")
    models_dir = Path("models")
    
    # Get ground truth
    xml_path = annotations_dir / f"{img_path.stem}.xml"
    
    if not xml_path.exists():
        print(f"❌ No annotation file: {xml_path}")
        return
    
    # Parse ground truth
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    ground_truth_boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        ground_truth_boxes.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    print(f"📍 Ground Truth: {len(ground_truth_boxes)} license plates")
    for i, gt in enumerate(ground_truth_boxes):
        bbox = gt['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        print(f"  {i+1}. {gt['name']}: {bbox} (size: {width}x{height})")
    
    # Load and test model
    model_path = models_dir / "license_plate_detector_quick.pt"
    if model_path.exists():
        try:
            model = YOLO(str(model_path))
            
            # Test with different confidence levels
            for conf in [0.01, 0.1, 0.3, 0.5]:
                results = model.predict(str(img_path), conf=conf, verbose=False)
                if results[0].boxes is not None:
                    print(f"🤖 Model (conf={conf}): {len(results[0].boxes)} detections")
                    break
                else:
                    print(f"🤖 Model (conf={conf}): 0 detections")
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
    else:
        print(f"❌ Model not found: {model_path}")

def create_visual_comparison():
    """Create visual comparison of ground truth vs predictions"""
    
    dataset_root = Path("/home/ikoafian/COMPANY/ieko-media/dataset")
    images_dir = dataset_root / "images"
    test_images = list(images_dir.glob("*.png"))[:2]  # 2 images
    
    if not test_images:
        print("❌ No images found for visualization")
        return
    
    models_dir = Path("models")
    model_path = models_dir / "license_plate_detector_quick.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(str(model_path))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        for i, img_path in enumerate(test_images):
            # Load image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ground truth visualization (from previous code)
            gt_img = img_rgb.copy()
            
            # Get ground truth
            annotations_dir = Path("/home/ikoafian/COMPANY/ieko-media/dataset/annotations")
            xml_path = annotations_dir / f"{img_path.stem}.xml"
            
            if xml_path.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    cv2.rectangle(gt_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(gt_img, 'GT License', (xmin, ymin-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Model prediction
            pred_img = img_rgb.copy()
            results = model.predict(str(img_path), conf=0.01, verbose=False)  # Very low confidence
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(pred_img, f'{class_name} {conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display
            axes[i, 0].imshow(gt_img)
            axes[i, 0].set_title(f'{img_path.stem} - Ground Truth')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f'{img_path.stem} - Model Prediction')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('debug_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visual comparison saved as 'debug_comparison.png'")
        
    except Exception as e:
        print(f"❌ Error creating visual comparison: {e}")

if __name__ == "__main__":
    # Run debug
    test_img = debug_model_predictions()
    
    if test_img:
        visualize_ground_truth_vs_prediction(test_img)
        create_visual_comparison()
    
    print("\n💡 Possible solutions:")
    print("1. Lower confidence threshold (try 0.01 instead of 0.3)")
    print("2. Train longer (50+ epochs instead of 5)")
    print("3. Check if model learned correct class (should be class 0)")
    print("4. Verify dataset format is correct")
    print("5. Use pre-trained model + post-processing")
