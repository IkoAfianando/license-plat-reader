"""
License Plate Reader - Basic Usage Examples
Demonstrates common use cases and integration patterns
"""

import cv2
import os
import json
import time
from pathlib import Path
import numpy as np

# Import LPR components
from src.offline.standalone_detector import StandaloneLPRDetector
from src.core.ocr_engine import OCREngine
from data.data_manager import DataManager

# Try to import Roboflow components (optional)
try:
    from src.core.detector import LicensePlateDetector
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("⚠️  Roboflow components not available (install 'roboflow' package)")

class LPRExamples:
    """Collection of usage examples for License Plate Reader"""
    
    def __init__(self):
        """Initialize examples with different detector configurations"""
        print("🚗 License Plate Reader - Basic Usage Examples")
        print("=" * 50)
        
        # Initialize offline detector
        self.offline_detector = StandaloneLPRDetector(confidence=0.5)
        print("✅ Offline detector initialized")
        
        # Initialize OCR engine
        self.ocr = OCREngine(engine='paddleocr')
        print("✅ OCR engine initialized")
        
        # Initialize Roboflow detector if available
        self.roboflow_detector = None
        if ROBOFLOW_AVAILABLE and os.getenv('ROBOFLOW_API_KEY'):
            try:
                roboflow_config = {
                    'api_key': os.getenv('ROBOFLOW_API_KEY'),
                    'project_id': 'license-plate-recognition-rxg4e',
                    'model_version': 4
                }
                self.roboflow_detector = LicensePlateDetector(roboflow_config=roboflow_config)
                print("✅ Roboflow detector initialized")
            except Exception as e:
                print(f"⚠️  Roboflow detector failed: {e}")
        
        # Initialize data manager
        self.data_manager = DataManager()
        print("✅ Data manager initialized")
    
    def example_1_single_image_detection(self, image_path: str):
        """
        Example 1: Basic single image detection
        Detect license plates in one image and extract text
        """
        print("\\n📸 Example 1: Single Image Detection")
        print("-" * 40)
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        print(f"📂 Processing: {Path(image_path).name}")
        print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect with offline method
        start_time = time.time()
        detections = self.offline_detector.detect(image)
        offline_time = time.time() - start_time
        
        print(f"🔍 Offline detection: {len(detections)} plates found ({offline_time:.3f}s)")
        
        # Extract plate regions and OCR
        plate_regions = self.offline_detector.extract_plate_regions(image, detections)
        
        for i, (detection, plate_data) in enumerate(zip(detections, plate_regions)):
            print(f"\\n  Plate {i+1}:")
            print(f"    Confidence: {detection['confidence']:.3f}")
            print(f"    Plate Score: {detection.get('plate_score', 0):.3f}")
            
            # Extract text
            ocr_result = self.ocr.extract_text(plate_data['image'])
            print(f"    Text: '{ocr_result['text']}' (confidence: {ocr_result['confidence']:.3f})")
        
        # Save annotated result
        output_path = f"outputs/images/example1_result_{int(time.time())}.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.offline_detector.save_detection_results(image, detections, output_path)
        print(f"💾 Result saved: {output_path}")
        
        return detections, plate_regions
    
    def example_2_compare_methods(self, image_path: str):
        """
        Example 2: Compare Roboflow vs Offline detection
        Side-by-side comparison of different detection methods
        """
        print("\\n⚖️  Example 2: Compare Detection Methods")
        print("-" * 40)
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        results = {}
        
        # Test offline detection
        print("💻 Testing offline detection...")
        start_time = time.time()
        offline_detections = self.offline_detector.detect(image)
        offline_time = time.time() - start_time
        
        results['offline'] = {
            'detections': len(offline_detections),
            'time': offline_time,
            'method': 'Local YOLOv8 + Heuristics'
        }
        
        # Test Roboflow detection if available
        if self.roboflow_detector:
            print("🌐 Testing Roboflow API...")
            try:
                start_time = time.time()
                roboflow_detections = self.roboflow_detector.detect(image, use_roboflow=True)
                roboflow_time = time.time() - start_time
                
                results['roboflow'] = {
                    'detections': len(roboflow_detections),
                    'time': roboflow_time,
                    'method': 'Roboflow Cloud API'
                }
            except Exception as e:
                print(f"❌ Roboflow detection failed: {e}")
                results['roboflow'] = {'error': str(e)}
        
        # Display comparison
        print("\\n📊 Comparison Results:")
        for method, result in results.items():
            if 'error' in result:
                print(f"  {method.title()}: Error - {result['error']}")
            else:
                print(f"  {method.title()}:")
                print(f"    Detections: {result['detections']}")
                print(f"    Time: {result['time']:.3f}s")
                print(f"    FPS: {1/result['time']:.1f}")
                print(f"    Method: {result['method']}")
        
        return results
    
    def example_3_batch_processing(self, image_dir: str):
        """
        Example 3: Batch process multiple images
        Efficiently process multiple images and generate summary
        """
        print("\\n📦 Example 3: Batch Processing")
        print("-" * 40)
        
        # Find all images in directory
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return
        
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        print(f"📂 Processing {len(image_files)} images from {image_dir}")
        
        batch_results = {
            'processed': 0,
            'total_detections': 0,
            'total_time': 0,
            'files': []
        }
        
        for img_path in image_files:
            try:
                print(f"  Processing {img_path.name}...")
                
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                start_time = time.time()
                detections = self.offline_detector.detect(image)
                processing_time = time.time() - start_time
                
                # Extract text from detected plates
                plate_regions = self.offline_detector.extract_plate_regions(image, detections)
                plate_texts = []
                
                for plate_data in plate_regions:
                    ocr_result = self.ocr.extract_text(plate_data['image'])
                    if ocr_result['text']:
                        plate_texts.append(ocr_result['text'])
                
                file_result = {
                    'filename': img_path.name,
                    'detections': len(detections),
                    'texts': plate_texts,
                    'processing_time': processing_time
                }
                
                batch_results['files'].append(file_result)
                batch_results['processed'] += 1
                batch_results['total_detections'] += len(detections)
                batch_results['total_time'] += processing_time
                
            except Exception as e:
                print(f"    ❌ Error processing {img_path.name}: {e}")
        
        # Generate summary
        if batch_results['processed'] > 0:
            avg_time = batch_results['total_time'] / batch_results['processed']
            avg_fps = 1 / avg_time if avg_time > 0 else 0
            
            print("\\n📈 Batch Processing Summary:")
            print(f"  Images processed: {batch_results['processed']}")
            print(f"  Total detections: {batch_results['total_detections']}")
            print(f"  Average time per image: {avg_time:.3f}s")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Total processing time: {batch_results['total_time']:.1f}s")
            
            # Show top detections
            files_with_plates = [f for f in batch_results['files'] if f['detections'] > 0]
            if files_with_plates:
                print("\\n🎯 Files with license plates:")
                for file_result in sorted(files_with_plates, key=lambda x: x['detections'], reverse=True)[:5]:
                    print(f"    {file_result['filename']}: {file_result['detections']} plates")
                    for text in file_result['texts']:
                        print(f"      → {text}")
        
        # Save batch results
        output_path = f"outputs/reports/batch_results_{int(time.time())}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"💾 Batch results saved: {output_path}")
        return batch_results
    
    def example_4_video_processing(self, video_path: str):
        """
        Example 4: Process video file
        Extract license plates from video frames
        """
        print("\\n🎥 Example 4: Video Processing")
        print("-" * 40)
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video info:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.1f}s")
        
        video_results = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'unique_plates': set(),
            'detections_by_frame': []
        }
        
        frame_skip = 10  # Process every 10th frame for efficiency
        frame_count = 0
        
        print(f"\\n🔄 Processing video (every {frame_skip} frames)...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                video_results['total_frames'] += 1
                
                # Skip frames for efficiency
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                detections = self.offline_detector.detect(frame)
                
                if detections:
                    # Extract text from detected plates
                    plate_regions = self.offline_detector.extract_plate_regions(frame, detections)
                    frame_texts = []
                    
                    for plate_data in plate_regions:
                        ocr_result = self.ocr.extract_text(plate_data['image'])
                        if ocr_result['text'] and ocr_result['confidence'] > 0.7:
                            frame_texts.append(ocr_result['text'])
                            video_results['unique_plates'].add(ocr_result['text'])
                    
                    frame_result = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': len(detections),
                        'texts': frame_texts
                    }
                    
                    video_results['detections_by_frame'].append(frame_result)
                    video_results['total_detections'] += len(detections)
                    
                    print(f"  Frame {frame_count}: {len(detections)} plates")
                    for text in frame_texts:
                        print(f"    → {text}")
                
                video_results['processed_frames'] += 1
                frame_count += 1
                
                # Process max 100 frames for demo
                if video_results['processed_frames'] >= 100:
                    print("  (Stopping at 100 processed frames for demo)")
                    break
                    
        except KeyboardInterrupt:
            print("\\n⏹️  Video processing interrupted")
        finally:
            cap.release()
        
        # Generate video summary
        print("\\n📊 Video Processing Summary:")
        print(f"  Total frames: {video_results['total_frames']}")
        print(f"  Processed frames: {video_results['processed_frames']}")
        print(f"  Total detections: {video_results['total_detections']}")
        print(f"  Unique license plates: {len(video_results['unique_plates'])}")
        
        if video_results['unique_plates']:
            print("\\n🎯 Detected license plates:")
            for plate_text in sorted(video_results['unique_plates']):
                print(f"    {plate_text}")
        
        # Save video results
        # Convert set to list for JSON serialization
        video_results['unique_plates'] = list(video_results['unique_plates'])
        
        output_path = f"outputs/reports/video_results_{int(time.time())}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(video_results, f, indent=2)
        
        print(f"💾 Video results saved: {output_path}")
        return video_results
    
    def example_5_data_management(self):
        """
        Example 5: Data management and organization
        Demonstrate dataset creation and management
        """
        print("\\n📊 Example 5: Data Management")
        print("-" * 40)
        
        # Create sample dataset
        print("🎨 Creating sample dataset...")
        from data.sample_data_generator import SampleDataGenerator
        
        generator = SampleDataGenerator(output_dir="data/raw/demo")
        sample_results = generator.generate_dataset(
            num_images=10,
            region='US',
            dataset_name='demo_dataset'
        )
        
        print(f"✅ Generated {sample_results['generated']} sample images")
        
        # Add dataset to data manager
        image_paths = [str(p) for p in Path("data/raw/demo/demo_dataset").glob("*.jpg")]
        
        dm_result = self.data_manager.add_raw_images(
            image_paths=image_paths[:5],  # Add first 5 images
            dataset_name="demo_managed",
            source="synthetic_generator"
        )
        
        print(f"📂 Added {dm_result['added']} images to managed dataset")
        
        # Create annotations template
        annotation_dir = self.data_manager.create_annotations_template(
            dataset_name="demo_managed",
            annotation_format="yolo"
        )
        
        print(f"📝 Created annotation templates in {annotation_dir}")
        
        # Validate dataset
        validation = self.data_manager.validate_dataset("demo_managed")
        print(f"✅ Dataset validation:")
        print(f"  Valid: {validation['valid']}")
        print(f"  Images: {validation['images']['valid']}/{validation['images']['total']}")
        
        # Get statistics
        stats = self.data_manager.get_dataset_statistics()
        print(f"\\n📈 Dataset Statistics:")
        print(f"  Total datasets: {stats['total_datasets']}")
        print(f"  Total images: {stats['total_images']}")
        
        for name, dataset_stats in stats['datasets'].items():
            size_mb = dataset_stats['total_size'] / (1024 * 1024)
            print(f"  {name}: {dataset_stats['images']} images ({size_mb:.1f} MB)")
        
        return stats
    
    def example_6_performance_benchmark(self):
        """
        Example 6: Performance benchmarking
        Test system performance with different configurations
        """
        print("\\n🏃 Example 6: Performance Benchmark")
        print("-" * 40)
        
        # Generate test images if they don't exist
        test_dir = Path("data/raw/benchmark")
        if not test_dir.exists() or len(list(test_dir.glob("*.jpg"))) < 5:
            print("🎨 Generating benchmark images...")
            from data.sample_data_generator import SampleDataGenerator
            
            generator = SampleDataGenerator(output_dir="data/raw/benchmark")
            generator.generate_dataset(
                num_images=10,
                region='US',
                dataset_name='benchmark'
            )
        
        test_images = list(Path("data/raw/benchmark/benchmark").glob("*.jpg"))[:5]
        test_image_paths = [str(p) for p in test_images]
        
        # Benchmark offline detector
        print("💻 Benchmarking offline detector...")
        benchmark_result = self.offline_detector.benchmark_performance(
            test_image_paths,
            runs=3
        )
        
        print(f"📊 Offline Performance:")
        print(f"  Average inference time: {benchmark_result['avg_inference_time']:.3f}s")
        print(f"  Estimated FPS: {benchmark_result['fps']:.1f}")
        print(f"  Total detections: {benchmark_result['total_detections']}")
        
        # Test different confidence thresholds
        print("\\n🎯 Testing confidence thresholds...")
        thresholds = [0.3, 0.5, 0.7]
        threshold_results = {}
        
        for threshold in thresholds:
            detector_test = StandaloneLPRDetector(confidence=threshold)
            total_detections = 0
            
            for img_path in test_image_paths:
                detections = detector_test.detect(img_path)
                total_detections += len(detections)
            
            threshold_results[threshold] = total_detections
            print(f"  Threshold {threshold}: {total_detections} total detections")
        
        # Find optimal threshold
        optimal_threshold = max(threshold_results.items(), key=lambda x: x[1])
        print(f"\\n🎯 Optimal threshold: {optimal_threshold[0]} ({optimal_threshold[1]} detections)")
        
        return benchmark_result, threshold_results


def main():
    """Run all examples with sample data"""
    
    # Initialize examples
    examples = LPRExamples()
    
    # Create sample data if needed
    sample_dir = Path("sample_images")
    if not sample_dir.exists() or len(list(sample_dir.glob("*.jpg"))) == 0:
        print("\\n🎨 Generating sample images for examples...")
        from data.sample_data_generator import SampleDataGenerator
        
        generator = SampleDataGenerator(output_dir="sample_images")
        generator.generate_dataset(
            num_images=5,
            region='US',
            dataset_name='examples'
        )
        
        # Move files to main sample_images directory
        os.makedirs("sample_images", exist_ok=True)
        source_dir = Path("sample_images/examples")
        if source_dir.exists():
            for img_file in source_dir.glob("*.jpg"):
                img_file.rename(f"sample_images/{img_file.name}")
    
    sample_images = list(Path("sample_images").glob("*.jpg"))
    
    if not sample_images:
        print("❌ No sample images available. Please add some car images to 'sample_images/' directory")
        return
    
    sample_image = str(sample_images[0])
    
    try:
        # Run examples
        print("\\n🚀 Running License Plate Reader Examples")
        print("=" * 50)
        
        # Example 1: Single image detection
        examples.example_1_single_image_detection(sample_image)
        
        # Example 2: Compare methods
        examples.example_2_compare_methods(sample_image)
        
        # Example 3: Batch processing
        examples.example_3_batch_processing("sample_images")
        
        # Example 5: Data management
        examples.example_5_data_management()
        
        # Example 6: Performance benchmark
        examples.example_6_performance_benchmark()
        
        print("\\n✅ All examples completed successfully!")
        print("\\n📁 Check the outputs/ directory for results:")
        print("   outputs/images/     - Annotated detection results")
        print("   outputs/reports/    - JSON reports and analytics")
        print("   outputs/logs/       - System logs")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()