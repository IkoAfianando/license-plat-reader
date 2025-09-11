"""
Video License Plate Detection and OCR with Roboflow
This script processes video files to detect license plates and perform OCR.
Outputs annotated video with detected license plates and recognized text.
"""

import os
from dotenv import load_dotenv
import json
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Optional
import datetime

# Load environment variables
load_dotenv()

# Import checks
try:
    from PIL import Image, ImageDraw, ImageFont
    print("✅ Pillow (PIL) library imported successfully")
except ImportError:
    print("❌ Pillow library not found. Install with: pip install Pillow")
    exit(1)

try:
    import easyocr
    print("✅ EasyOCR library imported successfully")
except ImportError:
    print("❌ EasyOCR library not found. Install with: pip install easyocr")
    exit(1)

try:
    from roboflow import Roboflow
    print("✅ Roboflow library imported successfully")
except ImportError:
    print("❌ Roboflow library not found. Install with: pip install roboflow")
    exit(1)


class RoboflowVideoProcessor:
    def __init__(self, api_key=None, workspace=None, project_slug=None, version=None):
        """Initialize Roboflow API connection and OCR reader"""
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        if not self.api_key:
            print("❌ API key required. Set ROBOFLOW_API_KEY environment variable")
            exit(1)
        
        # Initialize OCR Reader
        print("\n⏳ Initializing OCR engine (EasyOCR)... This may take a moment.")
        self.ocr_reader = easyocr.Reader(['en'])
        print("✅ OCR engine initialized.")
            
        try:
            self.rf = Roboflow(api_key=self.api_key)
            resolved_workspace = workspace or os.getenv('ROBOFLOW_WORKSPACE', 'test-aip6t')
            resolved_project = project_slug or os.getenv('ROBOFLOW_PROJECT', 'license-plate-recognition-8fvub-hvrra')
            resolved_version = version or int(os.getenv('ROBOFLOW_VERSION', '2'))

            self.project = self.rf.workspace(resolved_workspace).project(resolved_project)
            self.model = self.project.version(resolved_version).model
            print("✅ Connected to Roboflow API")
            print(f"🔗 Model: workspace='{resolved_workspace}', project='{resolved_project}', version={resolved_version}")
        except Exception as e:
            print(f"❌ Failed to connect to Roboflow: {e}")
            exit(1)

    def process_frame(self, frame: np.ndarray, confidence: int = 40, overlap: int = 30) -> Tuple[np.ndarray, List[dict]]:
        """Process single frame for license plate detection and OCR"""
        try:
            # Save frame temporarily for Roboflow processing
            temp_frame_path = "/tmp/temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            # Run Roboflow detection
            result = self.model.predict(temp_frame_path, confidence=confidence, overlap=overlap)
            detections = result.json()
            
            # Clean up temp file
            os.remove(temp_frame_path)
            
            processed_detections = []
            annotated_frame = frame.copy()
            
            if detections.get('predictions'):
                for detection in detections['predictions']:
                    # Extract bounding box coordinates
                    x_center = detection['x']
                    y_center = detection['y']
                    width = detection['width']
                    height = detection['height']
                    
                    left = int(x_center - (width / 2))
                    top = int(y_center - (height / 2))
                    right = int(x_center + (width / 2))
                    bottom = int(y_center + (height / 2))
                    
                    # Ensure coordinates are within frame bounds
                    left = max(0, left)
                    top = max(0, top)
                    right = min(frame.shape[1], right)
                    bottom = min(frame.shape[0], bottom)
                    
                    # Crop license plate region
                    cropped_plate = frame[top:bottom, left:right]
                    
                    # Run OCR on cropped region
                    ocr_results = self.ocr_reader.readtext(cropped_plate)
                    recognized_text = " ".join([res[1] for res in ocr_results]).strip()
                    
                    # Store detection info
                    detection_info = {
                        'bbox': [left, top, right, bottom],
                        'confidence': detection['confidence'],
                        'ocr_text': recognized_text,
                        'ocr_confidence': [res[2] for res in ocr_results] if ocr_results else []
                    }
                    processed_detections.append(detection_info)
                    
                    # Draw bounding box on frame
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Add text label
                    label = f"{recognized_text} ({detection['confidence']:.1f}%)"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, 
                                (left, top - label_size[1] - 10), 
                                (left + label_size[0], top), 
                                (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, 
                              (left, top - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 0, 0), 2)
            
            return annotated_frame, processed_detections
            
        except Exception as e:
            print(f"❌ Frame processing failed: {e}")
            return frame, []

    def process_video(self, input_video_path: str, output_dir: str, 
                     confidence: int = 40, overlap: int = 30, 
                     skip_frames: int = 1) -> dict:
        """Process entire video file"""
        print(f"\n🎬 Processing video: {input_video_path}")
        
        if not os.path.exists(input_video_path):
            print(f"❌ Video file not found: {input_video_path}")
            return None
            
        # Setup video capture
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video file: {input_video_path}")
            return None
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.1f}s")
        
        # Setup output video writer
        base_name = Path(input_video_path).stem
        output_video_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")
        output_json_path = os.path.join(output_dir, f"{base_name}_detections.json")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        processed_frames = 0
        all_detections = []
        start_time = time.time()
        
        print(f"🚀 Starting video processing (skip every {skip_frames} frames)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame based on skip_frames parameter
            if frame_count % (skip_frames + 1) == 0:
                annotated_frame, detections = self.process_frame(frame, confidence, overlap)
                processed_frames += 1
                
                # Store detections with timestamp
                if detections:
                    frame_detections = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': detections
                    }
                    all_detections.append(frame_detections)
                
                # Write annotated frame
                out.write(annotated_frame)
                
                # Progress update
                if processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Progress: {progress:.1f}% ({processed_frames} frames processed, {elapsed:.1f}s elapsed)")
            else:
                # Write original frame for skipped frames
                out.write(frame)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        
        # Save detection results to JSON
        results_summary = {
            'video_info': {
                'input_file': input_video_path,
                'output_file': output_video_path,
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            },
            'processing_info': {
                'processing_time': processing_time,
                'frames_processed': processed_frames,
                'skip_frames': skip_frames,
                'confidence_threshold': confidence,
                'overlap_threshold': overlap,
                'total_detections': len(all_detections)
            },
            'detections': all_detections
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✅ Video processing complete!")
        print(f"📹 Output video: {output_video_path}")
        print(f"📊 Detections JSON: {output_json_path}")
        print(f"⏱️  Total processing time: {processing_time:.1f}s")
        print(f"🔍 Total detections found: {len(all_detections)}")
        
        return results_summary


def resolve_input_dir(cli_input: Optional[str], repo_root: Path) -> str:
    """Resolve input directory for videos"""
    # Check for data/videos directory
    repo_default = repo_root / "data" / "videos"
    if repo_default.exists():
        return str(repo_default)
    
    # If CLI input provided, use it
    if cli_input:
        return cli_input
    
    # Environment variable fallback
    env_dir = os.getenv('VIDEO_INPUT_DIR')
    if env_dir:
        return env_dir
    
    # Create data/videos if it doesn't exist
    os.makedirs(repo_default, exist_ok=True)
    return str(repo_default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Roboflow license plate detection and OCR on video files. Automatically uses data/videos/ as input directory.")
    parser.add_argument("--input", dest="input_path", help="Input video file or directory (optional, defaults to data/videos/)")
    parser.add_argument("--output", dest="output_dir", help="Output directory (defaults to outputs/roboflow_video_<ts>)")
    parser.add_argument("--confidence", type=int, default=40, help="Confidence threshold (default: 40)")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap threshold (default: 30)")
    parser.add_argument("--skip", dest="skip_frames", type=int, default=2, help="Skip frames for processing speed (default: 2, process every 3rd frame)")
    parser.add_argument("--workspace", dest="workspace", help="Roboflow workspace slug")
    parser.add_argument("--project", dest="project", help="Roboflow project slug")
    parser.add_argument("--version", dest="version", type=int, help="Roboflow version number")
    return parser.parse_args()


def main():
    """Main video processing execution"""
    print("🎬 Roboflow Video License Plate Detection & OCR")
    print("=" * 60)
    args = parse_args()

    # Initialize processor
    processor = RoboflowVideoProcessor(
        workspace=args.workspace,
        project_slug=args.project,
        version=args.version,
    )

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = repo_root / "outputs" / f"roboflow_video_{timestamp}"
    
    os.makedirs(output_base, exist_ok=True)
    
    # Determine input path
    if args.input_path:
        input_path = Path(args.input_path)
    else:
        input_dir = resolve_input_dir(None, repo_root)
        input_path = Path(input_dir)
        
        if input_path.is_dir():
            # Find video files in directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            video_files = []
            for ext in video_extensions:
                video_files.extend(input_path.glob(f'*{ext}'))
                video_files.extend(input_path.glob(f'*{ext.upper()}'))
            
            if not video_files:
                print(f"⚠️  No video files found in {input_path}")
                print("   Supported formats: mp4, avi, mov, mkv, wmv, flv, webm")
                return
            
            # Process first video file found
            input_path = video_files[0]
            print(f"📁 Found {len(video_files)} video file(s), processing: {input_path.name}")

    print(f"📁 Output directory: {output_base}")
    
    # Process video
    if input_path.is_file():
        result = processor.process_video(
            str(input_path),
            str(output_base),
            confidence=args.confidence,
            overlap=args.overlap,
            skip_frames=args.skip_frames
        )
        
        if result:
            print(f"\n✅ Processing complete! Check results in {output_base}")
        else:
            print(f"\n⚠️  Processing failed.")
    else:
        print(f"❌ Input path is not a valid file: {input_path}")


if __name__ == "__main__":
    main()