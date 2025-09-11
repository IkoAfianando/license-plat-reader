"""
License Plate Detection Engine using YOLOv8
Core detection functionality with Roboflow integration
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
    from roboflow import Roboflow
except ImportError as e:
    logging.error(f"Required libraries not installed: {e}")
    raise

logger = logging.getLogger(__name__)

class LicensePlateDetector:
    """Main detection engine using YOLOv8 and Roboflow"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 roboflow_config: Optional[Dict] = None,
                 confidence: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize the detector
        
        Args:
            model_path: Path to local YOLO model file
            roboflow_config: Roboflow API configuration
            confidence: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
        """
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.model = None
        self.roboflow_model = None
        
        # Initialize models
        if model_path and Path(model_path).exists():
            self._load_local_model(model_path)
        
        if roboflow_config:
            self._load_roboflow_model(roboflow_config)
            
        if not self.model and not self.roboflow_model:
            logger.warning("No model loaded. Using YOLOv8n as fallback.")
            self._load_local_model('yolov8n.pt')  # Will download if not exists
    
    def _load_local_model(self, model_path: str):
        """Load local YOLOv8 model"""
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded local model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load local model {model_path}: {e}")
            raise
    
    def _load_roboflow_model(self, config: Dict):
        """Load Roboflow model"""
        try:
            rf = Roboflow(api_key=config['api_key'])
            workspace_name = config.get('workspace', 'test-aip6t')
            project_id = config.get('project_id', 'license-plate-recognition-8fvub-hvrra')
            model_version = config.get('model_version', 2)
            
            project = rf.workspace(workspace_name).project(project_id)
            self.roboflow_model = project.version(model_version).model
            self.roboflow_confidence = config.get('confidence', 40)
            self.roboflow_overlap = config.get('overlap', 30)
            logger.info(f"Loaded Roboflow model: {workspace_name}/{project_id}/v{model_version}")
        except Exception as e:
            logger.error(f"Failed to load Roboflow model: {e}")
            # Continue without Roboflow model
    
    def detect(self, 
               image: Union[str, np.ndarray], 
               use_roboflow: bool = True) -> List[Dict]:
        """
        Detect license plates in image
        
        Args:
            image: Image path or numpy array
            use_roboflow: Prefer Roboflow model if available
            
        Returns:
            List of detection dictionaries with bbox, confidence, etc.
        """
        detections = []
        
        # Try Roboflow first if requested and available
        if use_roboflow and self.roboflow_model:
            try:
                detections = self._detect_roboflow(image)
                if detections:
                    logger.debug(f"Roboflow detected {len(detections)} plates")
                    return detections
            except Exception as e:
                logger.warning(f"Roboflow detection failed: {e}")
        
        # Fallback to local model
        if self.model:
            try:
                detections = self._detect_local(image)
                logger.debug(f"Local model detected {len(detections)} plates")
            except Exception as e:
                logger.error(f"Local detection failed: {e}")
        
        return detections
    
    def _detect_roboflow(self, image: Union[str, np.ndarray]) -> List[Dict]:
        """Detect using Roboflow API"""
        # Convert numpy array to temp file if needed
        if isinstance(image, np.ndarray):
            temp_path = "/tmp/temp_detection.jpg"
            cv2.imwrite(temp_path, image)
            image_path = temp_path
        else:
            image_path = image
        
        # Run detection
        result = self.roboflow_model.predict(
            image_path, 
            confidence=self.roboflow_confidence,
            overlap=self.roboflow_overlap
        )
        
        # Parse results
        detections = []
        for pred in result.json().get('predictions', []):
            detection = {
                'bbox': [
                    pred['x'] - pred['width']/2,   # x1
                    pred['y'] - pred['height']/2,  # y1  
                    pred['x'] + pred['width']/2,   # x2
                    pred['y'] + pred['height']/2   # y2
                ],
                'confidence': pred['confidence'],
                'class': pred['class'],
                'source': 'roboflow'
            }
            detections.append(detection)
        
        return detections
    
    def _detect_local(self, image: Union[str, np.ndarray]) -> List[Dict]:
        """Detect using local YOLOv8 model"""
        results = self.model(image, conf=self.confidence, iou=self.nms_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Convert to CPU and numpy
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                        'confidence': conf,
                        'class': cls,
                        'source': 'local'
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_video_stream(self, 
                           video_source: Union[str, int],
                           output_callback=None,
                           max_frames: Optional[int] = None) -> None:
        """
        Process video stream for real-time detection
        
        Args:
            video_source: Video file path or camera index
            output_callback: Function to call with each detection result
            max_frames: Maximum frames to process (None for unlimited)
        """
        cap = cv2.VideoCapture(video_source)
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                detections = self.detect(frame)
                
                # Call callback if provided
                if output_callback:
                    output_callback(frame, detections, frame_count)
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
                    
        except KeyboardInterrupt:
            logger.info("Video processing interrupted")
        finally:
            cap.release()
    
    def extract_plate_regions(self, 
                            image: np.ndarray, 
                            detections: List[Dict],
                            padding: int = 10) -> List[np.ndarray]:
        """
        Extract license plate regions from image
        
        Args:
            image: Original image
            detections: Detection results
            padding: Padding around bounding box
            
        Returns:
            List of cropped plate images
        """
        plates = []
        h, w = image.shape[:2]
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding) 
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract region
            plate_region = image[y1:y2, x1:x2]
            if plate_region.size > 0:
                plates.append(plate_region)
        
        return plates
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'local_model': bool(self.model),
            'roboflow_model': bool(self.roboflow_model),
            'confidence_threshold': self.confidence,
            'nms_threshold': self.nms_threshold
        }
        
        if self.model:
            info['local_model_path'] = str(self.model.ckpt_path)
            
        if self.roboflow_model:
            info['roboflow_confidence'] = self.roboflow_confidence
            info['roboflow_overlap'] = self.roboflow_overlap
            
        return info