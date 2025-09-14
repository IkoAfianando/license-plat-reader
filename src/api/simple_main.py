"""
Simplified License Plate Reader API Server
FastAPI-based REST API without heavy model dependencies for initial testing
"""

import os
import io
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import cv2
    import numpy as np
    from PIL import Image
    import base64
    BASIC_DEPS_AVAILABLE = True
    print("✅ Basic dependencies (OpenCV, PIL, base64) loaded successfully")
except ImportError as e:
    BASIC_DEPS_AVAILABLE = False
    print(f"❌ Basic dependencies missing: {e}")

# Import visualization utility
try:
    from src.utils.visualization import draw_detections
    VISUALIZATION_AVAILABLE = True
    print("✅ Visualization utilities loaded")
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"❌ Visualization utilities not available: {e}")

class DetectionResponse(BaseModel):
    """Response model for detection"""
    success: bool
    detections: List[Dict[str, Any]]
    processing_time: float
    image_size: List[int]
    model_info: Dict[str, Any]  # Changed from Dict[str, str] to Dict[str, Any]
    message: Optional[str] = None
    annotated_image: Optional[str] = None  # Base64 encoded image

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# Create FastAPI app
app = FastAPI(
    title="License Plate Reader API (Simplified)",
    description="Simplified REST API for license plate detection testing",
    version="1.0.0-simple",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import pre-trained detector
try:
    from src.core.pretrained_detector import PretrainedLicensePlateDetector, get_available_models
    PRETRAINED_AVAILABLE = True
    print("✅ Pre-trained detector available")
except ImportError as e:
    PRETRAINED_AVAILABLE = False
    print(f"❌ Pre-trained detector not available: {e}")

# Smart detector that uses best available option
class SmartLicensePlateDetector:
    """Smart detector that uses the best available pre-trained model"""
    
    def __init__(self, preferred_model='yolov8_general'):
        self.detector = None
        self.model_info = {}
        
        if PRETRAINED_AVAILABLE:
            try:
                self.detector = PretrainedLicensePlateDetector(preferred_model, confidence=0.25)
                self.model_info = self.detector.get_model_info()
                self.model_info['type'] = 'pretrained_detector'
                print(f"✅ Using pre-trained model: {preferred_model}")
            except Exception as e:
                print(f"⚠️ Pre-trained model failed, using mock: {e}")
                self._use_mock_detector()
        else:
            self._use_mock_detector()
    
    def _use_mock_detector(self):
        """Fallback to mock detector"""
        self.detector = None
        self.model_info = {
            'type': 'mock_detector',
            'name': 'Mock License Plate Detector',
            'description': 'Fallback mock detector for testing',
            'confidence_threshold': 0.3,
            'note': 'Using mock detector - install ultralytics for real detection'
        }
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect license plates using best available method"""
        if self.detector:
            # Use real pre-trained detector
            return self.detector.detect(image)
        else:
            # Use mock detector
            return self._mock_detect(image)
    
    def _mock_detect(self, image: np.ndarray) -> List[Dict]:
        """Mock detection for fallback"""
        h, w = image.shape[:2]
        
        # More realistic mock detections
        detections = []
        
        # Simulate finding 1-2 license plates
        import random
        num_plates = random.choice([1, 1, 2])  # Usually 1 plate, sometimes 2
        
        for i in range(num_plates):
            # Random but realistic positions (bottom half of image)
            x1 = random.uniform(0.1, 0.6) * w
            y1 = random.uniform(0.6, 0.8) * h  
            plate_width = random.uniform(0.15, 0.25) * w
            plate_height = random.uniform(0.05, 0.08) * h
            
            x2 = min(w, x1 + plate_width)
            y2 = min(h, y1 + plate_height)
            
            # Sample license plate texts
            sample_texts = ['B1234ABC', 'D5678XYZ', 'AA123BB', 'L789MNO', 'R456DEF']
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': random.uniform(0.7, 0.95),
                'class': 'license_plate',
                'source': 'mock_detector',
                'text': random.choice(sample_texts),
                'text_confidence': random.uniform(0.8, 0.95)
            })
        
        return detections
    
    def get_model_info(self):
        """Get model information"""
        return self.model_info
    
    def set_confidence(self, confidence: float):
        """Set confidence threshold"""
        if self.detector and hasattr(self.detector, 'set_confidence'):
            self.detector.set_confidence(confidence)
        # Update local info
        self.model_info['confidence_threshold'] = confidence

# Try to use specific license plate model, fallback to general if not available
try:
    # Use specific pre-trained license plate model
    detector = SmartLicensePlateDetector('yolov8_license_plates')
    print("✅ Using specific license plate detection model")
except:
    # Fallback to general model
    detector = SmartLicensePlateDetector('yolov8_general')
    print("⚠️ Using general model as fallback")

def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and service status"""
    services_status = {
        "detector": "healthy",
        "basic_deps": "healthy" if BASIC_DEPS_AVAILABLE else "unavailable",
        "api": "healthy"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0-simple",
        services=services_status
    )

@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    extract_text: bool = Form(True),
    return_image: bool = Form(False)
):
    """Detect license plates in uploaded image (mock implementation)"""
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Check file size (max 10MB)
    contents = await file.read()
    max_size = 10 * 1024 * 1024  # 10MB
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {max_size} bytes"
        )
    
    try:
        # Convert to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        start_time = time.time()
        
        # Run mock detection
        detections = detector.detect(image)
        
        # Filter by confidence
        filtered_detections = [
            d for d in detections if d['confidence'] >= confidence
        ]
        
        processing_time = time.time() - start_time
        
        # Get model info and format for frontend compatibility
        model_info = detector.get_model_info()
        
        response_data = {
            "success": True,
            "detections": filtered_detections,
            "processing_time": processing_time,
            "image_size": [image.shape[1], image.shape[0]],  # width, height
            "model_info": {
                "detector": model_info.get('name', 'Pre-trained YOLO'),
                "ocr_engine": "integrated",  # Pre-trained model has integrated text detection
                **model_info  # Include all other model info
            },
            "message": f"Detection completed using {model_info.get('name', 'Pre-trained model')}. Found {len(filtered_detections)} license plates." if filtered_detections else f"No license plates detected. Using {model_info.get('name', 'Pre-trained model')}."
        }
        
        # Add annotated image if requested
        if return_image and VISUALIZATION_AVAILABLE and filtered_detections:
            try:
                annotated_image = draw_detections(image, filtered_detections, color=(0, 255, 0))
                response_data["annotated_image"] = image_to_base64(annotated_image)
            except Exception as e:
                response_data["message"] += f" (Note: Could not generate annotated image: {e})"
        elif return_image and not VISUALIZATION_AVAILABLE:
            response_data["message"] += " (Note: Visualization utilities not available for image annotation)"
        elif return_image and not filtered_detections:
            response_data["message"] += " (Note: No detections to annotate)"
        
        return DetectionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )

@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Form(0.5)
):
    """Process multiple images in batch (mock implementation)"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files per batch"
        )
    
    results = []
    total_processing_time = 0
    total_detections = 0
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image format"
                })
                continue
            
            start_time = time.time()
            
            # Mock detection
            detections = detector.detect(image)
            filtered_detections = [
                d for d in detections if d['confidence'] >= confidence
            ]
            
            processing_time = time.time() - start_time
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": filtered_detections,
                "processing_time": processing_time
            })
            
            total_processing_time += processing_time
            total_detections += len(filtered_detections)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "results": results,
        "total_processing_time": total_processing_time,
        "total_detections": total_detections,
        "message": "Mock batch processing completed"
    }

@app.get("/config")
async def get_config():
    """Get current system configuration"""
    model_info = detector.get_model_info()
    return {
        "detection": {
            "confidence_threshold": model_info.get('confidence_threshold', 0.3),
            "max_detections": 100,
            "model_type": model_info.get('type', 'unknown'),
            "model_name": model_info.get('name', 'Unknown Model'),
            "model_description": model_info.get('description', 'No description')
        },
        "api": {
            "version": "1.0.0-enhanced",
            "max_file_size": "10MB",
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"]
        },
        "models": {
            "current_model": model_info,
            "pretrained_available": PRETRAINED_AVAILABLE
        }
    }

@app.get("/models/available") 
def list_available_models():
    """Get list of available pre-trained models"""
    if PRETRAINED_AVAILABLE:
        try:
            available_models = get_available_models()
            return {
                "success": True,
                "models": available_models,
                "current_model": detector.get_model_info()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "models": {},
                "current_model": detector.get_model_info()
            }
    else:
        return {
            "success": False,
            "error": "Pre-trained models not available",
            "models": {
                "mock_detector": {
                    "name": "Mock Detector",
                    "description": "Fallback detector for testing",
                    "recommended": True,
                    "speed": "Fast",
                    "accuracy": "Mock"
                }
            },
            "current_model": detector.get_model_info()
        }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "License Plate Reader API (Simplified)",
        "version": "1.0.0-simple",
        "docs": "/docs",
        "health": "/health",
        "status": "running",
        "note": "This is a simplified version for testing API functionality"
    }

def main():
    """Run simplified API server"""
    print("🚀 Starting Simplified License Plate Reader API")
    print("=" * 60)
    print("📝 Note: This is a simplified version with mock detector")
    print("🔗 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    
    if not BASIC_DEPS_AVAILABLE:
        print("⚠️  Warning: Some dependencies missing, but API will still work")
    
    # Run server
    uvicorn.run(
        "src.api.simple_main:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '8000')),
        reload=os.getenv('API_RELOAD', 'true').lower() == 'true',
        log_level="info"
    )

if __name__ == "__main__":
    main()