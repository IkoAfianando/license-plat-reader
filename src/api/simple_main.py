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
    BASIC_DEPS_AVAILABLE = True
    print("✅ Basic dependencies (OpenCV, PIL) loaded successfully")
except ImportError as e:
    BASIC_DEPS_AVAILABLE = False
    print(f"❌ Basic dependencies missing: {e}")

class DetectionResponse(BaseModel):
    """Response model for detection"""
    success: bool
    detections: List[Dict[str, Any]]
    processing_time: float
    image_size: List[int]
    model_info: Dict[str, str]
    message: Optional[str] = None

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

# Mock detector for testing
class MockDetector:
    """Mock detector for testing API without actual model"""
    
    def __init__(self):
        self.model_info = {
            'type': 'mock_detector',
            'loaded': 'true',
            'note': 'This is a mock detector for API testing'
        }
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Mock detection that returns sample results"""
        h, w = image.shape[:2]
        
        # Sample detections (mock)
        mock_detections = [
            {
                'bbox': [w*0.2, h*0.6, w*0.5, h*0.8],  # [x1, y1, x2, y2]
                'confidence': 0.85,
                'class': 'license_plate',
                'source': 'mock_detector',
                'text': 'B1234XYZ',
                'text_confidence': 0.9
            },
            {
                'bbox': [w*0.6, h*0.4, w*0.9, h*0.6],
                'confidence': 0.72,
                'class': 'license_plate', 
                'source': 'mock_detector',
                'text': 'D5678ABC',
                'text_confidence': 0.87
            }
        ]
        
        return mock_detections
    
    def get_model_info(self):
        return self.model_info

# Initialize mock detector
detector = MockDetector()

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
        
        response_data = {
            "success": True,
            "detections": filtered_detections,
            "processing_time": processing_time,
            "image_size": [image.shape[1], image.shape[0]],  # width, height
            "model_info": detector.get_model_info(),
            "message": f"Mock detection completed. Found {len(filtered_detections)} plates."
        }
        
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
    return {
        "detection": {
            "confidence_threshold": 0.5,
            "max_detections": 100,
            "model_type": "mock"
        },
        "api": {
            "version": "1.0.0-simple",
            "max_file_size": "10MB",
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"]
        },
        "roboflow": {
            "enabled": False,
            "note": "Using mock detector for testing"
        }
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