"""
License Plate Reader API Server
FastAPI-based REST API for license plate detection and management
"""

import os
import io
import time
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # JWT disabled
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

try:
    import cv2
    import numpy as np
    from PIL import Image
    # import jwt  # JWT disabled
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("❌ Required dependencies not available. Install with:")
    print("   pip install fastapi uvicorn opencv-python pillow pyjwt")

# Import our LPR components
try:
    from src.offline.standalone_detector import StandaloneLPRDetector
    from src.core.ocr_engine import OCREngine
    from monitoring.metrics_collector import MetricsCollector
    from monitoring.alerts.alert_manager import AlertManager
    from models.model_manager import ModelManager
    from data.data_manager import DataManager
    LPR_COMPONENTS_AVAILABLE = True
except ImportError as e:
    LPR_COMPONENTS_AVAILABLE = False
    print(f"❌ LPR components not available: {e}")

# Import Roboflow components (optional)
try:
    from src.core.detector import LicensePlateDetector
    ROBOFLOW_AVAILABLE = True
    print("✅ Roboflow import OK")
except ImportError as e:
    ROBOFLOW_AVAILABLE = False
    print(f"❌ Roboflow import failed: {e}")

# Pydantic models for request/response validation
from pydantic import BaseModel, Field
from enum import Enum

class DetectionModel(str, Enum):
    """Available detection models"""
    OFFLINE = "offline"
    ROBOFLOW = "roboflow"
    AUTO = "auto"

class DetectionRequest(BaseModel):
    """Request model for detection"""
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    use_roboflow: bool = Field(True, description="Use Roboflow API if available")
    extract_text: bool = Field(True, description="Extract text using OCR")
    return_image: bool = Field(False, description="Return annotated image")

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

class LPRAPIServer:
    """Main API server class"""
    
    def __init__(self):
        """Initialize API server with all components"""
        self.app = FastAPI(
            title="License Plate Reader API",
            description="REST API for license plate detection and recognition",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configuration
        self.config = {
            'jwt_secret': os.getenv('JWT_SECRET', 'default-secret-key'),
            'jwt_expiry_hours': int(os.getenv('JWT_EXPIRY_HOURS', '24')),
            'max_file_size': int(os.getenv('MAX_FILE_SIZE', '10485760')),  # 10MB
            'allowed_extensions': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'},
            'video_extensions': {'.mp4', '.avi', '.mov', '.mkv'}
        }
        
        # Initialize components
        self.initialize_components()
        
        # Setup middleware
        self.setup_middleware()
        
        # Setup routes
        self.setup_routes()
        
        # Background tasks
        self.video_processing_tasks = {}
        
    def initialize_components(self):
        """Initialize all LPR components"""
        if not LPR_COMPONENTS_AVAILABLE:
            raise RuntimeError("LPR components not available")
            
        print("🚀 Initializing LPR API components...")
        
        # Initialize detectors (force CPU mode)
        # Prefer explicit model path via env/config when available
        yolo_model_path = os.getenv('YOLO_MODEL_PATH')
        if not yolo_model_path:
            # Also support selecting by name via DEFAULT_YOLO_MODEL (e.g., 'yolov8x.pt')
            default_model_name = os.getenv('DEFAULT_YOLO_MODEL', '').strip()
            if default_model_name:
                # Search common locations
                candidates = [
                    default_model_name,
                    str(Path('models/pretrained') / default_model_name)
                ]
                for p in candidates:
                    if Path(p).exists():
                        yolo_model_path = p
                        break

        self.offline_detector = StandaloneLPRDetector(
            model_path=yolo_model_path if yolo_model_path else None,
            confidence=0.5,
            device='cpu'
        )
        print("✅ Offline detector initialized")
        
        self.roboflow_detector = None
        print(f"🔍 Debug: ROBOFLOW_AVAILABLE = {ROBOFLOW_AVAILABLE}")
        print(f"🔍 Debug: ROBOFLOW_API_KEY = {bool(os.getenv('ROBOFLOW_API_KEY'))}")
        
        if ROBOFLOW_AVAILABLE and os.getenv('ROBOFLOW_API_KEY'):
            try:
                roboflow_config = {
                    'api_key': os.getenv('ROBOFLOW_API_KEY'),
                    'workspace': os.getenv('ROBOFLOW_WORKSPACE', 'test-aip6t'),
                    'project_id': os.getenv('ROBOFLOW_PROJECT', 'license-plate-recognition-8fvub-hvrra'),
                    'model_version': int(os.getenv('ROBOFLOW_VERSION', '2')),
                    'confidence': int(os.getenv('ROBOFLOW_CONFIDENCE', '40')),
                    'overlap': int(os.getenv('ROBOFLOW_OVERLAP', '30'))
                }
                print(f"🔍 Debug: Roboflow config = {roboflow_config}")
                self.roboflow_detector = LicensePlateDetector(roboflow_config=roboflow_config)
                print("✅ Roboflow detector initialized")
            except Exception as e:
                print(f"⚠️  Roboflow detector failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Roboflow detector skipped - requirements not met")
        
        # Initialize OCR engine (use EasyOCR for better compatibility)
        self.ocr_engine = OCREngine(engine='easyocr')
        print("✅ OCR engine initialized")
        
        # Initialize monitoring
        self.metrics_collector = MetricsCollector(enable_prometheus=True)
        self.metrics_collector.start_system_monitoring(interval=30)
        print("✅ Metrics collector initialized")
        
        # Initialize alert manager
        self.alert_manager = AlertManager()
        self.alert_manager.start_monitoring(self.metrics_collector, check_interval=60)
        print("✅ Alert manager initialized")
        
        # Initialize model manager
        self.model_manager = ModelManager()
        print("✅ Model manager initialized")
        
        # Initialize data manager
        self.data_manager = DataManager()
        print("✅ Data manager initialized")
        
        print("🎉 All components initialized successfully!")
    
    def setup_middleware(self):
        """Setup API middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting and metrics middleware
        @self.app.middleware("http")
        async def add_metrics_middleware(request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            # Record API metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_api_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time=processing_time,
                user_agent=request.headers.get('user-agent')
            )
            
            # Add response headers
            response.headers["X-Process-Time"] = str(processing_time)
            
            return response
    
    def setup_routes(self):
        """Setup all API routes"""
        
        # Health check
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Check API health and service status"""
            services_status = {
                "offline_detector": "healthy" if self.offline_detector else "unavailable",
                "roboflow_detector": "healthy" if self.roboflow_detector else "unavailable", 
                "ocr_engine": "healthy" if self.ocr_engine else "unavailable",
                "metrics": "healthy" if self.metrics_collector else "unavailable",
                "alerts": "healthy" if self.alert_manager else "unavailable"
            }
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                version="1.0.0",
                services=services_status
            )
        
        # JWT Authentication disabled for testing
        # @self.app.post("/auth/login")
        # async def login(username: str = Form(...), password: str = Form(...)):
        #     """Authenticate user and return JWT token"""
        #     # Simple authentication - replace with proper auth in production
        #     if username == "admin" and password == os.getenv("ADMIN_PASSWORD", "admin123"):
        #         payload = {
        #             "username": username,
        #             "exp": datetime.utcnow() + timedelta(hours=self.config['jwt_expiry_hours'])
        #         }
        #         token = jwt.encode(payload, self.config['jwt_secret'], algorithm="HS256")
        #         
        #         return {
        #             "access_token": token,
        #             "token_type": "bearer",
        #             "expires_in": self.config['jwt_expiry_hours'] * 3600
        #         }
        #     else:
        #         raise HTTPException(
        #             status_code=status.HTTP_401_UNAUTHORIZED,
        #             detail="Invalid credentials"
        #         )
        
        # Authentication dependency - DISABLED
        # security = HTTPBearer()
        
        # def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        #     """Verify JWT token"""
        #     try:
        #         payload = jwt.decode(credentials.credentials, self.config['jwt_secret'], algorithms=["HS256"])
        #         return payload.get("username")
        #     except jwt.ExpiredSignatureError:
        #         raise HTTPException(
        #             status_code=status.HTTP_401_UNAUTHORIZED,
        #             detail="Token expired"
        #         )
        #     except jwt.JWTError:
        #         raise HTTPException(
        #             status_code=status.HTTP_401_UNAUTHORIZED,
        #             detail="Invalid token"
        #         )
        
        # Mock verify function for no auth
        def verify_token():
            """Mock verify function - no authentication required"""
            return "test_user"
        
        # Detection endpoints
        @self.app.post("/detect/image", response_model=DetectionResponse)
        async def detect_image(
            file: UploadFile = File(...),
            confidence: float = Form(0.5),
            use_roboflow: bool = Form(True),
            extract_text: bool = Form(True),
            return_image: bool = Form(False),
            # current_user: str = Depends(verify_token)  # Auth disabled
        ):
            """Detect license plates in uploaded image"""
            
            # Validate file
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must be an image"
                )
            
            # Check file size
            contents = await file.read()
            if len(contents) > self.config['max_file_size']:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Max size: {self.config['max_file_size']} bytes"
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
                
                # Choose detector
                detector = None
                model_used = "unknown"
                
                if use_roboflow and self.roboflow_detector:
                    detector = self.roboflow_detector
                    model_used = "roboflow"
                    detections = detector.detect(image, use_roboflow=True)
                else:
                    detector = self.offline_detector
                    model_used = "offline"
                    detections = detector.detect(image)
                
                processing_time = time.time() - start_time
                
                # Extract text if requested
                texts = []
                if extract_text and detections:
                    if model_used == "roboflow":
                        # Extract plate regions from Roboflow detections
                        from PIL import Image
                        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        plate_regions = []
                        
                        for detection in detections:
                            # Convert Roboflow center coordinates to crop coordinates
                            x_center = detection['bbox'][0] + detection['bbox'][2] / 2
                            y_center = detection['bbox'][1] + detection['bbox'][3] / 2
                            width = detection['bbox'][2] 
                            height = detection['bbox'][3]
                            
                            left = x_center - (width / 2)
                            top = y_center - (height / 2)
                            right = x_center + (width / 2)
                            bottom = y_center + (height / 2)
                            
                            # Crop plate region
                            cropped = pil_image.crop((left, top, right, bottom))
                            plate_regions.append(np.array(cropped))
                    else:
                        plate_regions = self.offline_detector.extract_plate_regions(image, detections)
                    
                    if model_used == "roboflow":
                        # For Roboflow, plate_regions are direct numpy arrays
                        for plate_image in plate_regions:
                            ocr_result = self.ocr_engine.extract_text(plate_image)
                            texts.append(ocr_result)
                    else:
                        # For offline detector, plate_regions have 'image' key
                        for plate_data in plate_regions:
                            ocr_result = self.ocr_engine.extract_text(plate_data['image'])
                            texts.append(ocr_result)
                
                # Add text results to detections
                for i, detection in enumerate(detections):
                    if i < len(texts):
                        detection['text'] = texts[i]['text']
                        detection['text_confidence'] = texts[i]['confidence']
                
                # Record metrics
                confidence_avg = np.mean([d['confidence'] for d in detections]) if detections else 0
                self.metrics_collector.record_detection(
                    image_size=image.shape[:2],
                    detections_count=len(detections),
                    processing_time=processing_time,
                    confidence_avg=confidence_avg,
                    model_used=model_used,
                    success=True
                )
                
                response_data = {
                    "success": True,
                    "detections": detections,
                    "processing_time": processing_time,
                    "image_size": [image.shape[1], image.shape[0]],  # width, height
                    "model_info": {
                        "detector": model_used,
                        "ocr_engine": "paddleocr" if extract_text else "none"
                    }
                }
                
                # Add annotated image if requested
                if return_image:
                    # This would need implementation to return base64 encoded image
                    response_data["annotated_image"] = "base64_encoded_image_here"
                
                return DetectionResponse(**response_data)
                
            except Exception as e:
                # Record failed detection
                self.metrics_collector.record_detection(
                    image_size=(0, 0),
                    detections_count=0,
                    processing_time=time.time() - start_time,
                    confidence_avg=0,
                    model_used=model_used,
                    success=False,
                    error_message=str(e)
                )
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Detection failed: {str(e)}"
                )
        
        @self.app.post("/detect/batch")
        async def detect_batch(
            files: List[UploadFile] = File(...),
            confidence: float = Form(0.5),
            use_roboflow: bool = Form(True),
            extract_text: bool = Form(True),
            # current_user: str = Depends(verify_token)  # Auth disabled
        ):
            """Process multiple images in batch"""
            
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
                    # Process each file (similar to single image detection)
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
                    
                    # Detect (simplified - use offline for batch)
                    detections = self.offline_detector.detect(image)
                    processing_time = time.time() - start_time
                    
                    # Extract text if requested
                    if extract_text and detections:
                        plate_regions = self.offline_detector.extract_plate_regions(image, detections)
                        for i, plate_data in enumerate(plate_regions):
                            if i < len(detections):
                                ocr_result = self.ocr_engine.extract_text(plate_data['image'])
                                detections[i]['text'] = ocr_result['text']
                                detections[i]['text_confidence'] = ocr_result['confidence']
                    
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "detections": detections,
                        "processing_time": processing_time
                    })
                    
                    total_processing_time += processing_time
                    total_detections += len(detections)
                    
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
                "total_detections": total_detections
            }
        
        @self.app.post("/detect/video")
        async def detect_video(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            frame_skip: int = Form(1),
            max_frames: Optional[int] = Form(None),
            confidence: float = Form(0.5),
            # current_user: str = Depends(verify_token)  # Auth disabled
        ):
            """Process video file for license plate detection"""
            
            # Validate video file
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.config['video_extensions']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported video format. Supported: {self.config['video_extensions']}"
                )
            
            # Generate job ID
            job_id = f"video_job_{uuid.uuid4().hex[:8]}"
            
            # Save video file temporarily
            temp_video_path = f"/tmp/{job_id}_{file.filename}"
            
            contents = await file.read()
            with open(temp_video_path, 'wb') as f:
                f.write(contents)
            
            # Add to background tasks
            background_tasks.add_task(
                self.process_video_background,
                job_id,
                temp_video_path,
                frame_skip,
                max_frames,
                confidence
            )
            
            # Store job info
            self.video_processing_tasks[job_id] = {
                "status": "processing",
                "created": datetime.now(),
                "progress": 0,
                "results": None
            }
            
            return {
                "success": True,
                "job_id": job_id,
                "status": "processing",
                "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
            }
        
        @self.app.get("/detect/video/{job_id}/status")
        async def get_video_status(
            job_id: str,
            # current_user: str = Depends(verify_token)  # Auth disabled
        ):
            """Get video processing job status"""
            
            if job_id not in self.video_processing_tasks:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Job not found"
                )
            
            job_info = self.video_processing_tasks[job_id]
            
            return {
                "job_id": job_id,
                "status": job_info["status"],
                "progress": job_info["progress"],
                "results": job_info.get("results"),
                "created": job_info["created"].isoformat()
            }
        
        # Data management endpoints
        @self.app.get("/data/datasets")
        async def list_datasets():  # Auth disabled
            """Get list of available datasets"""
            stats = self.data_manager.get_dataset_statistics()
            
            datasets = []
            for name, dataset_stats in stats['datasets'].items():
                datasets.append({
                    "name": name,
                    "images": dataset_stats['images'],
                    "size": f"{dataset_stats['total_size'] / (1024*1024):.1f}MB",
                    "sources": dataset_stats['sources']
                })
            
            return {
                "datasets": datasets,
                "total_datasets": stats['total_datasets']
            }
        
        @self.app.post("/data/upload")
        async def upload_dataset(
            files: List[UploadFile] = File(...),
            dataset_name: str = Form(...),
            source: str = Form("api_upload"),
            # current_user: str = Depends(verify_token)  # Auth disabled
        ):
            """Upload images to create or extend a dataset"""
            
            # Save uploaded files temporarily
            temp_files = []
            for file in files:
                if not file.content_type.startswith('image/'):
                    continue
                
                temp_path = f"/tmp/upload_{uuid.uuid4().hex}_{file.filename}"
                contents = await file.read()
                
                with open(temp_path, 'wb') as f:
                    f.write(contents)
                
                temp_files.append(temp_path)
            
            # Add to data manager
            result = self.data_manager.add_raw_images(
                image_paths=temp_files,
                dataset_name=dataset_name,
                source=source
            )
            
            # Cleanup temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "added": result['added'],
                "skipped": result['skipped'],
                "errors": result['errors']
            }
        
        # Model management endpoints
        @self.app.get("/models")
        async def list_models():  # Auth disabled
            """Get list of available models"""
            models = self.model_manager.list_models()
            
            return {
                "models": [
                    {
                        "name": model["name"],
                        "type": model["type"],
                        "size_mb": round(model["size_mb"], 1),
                        "status": "loaded" if model["exists"] else "available"
                    }
                    for model in models
                ]
            }
        
        # Analytics endpoints
        @self.app.get("/analytics/stats")
        async def get_stats(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            # current_user: str = Depends(verify_token)  # Auth disabled
        ):
            """Get detection statistics and analytics"""
            
            # Get metrics summary
            summary = self.metrics_collector.get_metrics_summary(last_minutes=60)
            
            return {
                "period": {
                    "start": start_date or (datetime.now() - timedelta(hours=1)).isoformat(),
                    "end": end_date or datetime.now().isoformat()
                },
                "statistics": {
                    "total_detections": summary.get('detection', {}).get('total_requests', 0),
                    "success_rate": summary.get('detection', {}).get('success_rate', 0),
                    "avg_processing_time": summary.get('detection', {}).get('avg_processing_time', 0),
                    "avg_confidence": summary.get('detection', {}).get('avg_confidence', 0)
                },
                "system": {
                    "cpu_usage": summary.get('system', {}).get('avg_cpu_percent', 0),
                    "memory_usage": summary.get('system', {}).get('avg_memory_percent', 0)
                }
            }
        
        # Configuration endpoints
        @self.app.get("/config")
        async def get_config():  # Auth disabled
            """Get current system configuration"""
            return {
                "detection": {
                    "confidence_threshold": 0.5,
                    "max_detections": 100
                },
                "ocr": {
                    "engine": "paddleocr",
                    "confidence_threshold": 0.8
                },
                "roboflow": {
                    "enabled": bool(self.roboflow_detector),
                    "fallback_enabled": True
                }
            }
        
        # Metrics endpoints
        @self.app.get("/metrics")
        async def get_prometheus_metrics():
            """Get Prometheus metrics (for monitoring)"""
            return self.metrics_collector.export_prometheus_metrics()
        
        # WebSocket endpoint for real-time detection
        @self.app.websocket("/ws/detect")
        async def websocket_detect(websocket: WebSocket):
            """WebSocket endpoint for real-time detection"""
            await websocket.accept()
            
            try:
                while True:
                    # Receive image data
                    data = await websocket.receive_json()
                    
                    if data.get('type') == 'detection':
                        # Process base64 image
                        image_data = data['data']['image']
                        confidence = data['data'].get('confidence', 0.5)
                        
                        # Decode base64 image
                        import base64
                        image_bytes = base64.b64decode(image_data)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            start_time = time.time()
                            detections = self.offline_detector.detect(image)
                            processing_time = time.time() - start_time
                            
                            # Send results back
                            await websocket.send_json({
                                "type": "detection_result",
                                "data": {
                                    "detections": detections,
                                    "processing_time": processing_time
                                }
                            })
                        
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                await websocket.close()
    
    async def process_video_background(self, 
                                     job_id: str, 
                                     video_path: str,
                                     frame_skip: int,
                                     max_frames: Optional[int],
                                     confidence: float):
        """Background task for video processing"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            results = {
                "total_frames": total_frames,
                "processed_frames": 0,
                "total_detections": 0,
                "unique_plates": set(),
                "detections_by_frame": []
            }
            
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                detections = self.offline_detector.detect(frame)
                
                if detections:
                    # Extract text
                    plate_regions = self.offline_detector.extract_plate_regions(frame, detections)
                    frame_texts = []
                    
                    for plate_data in plate_regions:
                        ocr_result = self.ocr_engine.extract_text(plate_data['image'])
                        if ocr_result['text'] and ocr_result['confidence'] > 0.7:
                            frame_texts.append(ocr_result['text'])
                            results['unique_plates'].add(ocr_result['text'])
                    
                    results['detections_by_frame'].append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / 30.0,  # Assume 30 FPS
                        'detections': len(detections),
                        'texts': frame_texts
                    })
                    
                    results['total_detections'] += len(detections)
                
                processed_count += 1
                frame_count += 1
                
                # Update progress
                progress = min(100, (processed_count / (total_frames // frame_skip)) * 100)
                self.video_processing_tasks[job_id]['progress'] = progress
                
                # Check limits
                if max_frames and processed_count >= max_frames:
                    break
            
            cap.release()
            
            # Convert set to list for JSON serialization
            results['unique_plates'] = list(results['unique_plates'])
            results['processed_frames'] = processed_count
            
            # Update job status
            self.video_processing_tasks[job_id].update({
                "status": "completed",
                "progress": 100,
                "results": results
            })
            
        except Exception as e:
            self.video_processing_tasks[job_id].update({
                "status": "failed",
                "error": str(e)
            })
        
        finally:
            # Cleanup temp file
            try:
                os.unlink(video_path)
            except:
                pass
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the API server"""
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


# Global app instance for uvicorn
server = LPRAPIServer()
app = server.app

def main():
    """Run API server"""
    print("🚀 Starting License Plate Reader API Server")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Dependencies not available")
        return
    
    if not LPR_COMPONENTS_AVAILABLE:
        print("❌ LPR components not available")
        return
    
    # Run server
    server.run(
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '8000')),
        reload=os.getenv('API_RELOAD', 'false').lower() == 'true'
    )

if __name__ == "__main__":
    main()
