"""
Pydantic schemas for License Plate Reader API
Data validation and serialization models for detection endpoints
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import re

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import conint, confloat, constr

class DetectionModelType(str, Enum):
    """Available detection model types"""
    OFFLINE = "offline"
    ROBOFLOW = "roboflow" 
    AUTO = "auto"

class OCREngineType(str, Enum):
    """Available OCR engine types"""
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"

class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"

class VideoFormat(str, Enum):
    """Supported video formats"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"

class RegionCode(str, Enum):
    """Supported license plate regions"""
    US = "US"
    EU = "EU"
    GLOBAL = "GLOBAL"

# Base request/response models
class BaseRequest(BaseModel):
    """Base request model with common fields"""
    
    class Config:
        # Enable extra fields for extensibility
        extra = "forbid"
        # Use enum values instead of names
        use_enum_values = True

class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        use_enum_values = True

# Detection-specific models
class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Left edge coordinate")
    y1: float = Field(..., description="Top edge coordinate") 
    x2: float = Field(..., description="Right edge coordinate")
    y2: float = Field(..., description="Bottom edge coordinate")
    
    @validator('x2')
    def x2_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')
    def y2_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v
    
    def width(self) -> float:
        """Calculate bounding box width"""
        return self.x2 - self.x1
    
    def height(self) -> float:
        """Calculate bounding box height"""
        return self.y2 - self.y1
    
    def area(self) -> float:
        """Calculate bounding box area"""
        return self.width() * self.height()
    
    def center(self) -> tuple:
        """Calculate bounding box center point"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

class OCRResult(BaseModel):
    """OCR text extraction result"""
    text: str = Field(..., description="Extracted text")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="OCR confidence score")
    engine: OCREngineType = Field(..., description="OCR engine used")
    format_valid: bool = Field(default=False, description="Whether text matches regional format")
    region: Optional[RegionCode] = Field(None, description="Detected region")
    preprocessing_applied: bool = Field(default=True, description="Whether image preprocessing was applied")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('OCR text cannot be empty')
        return v.strip().upper()

class Detection(BaseModel):
    """Single license plate detection result"""
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Detection confidence score")
    plate_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="License plate likelihood score")
    class_id: int = Field(default=0, description="Object class ID")
    class_name: str = Field(default="license_plate", description="Object class name")
    source: str = Field(..., description="Detection source (offline/roboflow)")
    ocr_result: Optional[OCRResult] = Field(None, description="OCR text extraction result")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional detection metadata")

class ImageInfo(BaseModel):
    """Image metadata information"""
    width: conint(gt=0) = Field(..., description="Image width in pixels")
    height: conint(gt=0) = Field(..., description="Image height in pixels") 
    channels: conint(ge=1, le=4) = Field(default=3, description="Number of image channels")
    format: Optional[ImageFormat] = Field(None, description="Image format")
    size_bytes: Optional[conint(ge=0)] = Field(None, description="File size in bytes")
    
    def aspect_ratio(self) -> float:
        """Calculate image aspect ratio"""
        return self.width / self.height
    
    def megapixels(self) -> float:
        """Calculate image megapixels"""
        return (self.width * self.height) / 1_000_000

class ModelInfo(BaseModel):
    """Detection model information"""
    detector: DetectionModelType = Field(..., description="Detection model used")
    ocr_engine: Optional[OCREngineType] = Field(None, description="OCR engine used")
    model_version: Optional[str] = Field(None, description="Model version")
    confidence_threshold: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence threshold used")
    device: str = Field(default="cpu", description="Compute device used")

# Request models
class SingleImageDetectionRequest(BaseRequest):
    """Request for single image detection"""
    confidence: confloat(ge=0.0, le=1.0) = Field(
        default=0.5, 
        description="Detection confidence threshold"
    )
    model: DetectionModelType = Field(
        default=DetectionModelType.AUTO,
        description="Detection model to use"
    )
    extract_text: bool = Field(
        default=True,
        description="Whether to extract text using OCR"
    )
    ocr_engine: OCREngineType = Field(
        default=OCREngineType.PADDLEOCR,
        description="OCR engine to use"
    )
    return_annotated_image: bool = Field(
        default=False,
        description="Whether to return annotated image"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="License plate region for format validation"
    )
    apply_nms: bool = Field(
        default=True,
        description="Apply non-maximum suppression to filter overlapping detections"
    )
    max_detections: conint(ge=1, le=100) = Field(
        default=10,
        description="Maximum number of detections to return"
    )

class BatchDetectionRequest(BaseRequest):
    """Request for batch image detection"""
    confidence: confloat(ge=0.0, le=1.0) = Field(default=0.5)
    model: DetectionModelType = Field(default=DetectionModelType.OFFLINE)  # Use offline for batch
    extract_text: bool = Field(default=True)
    ocr_engine: OCREngineType = Field(default=OCREngineType.PADDLEOCR)
    region: RegionCode = Field(default=RegionCode.GLOBAL)
    max_files: conint(ge=1, le=50) = Field(
        default=10,
        description="Maximum number of files to process"
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop processing on first error"
    )

class VideoDetectionRequest(BaseRequest):
    """Request for video detection"""
    confidence: confloat(ge=0.0, le=1.0) = Field(default=0.5)
    model: DetectionModelType = Field(default=DetectionModelType.OFFLINE)  # Use offline for video
    extract_text: bool = Field(default=True)
    ocr_engine: OCREngineType = Field(default=OCREngineType.PADDLEOCR)
    frame_skip: conint(ge=1, le=60) = Field(
        default=5,
        description="Process every N frames"
    )
    max_frames: Optional[conint(ge=1)] = Field(
        None,
        description="Maximum number of frames to process"
    )
    output_video: bool = Field(
        default=False,
        description="Generate annotated output video"
    )
    region: RegionCode = Field(default=RegionCode.GLOBAL)

# Response models
class SingleImageDetectionResponse(BaseResponse):
    """Response for single image detection"""
    detections: List[Detection] = Field(..., description="Detected license plates")
    image_info: ImageInfo = Field(..., description="Input image information")
    model_info: ModelInfo = Field(..., description="Model information")
    annotated_image_url: Optional[str] = Field(None, description="URL to annotated image")
    
    @validator('detections')
    def sort_detections_by_confidence(cls, v):
        """Sort detections by confidence score (highest first)"""
        return sorted(v, key=lambda x: x.confidence, reverse=True)

class BatchImageResult(BaseModel):
    """Single image result in batch processing"""
    filename: str = Field(..., description="Original filename")
    success: bool = Field(..., description="Whether processing was successful")
    detections: List[Detection] = Field(default_factory=list, description="Detected license plates")
    image_info: Optional[ImageInfo] = Field(None, description="Image information")
    processing_time: float = Field(..., description="Processing time for this image")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class BatchDetectionResponse(BaseResponse):
    """Response for batch image detection"""
    results: List[BatchImageResult] = Field(..., description="Results for each image")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    model_info: ModelInfo = Field(..., description="Model information")
    
    @root_validator
    def calculate_summary(cls, values):
        """Calculate batch processing summary"""
        if 'results' in values:
            results = values['results']
            total_files = len(results)
            successful_files = sum(1 for r in results if r.success)
            total_detections = sum(len(r.detections) for r in results if r.success)
            total_processing_time = sum(r.processing_time for r in results)
            
            values['summary'] = {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': total_files - successful_files,
                'success_rate': successful_files / total_files if total_files > 0 else 0,
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / successful_files if successful_files > 0 else 0,
                'total_processing_time': total_processing_time,
                'avg_processing_time': total_processing_time / total_files if total_files > 0 else 0
            }
        return values

class VideoDetectionJobResponse(BaseResponse):
    """Response for video detection job creation"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(default="queued", description="Job status")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    video_info: Optional[Dict[str, Any]] = Field(None, description="Video file information")

class VideoFrameResult(BaseModel):
    """Detection result for a single video frame"""
    frame_number: int = Field(..., description="Frame number in video")
    timestamp: float = Field(..., description="Timestamp in video (seconds)")
    detections: List[Detection] = Field(..., description="Detected license plates")
    processing_time: float = Field(..., description="Processing time for this frame")

class VideoDetectionResult(BaseModel):
    """Complete video detection results"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job completion status")
    video_info: Dict[str, Any] = Field(..., description="Video information")
    frame_results: List[VideoFrameResult] = Field(..., description="Per-frame detection results")
    summary: Dict[str, Any] = Field(..., description="Video processing summary")
    output_video_url: Optional[str] = Field(None, description="URL to annotated output video")
    
    @root_validator
    def calculate_video_summary(cls, values):
        """Calculate video processing summary"""
        if 'frame_results' in values:
            frame_results = values['frame_results']
            
            # Collect all unique license plate texts
            unique_plates = set()
            total_detections = 0
            
            for frame_result in frame_results:
                for detection in frame_result.detections:
                    total_detections += 1
                    if detection.ocr_result and detection.ocr_result.text:
                        unique_plates.add(detection.ocr_result.text)
            
            total_processing_time = sum(fr.processing_time for fr in frame_results)
            
            values['summary'] = {
                'total_frames_processed': len(frame_results),
                'total_detections': total_detections,
                'unique_license_plates': len(unique_plates),
                'license_plates_detected': list(unique_plates),
                'avg_detections_per_frame': total_detections / len(frame_results) if frame_results else 0,
                'total_processing_time': total_processing_time,
                'avg_processing_time_per_frame': total_processing_time / len(frame_results) if frame_results else 0
            }
        return values

# WebSocket models
class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)

class WebSocketDetectionRequest(WebSocketMessage):
    """WebSocket detection request"""
    type: str = Field(default="detection_request", const=True)
    image_data: str = Field(..., description="Base64 encoded image data")
    confidence: confloat(ge=0.0, le=1.0) = Field(default=0.5)
    extract_text: bool = Field(default=True)

class WebSocketDetectionResponse(WebSocketMessage):
    """WebSocket detection response"""
    type: str = Field(default="detection_response", const=True)
    detections: List[Detection] = Field(..., description="Detected license plates")
    processing_time: float = Field(..., description="Processing time")
    success: bool = Field(..., description="Whether detection was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")

# Error response models
class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Any = Field(None, description="The invalid value provided")

class ErrorResponse(BaseModel):
    """API error response"""
    success: bool = Field(default=False, const=True)
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    validation_errors: Optional[List[ValidationError]] = Field(None, description="Field validation errors")
    timestamp: datetime = Field(default_factory=datetime.now)

# Configuration models
class DetectionConfig(BaseModel):
    """Detection configuration"""
    confidence_threshold: confloat(ge=0.0, le=1.0) = Field(default=0.5)
    nms_threshold: confloat(ge=0.0, le=1.0) = Field(default=0.4)
    max_detections: conint(ge=1, le=1000) = Field(default=100)
    input_size: conint(ge=32, le=2048) = Field(default=640, description="Model input size")

class OCRConfig(BaseModel):
    """OCR configuration"""
    engine: OCREngineType = Field(default=OCREngineType.PADDLEOCR)
    confidence_threshold: confloat(ge=0.0, le=1.0) = Field(default=0.8)
    use_preprocessing: bool = Field(default=True)
    supported_languages: List[str] = Field(default=['en'])

class SystemConfig(BaseModel):
    """System configuration"""
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    max_file_size_mb: conint(ge=1, le=100) = Field(default=10)
    supported_image_formats: List[ImageFormat] = Field(
        default=[ImageFormat.JPEG, ImageFormat.PNG, ImageFormat.BMP]
    )
    supported_video_formats: List[VideoFormat] = Field(
        default=[VideoFormat.MP4, VideoFormat.AVI, VideoFormat.MOV]
    )

# Utility functions for validation
def validate_license_plate_text(text: str, region: RegionCode = RegionCode.US) -> bool:
    """Validate license plate text against regional patterns"""
    if not text or len(text.strip()) == 0:
        return False
    
    text = text.strip().upper()
    
    # US license plate patterns
    if region == RegionCode.US:
        us_patterns = [
            r'^[A-Z]{2,3}[0-9]{3,4}$',  # ABC123, AB1234
            r'^[0-9]{3}[A-Z]{3}$',      # 123ABC
            r'^[A-Z][0-9]{2}[A-Z][0-9]{3}$',  # A12B345
        ]
        return any(re.match(pattern, text) for pattern in us_patterns)
    
    # EU license plate patterns  
    elif region == RegionCode.EU:
        eu_patterns = [
            r'^[A-Z]{1,3}[0-9]{3,4}[A-Z]{1,3}$',  # AB123CD, A1234BC
        ]
        return any(re.match(pattern, text) for pattern in eu_patterns)
    
    # Global - accept any alphanumeric combination
    else:
        return bool(re.match(r'^[A-Z0-9]{4,10}$', text))

def main():
    """Demo/test function for schemas"""
    print("📝 License Plate Reader API Schemas Demo")
    print("=" * 45)
    
    # Create sample detection request
    request = SingleImageDetectionRequest(
        confidence=0.7,
        model=DetectionModelType.ROBOFLOW,
        extract_text=True,
        region=RegionCode.US
    )
    
    print("🔍 Sample Detection Request:")
    print(request.json(indent=2))
    
    # Create sample detection response
    bbox = BoundingBox(x1=100, y1=50, x2=200, y2=100)
    ocr_result = OCRResult(
        text="ABC123",
        confidence=0.95,
        engine=OCREngineType.PADDLEOCR,
        format_valid=True,
        region=RegionCode.US
    )
    
    detection = Detection(
        bbox=bbox,
        confidence=0.92,
        plate_score=0.88,
        source="roboflow",
        ocr_result=ocr_result
    )
    
    image_info = ImageInfo(width=640, height=480, format=ImageFormat.JPEG)
    model_info = ModelInfo(
        detector=DetectionModelType.ROBOFLOW,
        ocr_engine=OCREngineType.PADDLEOCR,
        confidence_threshold=0.7,
        device="cuda"
    )
    
    response = SingleImageDetectionResponse(
        success=True,
        detections=[detection],
        image_info=image_info,
        model_info=model_info,
        processing_time=0.45
    )
    
    print("\\n✅ Sample Detection Response:")
    print(response.json(indent=2))
    
    # Test validation
    print("\\n🧪 Testing License Plate Validation:")
    test_plates = ["ABC123", "123ABC", "INVALID", "A12B345", ""]
    
    for plate in test_plates:
        valid = validate_license_plate_text(plate, RegionCode.US)
        status = "✅ Valid" if valid else "❌ Invalid"
        print(f"  '{plate}': {status}")
    
    print("\\n✅ Schema validation demo completed!")

if __name__ == "__main__":
    main()