"""
SQLAlchemy Database Models for License Plate Reader
ORM models for all database tables and relationships
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, 
    ForeignKey, UniqueConstraint, CheckConstraint, Index, BigInteger,
    Enum, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property

try:
    from geoalchemy2 import Geography
    GEOALCHEMY_AVAILABLE = True
except ImportError:
    GEOALCHEMY_AVAILABLE = False

Base = declarative_base()

# Enums
class DetectionStatus(PyEnum):
    SUCCESS = "success"
    FAILED = "failed"
    PROCESSING = "processing"

class ModelType(PyEnum):
    OFFLINE = "offline"
    ROBOFLOW = "roboflow"
    CUSTOM = "custom"

class OCREngine(PyEnum):
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"

class AlertSeverity(PyEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class JobStatus(PyEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Base model with common fields
class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class UUIDMixin:
    """Mixin for UUID primary key"""
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)

# User and Authentication Models
class User(Base, UUIDMixin, TimestampMixin):
    """User account model"""
    __tablename__ = 'users'
    
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True))
    api_key = Column(String(255), unique=True)
    rate_limit_tier = Column(String(20), default='standard')
    
    # Relationships
    images = relationship("Image", back_populates="uploaded_by_user")
    detection_sessions = relationship("DetectionSession", back_populates="user")
    video_jobs = relationship("VideoJob", back_populates="user")
    api_sessions = relationship("APISession", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

class APISession(Base, UUIDMixin):
    """API session/token model"""
    __tablename__ = 'api_sessions'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    token_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user_agent = Column(Text)
    ip_address = Column(INET)
    
    # Relationships
    user = relationship("User", back_populates="api_sessions")

# Model Management
class DetectionModel(Base, UUIDMixin, TimestampMixin):
    """Detection model registry"""
    __tablename__ = 'detection_models'
    
    name = Column(String(100), unique=True, nullable=False)
    type = Column(Enum(ModelType), nullable=False)
    version = Column(String(20))
    file_path = Column(Text)
    file_size_bytes = Column(BigInteger)
    file_hash = Column(String(64))
    accuracy_map50 = Column(Float)
    accuracy_map95 = Column(Float)
    inference_time_ms = Column(Integer)
    is_active = Column(Boolean, default=True, nullable=False)
    description = Column(Text)
    training_data_info = Column(JSONB)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    detections = relationship("Detection", back_populates="model")
    
    def __repr__(self):
        return f"<DetectionModel(name='{self.name}', type='{self.type}')>"

# Image Management
class Image(Base, UUIDMixin):
    """Uploaded/processed image model"""
    __tablename__ = 'images'
    
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    mime_type = Column(String(50), nullable=False)
    width_px = Column(Integer, nullable=False)
    height_px = Column(Integer, nullable=False)
    channels = Column(Integer, default=3)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    upload_timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    uploaded_by_user = relationship("User", back_populates="images")
    detections = relationship("Detection", back_populates="image")
    dataset_associations = relationship("DatasetImage", back_populates="image")
    
    @hybrid_property
    def aspect_ratio(self):
        """Calculate image aspect ratio"""
        return self.width_px / self.height_px if self.height_px > 0 else 0
    
    @hybrid_property 
    def megapixels(self):
        """Calculate image megapixels"""
        return (self.width_px * self.height_px) / 1_000_000
    
    def __repr__(self):
        return f"<Image(filename='{self.filename}', size={self.width_px}x{self.height_px})>"

# Detection Management
class DetectionSession(Base, UUIDMixin, TimestampMixin):
    """Detection session for grouping related detections"""
    __tablename__ = 'detection_sessions'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(100))
    description = Column(Text)
    is_batch = Column(Boolean, default=False)
    total_images = Column(Integer, default=0)
    completed_images = Column(Integer, default=0)
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    
    # Relationships
    user = relationship("User", back_populates="detection_sessions")
    detections = relationship("Detection", back_populates="session")
    
    @hybrid_property
    def completion_percentage(self):
        """Calculate completion percentage"""
        if self.total_images == 0:
            return 0.0
        return (self.completed_images / self.total_images) * 100
    
    def __repr__(self):
        return f"<DetectionSession(name='{self.name}', status='{self.status}')>"

class Detection(Base, UUIDMixin):
    """Individual detection result"""
    __tablename__ = 'detections'
    
    session_id = Column(UUID(as_uuid=True), ForeignKey('detection_sessions.id', ondelete='CASCADE'))
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.id', ondelete='CASCADE'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('detection_models.id'))
    
    # Detection coordinates
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # Detection scores
    confidence = Column(Float, nullable=False)
    plate_score = Column(Float)
    class_id = Column(Integer, default=0)
    class_name = Column(String(50), default='license_plate')
    
    # OCR results
    ocr_text = Column(String(20))
    ocr_confidence = Column(Float)
    ocr_engine = Column(Enum(OCREngine))
    text_format_valid = Column(Boolean, default=False)
    detected_region = Column(String(10))
    
    # Processing metadata
    processing_time_ms = Column(Integer)
    model_version = Column(String(20))
    status = Column(Enum(DetectionStatus), default=DetectionStatus.SUCCESS)
    error_message = Column(Text)
    
    # Timestamps
    detected_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Additional metadata
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    session = relationship("DetectionSession", back_populates="detections")
    image = relationship("Image", back_populates="detections")
    model = relationship("DetectionModel", back_populates="detections")
    license_plate_associations = relationship("DetectionLicensePlate", back_populates="detection")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('bbox_x2 > bbox_x1 AND bbox_y2 > bbox_y1', name='valid_bbox'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='valid_confidence'),
        CheckConstraint('plate_score IS NULL OR (plate_score >= 0 AND plate_score <= 1)', name='valid_plate_score'),
        CheckConstraint('ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 1)', name='valid_ocr_confidence'),
        Index('idx_detections_image_detected', image_id, detected_at),
        Index('idx_detections_ocr_text', ocr_text),
        Index('idx_detections_confidence', confidence),
    )
    
    @hybrid_property
    def bbox_width(self):
        """Calculate bounding box width"""
        return self.bbox_x2 - self.bbox_x1
    
    @hybrid_property
    def bbox_height(self):
        """Calculate bounding box height"""
        return self.bbox_y2 - self.bbox_y1
    
    @hybrid_property
    def bbox_area(self):
        """Calculate bounding box area"""
        return self.bbox_width * self.bbox_height
    
    def __repr__(self):
        return f"<Detection(id='{self.id}', confidence={self.confidence}, text='{self.ocr_text}')>"

# License Plate Registry
class LicensePlate(Base, UUIDMixin):
    """Unique license plates registry"""
    __tablename__ = 'license_plates'
    
    plate_text = Column(String(20), unique=True, nullable=False, index=True)
    normalized_text = Column(String(20), nullable=False, index=True)
    detected_region = Column(String(10))
    format_pattern = Column(String(50))
    confidence_avg = Column(Float)
    detection_count = Column(Integer, default=1)
    first_seen = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_flagged = Column(Boolean, default=False, index=True)
    flag_reason = Column(Text)
    notes = Column(Text)
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    detection_associations = relationship("DetectionLicensePlate", back_populates="license_plate")
    
    @hybrid_property
    def days_since_first_seen(self):
        """Days since first detection"""
        return (datetime.utcnow() - self.first_seen).days
    
    @hybrid_property
    def detection_frequency(self):
        """Detection frequency per day"""
        days = max(1, self.days_since_first_seen)
        return self.detection_count / days
    
    def __repr__(self):
        return f"<LicensePlate(text='{self.plate_text}', count={self.detection_count})>"

class DetectionLicensePlate(Base):
    """Association table for detections and license plates"""
    __tablename__ = 'detection_license_plates'
    
    detection_id = Column(UUID(as_uuid=True), ForeignKey('detections.id', ondelete='CASCADE'), primary_key=True)
    license_plate_id = Column(UUID(as_uuid=True), ForeignKey('license_plates.id', ondelete='CASCADE'), primary_key=True)
    
    # Relationships
    detection = relationship("Detection", back_populates="license_plate_associations")
    license_plate = relationship("LicensePlate", back_populates="detection_associations")

# Video Processing
class VideoJob(Base, UUIDMixin):
    """Video processing job"""
    __tablename__ = 'video_jobs'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    
    # Video metadata
    duration_seconds = Column(Float)
    fps = Column(Float)
    total_frames = Column(Integer)
    width_px = Column(Integer)
    height_px = Column(Integer)
    
    # Processing parameters
    frame_skip = Column(Integer, default=1)
    max_frames = Column(Integer)
    confidence_threshold = Column(Float, default=0.5)
    model_id = Column(UUID(as_uuid=True), ForeignKey('detection_models.id'))
    
    # Job status
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    progress_percent = Column(Float, default=0)
    frames_processed = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    unique_plates_count = Column(Integer, default=0)
    
    # Output
    output_video_path = Column(Text)
    results_json = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="video_jobs")
    
    @hybrid_property
    def processing_duration(self):
        """Calculate processing duration"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def __repr__(self):
        return f"<VideoJob(filename='{self.filename}', status='{self.status}')>"

# Monitoring and Analytics
class SystemMetric(Base, UUIDMixin):
    """System performance metrics"""
    __tablename__ = 'system_metrics'
    
    metric_name = Column(String(50), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    tags = Column(JSONB, default=dict)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_system_metrics_name_time', metric_name, recorded_at),
    )
    
    def __repr__(self):
        return f"<SystemMetric(name='{self.metric_name}', value={self.metric_value})>"

class PerformanceMetric(Base, UUIDMixin):
    """API performance metrics"""
    __tablename__ = 'performance_metrics'
    
    endpoint = Column(String(100), index=True)
    method = Column(String(10))
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    ip_address = Column(INET)
    user_agent = Column(Text)
    request_size_bytes = Column(Integer)
    response_size_bytes = Column(Integer)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_performance_metrics_endpoint_time', endpoint, timestamp),
    )

class Alert(Base, UUIDMixin):
    """System alerts"""
    __tablename__ = 'alerts'
    
    rule_name = Column(String(100), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert lifecycle
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resolved_at = Column(DateTime(timezone=True))
    
    # Context
    trigger_data = Column(JSONB, default=dict)
    affected_components = Column(ARRAY(String))
    
    # Notification status
    email_sent = Column(Boolean, default=False)
    webhook_sent = Column(Boolean, default=False)
    slack_sent = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_alerts_triggered_at', triggered_at),
        Index('idx_alerts_severity', severity),
    )
    
    def __repr__(self):
        return f"<Alert(rule='{self.rule_name}', severity='{self.severity}')>"

# Rate Limiting
class RateLimit(Base, UUIDMixin):
    """Rate limiting tracking"""
    __tablename__ = 'rate_limits'
    
    key_hash = Column(String(64), nullable=False)
    endpoint = Column(String(100), nullable=False)
    request_count = Column(Integer, default=1)
    window_start = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    window_end = Column(DateTime(timezone=True))
    is_blocked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        UniqueConstraint('key_hash', 'endpoint', 'window_start'),
        Index('idx_rate_limits_key_endpoint', key_hash, endpoint),
        Index('idx_rate_limits_window', window_start, window_end),
    )

# Audit Logging
class AuditLog(Base, UUIDMixin):
    """System audit log"""
    __tablename__ = 'audit_log'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    ip_address = Column(INET)
    user_agent = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    session_id = Column(UUID(as_uuid=True))
    
    __table_args__ = (
        Index('idx_audit_log_user_timestamp', user_id, timestamp),
        Index('idx_audit_log_resource', resource_type, resource_id),
    )

# Data Management
class Dataset(Base, UUIDMixin, TimestampMixin):
    """Dataset management"""
    __tablename__ = 'datasets'
    
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    version = Column(String(20), default='1.0')
    
    # Dataset statistics
    total_images = Column(Integer, default=0)
    total_annotations = Column(Integer, default=0)
    size_bytes = Column(BigInteger, default=0)
    
    # Splits
    train_split = Column(Float, default=0.8)
    val_split = Column(Float, default=0.1)
    test_split = Column(Float, default=0.1)
    
    # Export information
    export_format = Column(String(20))
    export_path = Column(Text)
    last_exported = Column(DateTime(timezone=True))
    
    # Metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    tags = Column(ARRAY(String))
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    image_associations = relationship("DatasetImage", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', images={self.total_images})>"

class DatasetImage(Base):
    """Association table for datasets and images"""
    __tablename__ = 'dataset_images'
    
    dataset_id = Column(UUID(as_uuid=True), ForeignKey('datasets.id', ondelete='CASCADE'), primary_key=True)
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.id', ondelete='CASCADE'), primary_key=True)
    split_type = Column(String(10))  # train, val, test
    annotation_path = Column(Text)
    added_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="image_associations")
    image = relationship("Image", back_populates="dataset_associations")
    
    __table_args__ = (
        CheckConstraint("split_type IN ('train', 'val', 'test')", name='valid_split_type'),
    )

# Processing Pipelines
class ProcessingPipeline(Base, UUIDMixin, TimestampMixin):
    """Data processing pipeline configuration"""
    __tablename__ = 'processing_pipelines'
    
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    config = Column(JSONB, nullable=False)
    is_active = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    def __repr__(self):
        return f"<ProcessingPipeline(name='{self.name}', active={self.is_active})>"

def main():
    """Demo function to show model usage"""
    print("🗄️  License Plate Reader Database Models Demo")
    print("=" * 50)
    
    # Show model information
    models = [
        User, APISession, DetectionModel, Image, DetectionSession,
        Detection, LicensePlate, VideoJob, SystemMetric, Alert, Dataset
    ]
    
    print("📋 Available Models:")
    for model in models:
        print(f"  • {model.__name__}: {model.__doc__.strip() if model.__doc__ else 'No description'}")
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total models: {len(models)}")
    print(f"  With relationships: {len([m for m in models if hasattr(m, '__mapper__') and m.__mapper__.relationships])}")
    print(f"  With hybrid properties: {len([m for m in models if any(hasattr(m, attr) for attr in dir(m) if attr.startswith('hybrid_'))])}")
    
    # Show example of model relationships
    print(f"\n🔗 Example Relationships:")
    print(f"  User -> Images: One-to-Many")
    print(f"  Image -> Detections: One-to-Many") 
    print(f"  Detection -> LicensePlate: Many-to-Many")
    print(f"  DetectionSession -> Detections: One-to-Many")
    
    print(f"\n✅ Database models ready for use!")
    print(f"   Use these models with SQLAlchemy ORM")
    print(f"   Initialize database with: database/schemas/init.sql")

if __name__ == "__main__":
    main()