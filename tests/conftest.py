"""
Test Configuration and Fixtures
Shared test setup for the license plate reader system
"""

import os
import sys
import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test dependencies
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import redis
from httpx import AsyncClient

# Import modules from our system
from database.models import Base, Detection, LicensePlate, ProcessingJob
from src.api.main import app, get_db
from src.pipeline.image_processor import ImageProcessor, LicensePlateImageProcessor
from data.data_manager import DatasetManager
from models.model_manager import ModelManager
from monitoring.metrics_collector import MetricsCollector

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")

@pytest.fixture
def test_db_session(test_engine) -> Generator[Session, None, None]:
    """Create test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def test_client(test_db_session):
    """Create test client with database override"""
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

@pytest.fixture
async def async_test_client(test_db_session):
    """Create async test client"""
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_image() -> np.ndarray:
    """Create sample test image"""
    # Create a simple test image with license plate-like rectangle
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[200:280, 200:440] = [255, 255, 255]  # White rectangle
    
    # Add some noise to make it realistic
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image

@pytest.fixture
def sample_license_plate_image() -> np.ndarray:
    """Create realistic license plate image"""
    # Create white background
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Add black border
    cv2.rectangle(img, (5, 5), (295, 95), (0, 0, 0), 2)
    
    # Add text (simulated license plate)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "ABC 123", (50, 60), font, 1.5, (0, 0, 0), 2)
    
    # Add some noise and blur for realism
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

@pytest.fixture
def sample_pil_image(sample_image) -> Image.Image:
    """Convert numpy array to PIL Image"""
    return Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing"""
    mock_model = Mock()
    
    # Mock detection results
    mock_result = Mock()
    mock_result.boxes.xyxy = np.array([[200, 200, 440, 280]])  # Bounding box
    mock_result.boxes.conf = np.array([0.95])  # Confidence
    mock_result.boxes.cls = np.array([0])  # Class (0 for license plate)
    
    mock_model.return_value = [mock_result]
    mock_model.names = {0: 'license-plate'}
    
    return mock_model

@pytest.fixture
def mock_roboflow_response():
    """Mock Roboflow API response"""
    return {
        "predictions": [
            {
                "x": 320,
                "y": 240,
                "width": 240,
                "height": 80,
                "confidence": 0.95,
                "class": "license-plate",
                "class_id": 0
            }
        ],
        "image": {
            "width": 640,
            "height": 480
        }
    }

@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    mock_redis = Mock(spec=redis.Redis)
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.pipeline.return_value = mock_redis
    mock_redis.execute.return_value = [0, 0]
    mock_redis.zadd.return_value = True
    mock_redis.expire.return_value = True
    mock_redis.zrange.return_value = []
    mock_redis.zcard.return_value = 0
    mock_redis.zremrangebyscore.return_value = 0
    return mock_redis

@pytest.fixture
def image_processor():
    """Create ImageProcessor instance"""
    return ImageProcessor(
        config={
            'noise_reduction': {'enabled': True, 'kernel_size': 5},
            'contrast_enhancement': {'enabled': True, 'clip_limit': 2.0},
            'sharpening': {'enabled': True, 'strength': 1.5},
            'morphological': {'enabled': True, 'kernel_size': 3}
        }
    )

@pytest.fixture
def lp_image_processor():
    """Create LicensePlateImageProcessor instance"""
    return LicensePlateImageProcessor(
        config={
            'noise_reduction': {'enabled': True, 'kernel_size': 5},
            'contrast_enhancement': {'enabled': True, 'clip_limit': 3.0},
            'sharpening': {'enabled': True, 'strength': 2.0},
            'morphological': {'enabled': True, 'kernel_size': 3},
            'license_plate': {
                'edge_enhancement': True,
                'text_enhancement': True,
                'background_removal': True
            }
        }
    )

@pytest.fixture
def dataset_manager(temp_dir):
    """Create DatasetManager instance"""
    return DatasetManager(str(temp_dir))

@pytest.fixture
def model_manager(temp_dir):
    """Create ModelManager instance with temporary directory"""
    config = {
        'model_dir': str(temp_dir / 'models'),
        'yolo': {
            'model_size': 'nano',
            'device': 'cpu',
            'conf_threshold': 0.5
        },
        'roboflow': {
            'api_key': 'test_key',
            'project': 'test_project',
            'version': 1
        }
    }
    return ModelManager(config)

@pytest.fixture
def metrics_collector():
    """Create MetricsCollector instance"""
    config = {
        'prometheus': {'enabled': False},  # Disable for testing
        'influxdb': {'enabled': False},
        'collection_interval': 1
    }
    return MetricsCollector(config)

@pytest.fixture
def jwt_token():
    """Create test JWT token"""
    import jwt
    payload = {
        "username": "testuser",
        "role": "user",
        "exp": 9999999999  # Far future expiry
    }
    return jwt.encode(payload, "test_secret", algorithm="HS256")

@pytest.fixture
def auth_headers(jwt_token):
    """Create authorization headers"""
    return {"Authorization": f"Bearer {jwt_token}"}

@pytest.fixture
def sample_detection_result():
    """Sample detection result for testing"""
    return {
        "license_plates": [
            {
                "text": "ABC123",
                "confidence": 0.95,
                "bbox": {
                    "x1": 200,
                    "y1": 200,
                    "x2": 440,
                    "y2": 280
                },
                "region": "US",
                "state": "CA"
            }
        ],
        "processing_time": 1.25,
        "image_info": {
            "width": 640,
            "height": 480,
            "channels": 3
        },
        "metadata": {
            "model_used": "yolov8n",
            "confidence_threshold": 0.5,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }

@pytest.fixture
def sample_batch_images(temp_dir):
    """Create sample batch of images for testing"""
    images_dir = temp_dir / "batch_images"
    images_dir.mkdir()
    
    image_paths = []
    for i in range(3):
        # Create simple test image
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        img_path = images_dir / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        image_paths.append(str(img_path))
    
    return image_paths

# Environment setup
@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables"""
    os.environ.update({
        "TESTING": "true",
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": "redis://localhost:6379/1",
        "JWT_SECRET_KEY": "test_secret_key_for_testing_only",
        "ROBOFLOW_API_KEY": "test_roboflow_key",
        "LOG_LEVEL": "DEBUG"
    })
    yield
    # Cleanup is automatic when test ends

# Async fixtures
@pytest_asyncio.fixture
async def async_image_processor():
    """Async image processor fixture"""
    processor = ImageProcessor()
    yield processor
    await processor.cleanup() if hasattr(processor, 'cleanup') else None

# Mock external services
@pytest.fixture
def mock_external_services():
    """Mock all external service calls"""
    with patch('requests.post') as mock_post, \
         patch('smtplib.SMTP') as mock_smtp, \
         patch('redis.Redis') as mock_redis:
        
        # Mock Roboflow API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "predictions": [
                {
                    "x": 320, "y": 240, "width": 240, "height": 80,
                    "confidence": 0.95, "class": "license-plate"
                }
            ]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        yield {
            'requests': mock_post,
            'smtp': mock_smtp,
            'redis': mock_redis
        }

# Performance testing fixtures
@pytest.fixture
def performance_test_config():
    """Configuration for performance tests"""
    return {
        'max_response_time': 5.0,  # seconds
        'max_memory_usage': 500,   # MB
        'concurrent_requests': 10,
        'test_duration': 30        # seconds
    }

# Data validation fixtures
@pytest.fixture
def valid_license_plate_formats():
    """Valid license plate formats for testing"""
    return {
        'US': ['ABC123', 'AB1234', '123ABC'],
        'EU': ['AB-123-CD', 'ABC-1234'],
        'ASIA': ['ABC-123', '1234-AB']
    }

@pytest.fixture
def invalid_license_plate_formats():
    """Invalid license plate formats for testing"""
    return [
        '', '1', 'A', '!@#$%', 'TOOLONGTEXT123456789',
        'ABC', '123', 'AB', '12'
    ]