# 🚀 Local Deployment Guide

This project already has a complete license plate detection system that can be run locally. Here's how to run it:

## 📋 Prerequisites

### 1. Install Dependencies
```bash
# Basic dependencies
pip install fastapi uvicorn opencv-python pillow pyjwt
pip install ultralytics roboflow easyocr
pip install prometheus-client psycopg2-binary redis

# Optional for monitoring
pip install grafana-api influxdb-client
```

### 2. Setup Environment Variables
The `.env` file is already configured with the correct Roboflow model:
```bash
# Roboflow Configuration (already updated)
ROBOFLOW_API_KEY=kTbpsoZiESpDU3BfkZx3
ROBOFLOW_WORKSPACE=test-aip6t
ROBOFLOW_PROJECT=license-plate-recognition-8fvub-hvrra
ROBOFLOW_VERSION=2

# Database & Services
DB_PASSWORD=your_postgres_password  
REDIS_PASSWORD=your_redis_password
JWT_SECRET=your_jwt_secret_key_here

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
ADMIN_PASSWORD=admin123
```

## 🎯 Deployment Options

### Option 1: API Server (Recommended)

**🚀 Full API Server (Now Working - JWT Disabled):**
```bash
# From root project directory
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**🔧 Simplified API Server (Alternative):**
```bash
# Lightweight version for basic testing
uvicorn src.api.simple_main:app --host 0.0.0.0 --port 8000 --reload
```

**Features:**
- 🌐 REST API endpoints 
- 🚫 No Authentication (JWT disabled for testing)
- 📊 Real-time WebSocket detection
- 🎬 Video processing (background jobs)
- 📈 Metrics & monitoring
- 🔄 Batch processing
- 🤖 Multiple detection models (Offline + Roboflow)
- 🔍 OCR text extraction (PaddleOCR)

**API Endpoints (No Authentication Required):**
```bash
# Health check
GET http://localhost:8000/health

# System configuration  
GET http://localhost:8000/config

# Available models
GET http://localhost:8000/models

# Available datasets
GET http://localhost:8000/data/datasets

# Analytics & stats
GET http://localhost:8000/analytics/stats

# Single image detection
POST http://localhost:8000/detect/image
# Body: file + confidence + use_roboflow + extract_text + return_image

# Batch image detection
POST http://localhost:8000/detect/batch  

# Video processing (background job)
POST http://localhost:8000/detect/video

# Prometheus metrics
GET http://localhost:8000/metrics

# Interactive docs
GET http://localhost:8000/docs
```

**🧪 Test Commands:**
```bash
# Test health
curl http://localhost:8000/health | jq .

# Test config
curl http://localhost:8000/config | jq .

# Test image detection (offline mode)
curl -X POST http://localhost:8000/detect/image \
  -F "file=@data/images/Cars0.png" \
  -F "confidence=0.5" \
  -F "use_roboflow=false" \
  -F "extract_text=true" | jq .

# Test batch detection
curl -X POST http://localhost:8000/detect/batch \
  -F "files=@data/images/Cars0.png" \
  -F "files=@data/images/Cars1.png" \
  -F "confidence=0.4" | jq .
```

**🎯 Use .http Files:**
- `api_tests_no_auth.http` - Complete test suite without authentication
- `api_tests.http` - Original tests (for reference)

### Option 2: Docker Deployment

**Start with Docker Compose:**
```bash
# From root project directory
cd deployment/docker
docker-compose up -d
```

**Services included:**
- `lpr-app`: Main application (port 8000)
- `postgres`: Database (port 5432)
- `redis`: Cache (port 6379)  
- `prometheus`: Metrics (port 9090)
- `grafana`: Dashboard (port 3000)
- `influxdb`: Time-series data (port 8086)
- `nginx`: Load balancer (port 80/443)

### Option 3: Standalone Detection

**Offline detection tanpa API:**
```bash
python src/offline/standalone_detector.py
```

**Features:**
- 🔒 No internet required
- ⚡ Fast local inference
- 🎯 YOLOv8-based detection
- 📷 Support images & video
- 🔧 Customizable filtering

## 🎮 Usage Examples

### 1. Single Image Detection (API)
```python
import requests

# Login
login_response = requests.post('http://localhost:8000/auth/login', 
                              data={'username': 'admin', 'password': 'admin123'})
token = login_response.json()['access_token']

# Detect
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/detect/image',
                           headers={'Authorization': f'Bearer {token}'},
                           files={'file': f},
                           data={'confidence': 0.5, 'extract_text': True})
    
result = response.json()
print(f"Found {len(result['detections'])} license plates")
```

### 2. Direct Python Usage
```python
from src.offline.standalone_detector import StandaloneLPRDetector
import cv2

# Initialize detector
detector = StandaloneLPRDetector(confidence=0.5)

# Load image
image = cv2.imread('data/images/car.jpg')

# Detect license plates
detections = detector.detect(image)
print(f"Found {len(detections)} plates")

# Extract plate regions for OCR
plates = detector.extract_plate_regions(image, detections)
for i, plate_data in enumerate(plates):
    cv2.imwrite(f'plate_{i}.jpg', plate_data['image'])
```

### 3. Video Processing
```python
from src.api.main import LPRAPIServer
import asyncio

# Initialize server components
server = LPRAPIServer()

# Process video file
async def process_video():
    await server.process_video_background(
        job_id='test_job',
        video_path='data/videos/traffic.mp4',
        frame_skip=2,
        max_frames=100,
        confidence=0.5
    )

asyncio.run(process_video())
```

## 🔧 Configuration

### Model Configuration
Project supports multiple detection models:

1. **Roboflow Model (Online)**
   - Model: `test-aip6t/license-plate-recognition-8fvub-hvrra/2`
   - Confidence: 40%
   - Uses your specific trained model

2. **YOLOv8 Local Model (Offline)**
   - Model: YOLOv8n (downloads automatically)
   - Fully offline, no internet required
   - Faster inference, good for real-time

3. **Custom Local Model**
   - Place `.pt` file in `models/` directory
   - Configure path in detector initialization

### System Architecture
```
src/
├── api/          # FastAPI server & endpoints
├── core/         # Core detection engines  
├── offline/      # Standalone offline detector
└── ocr/          # OCR engines (EasyOCR, PaddleOCR)

models/           # Local model files
data/            # Input images/videos
outputs/         # Detection results
config/          # Configuration files
monitoring/      # Prometheus, Grafana configs
deployment/      # Docker, nginx configs
```

## 🚦 Testing

### Test API Server
```bash
# Start server
python -m src.api.main

# Test with curl
curl -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=admin123"

# Test detection
curl -X POST http://localhost:8000/detect/image \
  -H "Authorization: Bearer [token]" \
  -F "file=@data/images/sample.jpg" \
  -F "confidence=0.5"
```

### Test Standalone Detector
```bash
# Place test images in current directory
python src/offline/standalone_detector.py
```

## 📊 Monitoring

### Grafana Dashboard
- URL: http://localhost:3000 
- User: admin
- Pass: admin (or GRAFANA_PASSWORD from .env)

### Prometheus Metrics
- URL: http://localhost:9090
- Metrics: API requests, detection stats, system performance

### API Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

## 🔍 Model Comparison

| Feature | Roboflow Model | YOLOv8 Local | Custom Model |
|---------|----------------|--------------|--------------|
| **Accuracy** | High (trained specific) | Medium | Variable |
| **Speed** | Medium (API call) | Fast | Fast |
| **Offline** | ❌ No | ✅ Yes | ✅ Yes |
| **Cost** | API calls | Free | Free |
| **Customization** | Limited | Limited | Full |

## 🚨 Troubleshooting

### Common Issues:

1. **Import Errors**
   ```bash
   pip install ultralytics roboflow easyocr opencv-python
   ```

2. **Model Download Fails**
   ```bash
   # YOLOv8 will download automatically, ensure internet is active
   # Or download manually and place in models/
   ```

3. **Roboflow API Error**
   ```bash
   # Check API key di .env file
   echo $ROBOFLOW_API_KEY
   ```

4. **Port Already Used**
   ```bash
   # Change port di .env atau command line
   API_PORT=8001 python -m src.api.main
   ```

## 🎯 Model Performance

Your Roboflow model (`test-aip6t/license-plate-recognition-8fvub-hvrra/v2`) is now configured in:
- API server (`src/api/main.py:137-141`)
- Core detector (`src/core/detector.py:67-75`)
- Environment variables (`.env`)
