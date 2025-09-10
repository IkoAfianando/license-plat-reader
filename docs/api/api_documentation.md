# License Plate Reader API Documentation

## Overview
The License Plate Reader API provides endpoints for detecting and recognizing license plates in images and video streams. The API supports both Roboflow cloud detection and offline local detection modes.

## Base URL
```
http://localhost:8000
```

## Authentication
The API uses JWT tokens for authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

## Rate Limiting
- **General API**: 100 requests per minute per IP
- **Detection Endpoint**: 10 requests per second per IP
- **Batch Processing**: 5 requests per minute per IP

## Endpoints

### Health Check
Check if the API is running and healthy.

**GET** `/health`

#### Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "redis": "connected",
    "models": "loaded"
  }
}
```

### Authentication

#### Login
**POST** `/auth/login`

##### Request Body
```json
{
  "username": "admin",
  "password": "password123"
}
```

##### Response
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Detection Endpoints

#### Single Image Detection
Detect license plates in a single image.

**POST** `/detect/image`

##### Request
- Content-Type: `multipart/form-data`
- Body: Form data with image file

##### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Image file (JPG, PNG) |
| `confidence` | Float | No | 0.5 | Detection confidence threshold (0.0-1.0) |
| `use_roboflow` | Boolean | No | true | Use Roboflow API if available |
| `extract_text` | Boolean | No | true | Extract text using OCR |
| `return_image` | Boolean | No | false | Return annotated image |

##### Response
```json
{
  "success": true,
  "detections": [
    {
      "bbox": [100, 50, 200, 80],
      "confidence": 0.95,
      "plate_score": 0.87,
      "text": "ABC123",
      "text_confidence": 0.92,
      "source": "roboflow"
    }
  ],
  "processing_time": 0.45,
  "image_size": [640, 480],
  "model_info": {
    "detector": "roboflow",
    "ocr_engine": "paddleocr"
  }
}
```

#### Batch Image Detection
Process multiple images in a single request.

**POST** `/detect/batch`

##### Request
- Content-Type: `multipart/form-data`
- Body: Form data with multiple image files

##### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `files` | File[] | Yes | - | Array of image files |
| `confidence` | Float | No | 0.5 | Detection confidence threshold |
| `use_roboflow` | Boolean | No | true | Use Roboflow API if available |
| `extract_text` | Boolean | No | true | Extract text using OCR |

##### Response
```json
{
  "success": true,
  "results": [
    {
      "filename": "car1.jpg",
      "detections": [...],
      "processing_time": 0.45
    },
    {
      "filename": "car2.jpg", 
      "detections": [...],
      "processing_time": 0.52
    }
  ],
  "total_processing_time": 0.97,
  "total_detections": 3
}
```

#### Video Stream Detection
Process video file or stream for license plate detection.

**POST** `/detect/video`

##### Request
- Content-Type: `multipart/form-data` 
- Body: Form data with video file

##### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Video file (MP4, AVI, MOV) |
| `frame_skip` | Integer | No | 1 | Process every N frames |
| `max_frames` | Integer | No | null | Maximum frames to process |
| `confidence` | Float | No | 0.5 | Detection confidence threshold |

##### Response
```json
{
  "success": true,
  "job_id": "video_job_12345",
  "status": "processing",
  "estimated_completion": "2024-01-20T10:35:00Z"
}
```

#### Get Video Processing Status
Check status of video processing job.

**GET** `/detect/video/{job_id}/status`

##### Response
```json
{
  "job_id": "video_job_12345",
  "status": "completed",
  "progress": 100,
  "results": {
    "total_frames": 1000,
    "processed_frames": 1000,
    "total_detections": 45,
    "unique_plates": 12,
    "output_file": "outputs/videos/processed_video_12345.mp4"
  },
  "processing_time": 120.5
}
```

### Data Management

#### List Datasets
Get list of available datasets.

**GET** `/data/datasets`

##### Response
```json
{
  "datasets": [
    {
      "name": "training_set_1",
      "created": "2024-01-15T08:00:00Z",
      "images": 1500,
      "size": "2.5GB",
      "annotations": true
    }
  ],
  "total_datasets": 1
}
```

#### Upload Dataset
Upload new images to create or extend a dataset.

**POST** `/data/upload`

##### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `files` | File[] | Yes | Image files to upload |
| `dataset_name` | String | Yes | Name of the dataset |
| `source` | String | No | Source of the images |

##### Response
```json
{
  "success": true,
  "dataset_name": "new_dataset",
  "added": 50,
  "skipped": 2,
  "errors": 0
}
```

### Model Management

#### List Models
Get list of available models.

**GET** `/models`

##### Response
```json
{
  "models": [
    {
      "name": "yolov8n",
      "type": "detection",
      "status": "loaded",
      "accuracy": 0.87,
      "size": "6.2MB"
    },
    {
      "name": "custom_plates_v1",
      "type": "detection", 
      "status": "available",
      "accuracy": 0.94,
      "size": "25.1MB"
    }
  ]
}
```

#### Load Model
Load a specific model for detection.

**POST** `/models/{model_name}/load`

##### Response
```json
{
  "success": true,
  "model_name": "custom_plates_v1",
  "status": "loaded",
  "load_time": 2.3
}
```

### Analytics

#### Detection Statistics
Get detection statistics and analytics.

**GET** `/analytics/stats`

##### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_date` | String | No | Start date (ISO format) |
| `end_date` | String | No | End date (ISO format) |
| `dataset` | String | No | Filter by dataset name |

##### Response
```json
{
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-20T23:59:59Z"
  },
  "statistics": {
    "total_detections": 15420,
    "unique_plates": 8930,
    "avg_confidence": 0.87,
    "processing_time_avg": 0.34,
    "accuracy_metrics": {
      "precision": 0.94,
      "recall": 0.91,
      "f1_score": 0.92
    }
  },
  "daily_stats": [
    {
      "date": "2024-01-20",
      "detections": 145,
      "unique_plates": 87
    }
  ]
}
```

### System Configuration

#### Get Configuration
Get current system configuration.

**GET** `/config`

##### Response
```json
{
  "detection": {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "max_detections": 100
  },
  "ocr": {
    "engine": "paddleocr",
    "confidence_threshold": 0.8,
    "languages": ["en"]
  },
  "roboflow": {
    "enabled": true,
    "model_version": 4,
    "fallback_enabled": true
  }
}
```

#### Update Configuration
Update system configuration.

**PUT** `/config`

##### Request Body
```json
{
  "detection": {
    "confidence_threshold": 0.6
  },
  "ocr": {
    "engine": "easyocr"
  }
}
```

##### Response
```json
{
  "success": true,
  "updated_config": {
    "detection.confidence_threshold": 0.6,
    "ocr.engine": "easyocr"
  }
}
```

## Error Responses

### Standard Error Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format. Supported formats: JPG, PNG",
    "details": {
      "field": "file",
      "received_format": "GIF"
    }
  },
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `PROCESSING_ERROR` | 500 | Internal processing error |
| `MODEL_NOT_LOADED` | 503 | Detection model not available |

## WebSocket Endpoints

### Real-time Detection
Connect to real-time detection stream.

**WebSocket** `/ws/detect`

#### Message Format
```json
{
  "type": "detection",
  "data": {
    "image": "base64_encoded_image",
    "confidence": 0.5
  }
}
```

#### Response Format
```json
{
  "type": "detection_result",
  "data": {
    "detections": [...],
    "processing_time": 0.45
  }
}
```

## SDK Examples

### Python SDK
```python
import requests

# Initialize client
api_url = "http://localhost:8000"
token = "your-jwt-token"
headers = {"Authorization": f"Bearer {token}"}

# Detect license plates
with open("car.jpg", "rb") as f:
    response = requests.post(
        f"{api_url}/detect/image",
        files={"file": f},
        params={"confidence": 0.6},
        headers=headers
    )
    
result = response.json()
print(f"Found {len(result['detections'])} license plates")
```

### JavaScript SDK
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect/image', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`
    },
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`Found ${data.detections.length} license plates`);
});
```

### cURL Examples
```bash
# Health check
curl -X GET http://localhost:8000/health

# Detect license plates
curl -X POST \
  -H "Authorization: Bearer your-token" \
  -F "file=@car.jpg" \
  -F "confidence=0.6" \
  http://localhost:8000/detect/image

# Get detection statistics
curl -X GET \
  -H "Authorization: Bearer your-token" \
  "http://localhost:8000/analytics/stats?start_date=2024-01-01"
```

## Performance Guidelines

### Optimization Tips
1. **Image Size**: Resize images to 640x480 for optimal speed/accuracy balance
2. **Batch Processing**: Use batch endpoint for multiple images
3. **Confidence Threshold**: Adjust based on accuracy requirements
4. **Caching**: Enable Redis caching for repeated requests
5. **GPU**: Use GPU-enabled deployment for better performance

### Rate Limits
- Consider implementing client-side rate limiting
- Use exponential backoff for retry logic
- Monitor response headers for rate limit status

### Error Handling
- Implement proper error handling for all API calls
- Use appropriate HTTP status codes
- Log errors for debugging and monitoring