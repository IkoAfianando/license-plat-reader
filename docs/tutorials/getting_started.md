# Getting Started with License Plate Reader

## Quick Start Guide

This tutorial will get you up and running with the License Plate Reader system in just a few minutes.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.9+**
- **Docker & Docker Compose** (for full deployment)
- **Git** (to clone the repository)

## Installation Options

### Option 1: Quick Test (Offline Only)
Perfect for trying out the system without external dependencies.

```bash
# Clone the repository
git clone 
cd license-plate-reader

# Install minimal dependencies
pip install ultralytics opencv-python paddleocr numpy pillow pyyaml

# Generate sample data
python data/sample_data_generator.py --num_images 10

# Test offline detection
python src/offline/standalone_detector.py
```

### Option 2: Full Installation
Complete setup with API server, database, and monitoring.

```bash
# Clone the repository
git clone <repository-url>
cd license-plate-reader

# Install all dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Deploy with Docker
chmod +x deployment/scripts/deploy.sh
./deployment/scripts/deploy.sh
```

### Option 3: Roboflow Integration
High accuracy cloud-based detection.

```bash
# Get your Roboflow API key from https://roboflow.com
# Add to .env file:
echo "ROBOFLOW_API_KEY=your_api_key_here" >> .env

# Install Roboflow dependencies
pip install roboflow python-dotenv

# Test Roboflow integration
python scripts/test_roboflow.py
```

## Your First Detection

### 1. Prepare Test Images
```bash
# Create test images directory
mkdir test_images

# Add your car images to test_images/ folder
# Or generate synthetic images:
python data/sample_data_generator.py --num_images 5 --dataset_name test_images
```

### 2. Run Basic Detection

#### Offline Detection
```python
from src.offline.standalone_detector import StandaloneLPRDetector
import cv2

# Initialize detector
detector = StandaloneLPRDetector(confidence=0.5)

# Load test image
image = cv2.imread('test_images/car.jpg')

# Detect license plates
detections = detector.detect(image)

# Display results
print(f"Found {len(detections)} license plates:")
for i, detection in enumerate(detections):
    conf = detection['confidence']
    plate_score = detection.get('plate_score', 0)
    print(f"  Detection {i+1}: confidence={conf:.3f}, plate_score={plate_score:.3f}")

# Save results with annotations
detector.save_detection_results(image, detections, 'result.jpg')
print("Results saved to result.jpg")
```

#### Roboflow Detection
```python
from src.core.detector import LicensePlateDetector
import os

# Configure Roboflow
roboflow_config = {
    'api_key': os.getenv('ROBOFLOW_API_KEY'),
    'project_id': 'license-plate-recognition-rxg4e',
    'model_version': 4
}

# Initialize detector
detector = LicensePlateDetector(roboflow_config=roboflow_config)

# Detect license plates
detections = detector.detect('test_images/car.jpg', use_roboflow=True)

print(f"Roboflow found {len(detections)} license plates")
```

### 3. Add OCR Text Recognition
```python
from src.core.ocr_engine import OCREngine
import cv2

# Initialize OCR engine
ocr = OCREngine(engine='paddleocr')

# Extract plate regions
image = cv2.imread('test_images/car.jpg')
detections = detector.detect(image)
plate_regions = detector.extract_plate_regions(image, detections)

# Extract text from each plate
for i, plate_data in enumerate(plate_regions):
    result = ocr.extract_text(plate_data['image'])
    print(f"Plate {i+1}: '{result['text']}' (confidence: {result['confidence']:.3f})")
```

## Understanding the Results

### Detection Output
Each detection contains:
```python
{
    'bbox': [x1, y1, x2, y2],        # Bounding box coordinates
    'confidence': 0.95,               # Model confidence (0-1)
    'plate_score': 0.87,             # Plate likelihood score (offline only)
    'source': 'roboflow'             # Detection source
}
```

### OCR Output
Each OCR result contains:
```python
{
    'text': 'ABC123',                # Extracted text
    'confidence': 0.92,              # OCR confidence (0-1)
    'engine': 'paddleocr',          # OCR engine used
    'format_valid': True,           # Regional format validation
    'region': 'US'                  # Detected region
}
```

## Compare Detection Methods

Run the comparison script to see which method works best for your data:

```bash
# Add sample images to test with
cp your_car_images/* ./

# Run comparison
python scripts/compare_implementations.py

# View results
cat comparison_report.md
open visual_comparison.jpg
```

## Web API Usage

If you deployed with Docker, you can use the REST API:

### Test API Health
```bash
curl http://localhost:8000/health
```

### Detect via API
```bash
# Upload image for detection
curl -X POST \
  -F "file=@test_images/car.jpg" \
  -F "confidence=0.6" \
  http://localhost:8000/detect/image
```

### Python API Client
```python
import requests

# Upload image
with open('test_images/car.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/image',
        files={'file': f},
        params={'confidence': 0.6, 'extract_text': True}
    )

result = response.json()
print(f"API found {len(result['detections'])} plates")
for detection in result['detections']:
    print(f"  Text: {detection.get('text', 'N/A')}")
```

## Data Management

### Organize Your Data
```python
from data.data_manager import DataManager

# Initialize data manager
dm = DataManager()

# Add images to dataset
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
result = dm.add_raw_images(image_paths, dataset_name='my_dataset', source='camera_1')

print(f"Added {result['added']} images")

# Create annotation templates
annotation_dir = dm.create_annotations_template('my_dataset', 'yolo')
print(f"Annotation templates created in {annotation_dir}")

# Export dataset for training
export_path = dm.export_dataset('my_dataset', 'yolo', train_split=0.8)
print(f"Dataset exported to {export_path}")
```

## Performance Optimization

### GPU Acceleration
```python
# Use GPU for faster detection
detector = StandaloneLPRDetector(device='cuda')

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Batch Processing
```python
# Process multiple images efficiently
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']

# Offline batch processing
results = []
for img_path in image_paths:
    detections = detector.detect(img_path)
    results.append({'image': img_path, 'detections': detections})

# API batch processing
with open('img1.jpg', 'rb') as f1, open('img2.jpg', 'rb') as f2:
    files = {'files': [f1, f2]}
    response = requests.post('http://localhost:8000/detect/batch', files=files)
```

## Common Issues & Solutions

### Issue: Low Detection Accuracy
**Solutions:**
1. Adjust confidence threshold: `detector = StandaloneLPRDetector(confidence=0.3)`
2. Try different image preprocessing
3. Use Roboflow API for higher accuracy
4. Train custom model with your data

### Issue: OCR Text Not Recognized
**Solutions:**
1. Check image quality and resolution
2. Try different OCR engines: `ocr = OCREngine(engine='easyocr')`
3. Adjust OCR preprocessing parameters
4. Validate regional format settings

### Issue: API Connection Failed
**Solutions:**
1. Check if services are running: `docker-compose ps`
2. Verify port availability: `curl http://localhost:8000/health`
3. Check logs: `docker-compose logs lpr-app`
4. Restart services: `docker-compose restart`

### Issue: Out of Memory
**Solutions:**
1. Reduce image size: `image = cv2.resize(image, (640, 480))`
2. Use smaller batch sizes
3. Process images sequentially instead of batch
4. Clear GPU memory: `torch.cuda.empty_cache()`

## Next Steps

### 1. Custom Model Training
If you need higher accuracy for your specific use case:
```bash
# Collect and annotate your data
python data/data_manager.py

# Train custom model (coming soon)
python models/train_custom.py --dataset my_dataset --epochs 100
```

### 2. Production Deployment
Deploy to production environment:
```bash
# Deploy with monitoring
./deployment/scripts/deploy.sh

# Access monitoring
open http://localhost:3000  # Grafana dashboard
open http://localhost:9090  # Prometheus metrics
```

### 3. Integration with Existing Systems
```python
# Database integration
from database.models import DetectionResult

# Save detection to database
detection_record = DetectionResult(
    image_path='car.jpg',
    plate_text='ABC123',
    confidence=0.95,
    timestamp=datetime.now()
)
detection_record.save()
```

### 4. Real-time Processing
```python
# Process video stream
def process_frame(frame, detections, frame_info):
    # Your processing logic here
    print(f"Frame {frame_info['frame_number']}: {len(detections)} plates")

# Start real-time processing
detector.detect_video_stream(0, output_callback=process_frame)  # Webcam
```

## Getting Help

### Documentation
- **API Reference**: `docs/api/api_documentation.md`
- **Architecture**: `docs/architecture.md`
- **Deployment Guide**: `IMPLEMENTATION_GUIDE.md`

### Community
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Examples**: Check `docs/examples/` for more code samples

### Support
- Review logs in `outputs/logs/`
- Check system health at `http://localhost:8000/health`
- Use monitoring dashboard at `http://localhost:3000`

## Congratulations! 🎉

You now have a working License Plate Reader system. Start with the offline detection for quick testing, then move to the full API deployment for production use. Don't forget to check out the comparison tools to find the best approach for your specific needs.