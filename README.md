# License Plate Reader System Research & Development

## Executive Summary
The License Plate Reader (LPR) system uses YOLOv8 as the backbone detection model with the Roboflow Universe dataset. This system is capable of detecting, reading, and storing vehicle license plate data in real-time with cross-referencing database capabilities.

### Key Features
- Real-time license plate detection
- OCR for reading plate characters  
- Database integration for cross-comparing
- Multi-camera support
- Alert system for specific license plates
- Data analytics and reporting

## Technology Stack

### Core Technologies
1. **YOLOv8 (Detection Model)**
   - Framework: Ultralytics YOLOv8
   - Model Variants: YOLOv8n (speed), YOLOv8s (balanced), YOLOv8m (accuracy), YOLOv8l (maximum)

2. **OCR Technologies**
   - Primary: PaddleOCR (Best for license plates)
   - Secondary: EasyOCR (Multi-language support)
   - Fallback: Tesseract OCR (Lightweight)

3. **Database Systems**
   - Primary DB: PostgreSQL (Main storage)
   - Cache: Redis (Real-time caching)
   - Time-series: InfluxDB (Analytics data)

## Roboflow Dataset
- **Source**: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
- **Total Images**: 24,242 images
- **Annotations**: Bounding box format
- **Classes**: License plates (single class)
- **Formats Available**: YOLOv8, COCO, Pascal VOC

## Performance Comparison

| Feature | Roboflow API | Offline Implementation |
|---------|-------------|------------------------|
| **Detection Accuracy** | 99% mAP@0.5 | 85-95% mAP@0.5 |
| **OCR Accuracy** | N/A (detection only) | >90% character accuracy |
| **Processing Speed** | 45ms (cloud) | 15-30ms (local) |
| **Internet Required** | Yes | No |
| **Cost per Image** | $0.0005 | $0 |
| **Privacy** | Data sent to cloud | Fully local |
| **Customization** | Limited | Full control |

## Implementation Options

### Option 1: Roboflow API (Cloud-based)
```bash
pip install roboflow ultralytics opencv-python paddleocr python-dotenv
cd license-plate-reader
cp .env.example .env
# Edit .env with your ROBOFLOW_API_KEY
python scripts/test_roboflow.py
```

### Option 2: Offline/Standalone (No API Required)
```bash
pip install ultralytics opencv-python paddleocr easyocr
cd license-plate-reader
python src/offline/standalone_detector.py
```

### Option 3: Run the API Server
```bash
pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs for interactive API docs
```

## Project Structure
```
license-plate-reader/
├── src/
│   ├── core/
│   │   ├── detector.py          # Main detection engine (supports both)
│   │   └── ocr_engine.py        # OCR processing
│   ├── roboflow/
│   │   └── detector.py          # Lightweight Roboflow API adapter
│   └── offline/
│       └── standalone_detector.py # Pure offline implementation
│   └── utils/
│       └── visualization.py     # Drawing boxes, saving images, HTML reports
├── scripts/
│   └── test_roboflow.py         # Roboflow API testing
├── docs/
│   ├── architecture.md
│   ├── roboflow_integration.md  # Roboflow implementation guide
│   └── offline_implementation.md # Offline implementation guide
├── config/
│   └── settings.yaml           # Configuration settings
├── .env.example               # Environment variables template
└── requirements.txt
```

## Documentation
- `docs/roboflow_integration.md` - Complete Roboflow API implementation guide
- `docs/offline_implementation.md` - Standalone/offline implementation guide
- `docs/architecture.md` - System architecture overview

## Testing
```bash
pip install -r requirements.txt
pytest -q
```

## Output & Visualization
- Annotated images: use `src/utils/visualization.py`
  - `save_annotated_image(image, detections, "outputs/visualizations/example.jpg")`
- HTML reports: `generate_html_report(result, "outputs/reports/report.html")`
