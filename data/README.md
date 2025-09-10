# Data Management

This directory contains all data-related files for the License Plate Reader system.

## Directory Structure

```
data/
├── raw/           # Original, unprocessed images and videos
├── processed/     # Preprocessed and augmented data
├── annotations/   # Label files and annotation data
├── models/        # Trained model files and checkpoints
└── exports/       # Exported datasets and results
```

## Data Organization Guidelines

### Raw Data (`raw/`)
- Original images from cameras/sources
- Video files for training/testing
- Metadata and acquisition logs
- File naming convention: `YYYY-MM-DD_HH-MM-SS_camera_id.jpg`

### Processed Data (`processed/`)
- Cropped license plate regions
- Augmented training images
- Normalized and resized images
- Quality-filtered datasets

### Annotations (`annotations/`)
- YOLO format label files (.txt)
- COCO format annotations (.json)
- Regional format validation files
- Ground truth OCR text files

### Models (`models/`)
- Custom trained YOLOv8 models (.pt)
- Model performance metrics
- Training logs and configs
- Model version history

### Exports (`exports/`)
- Processed datasets for sharing
- Performance reports
- Inference results
- Batch processing outputs