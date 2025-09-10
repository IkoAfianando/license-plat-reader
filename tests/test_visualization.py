#!/usr/bin/env python3
"""
Visualization Tests
Test drawing, saving, and report generation utilities.
"""

import numpy as np
from pathlib import Path

from src.utils.visualization import draw_detections, save_annotated_image, generate_html_report


def _make_image(w=200, h=120):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    return img


def test_draw_and_save_detection_image(temp_dir):
    img = _make_image()
    detections = [
        {"bbox": {"x1": 20, "y1": 30, "x2": 180, "y2": 70}, "confidence": 0.9, "text": "ABC123"}
    ]
    out_path = temp_dir / "outputs" / "visualizations" / "annotated.jpg"
    saved = save_annotated_image(img, detections, str(out_path))
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0


def test_generate_html_report(temp_dir):
    results = {
        "license_plates": [
            {"bbox": {"x1": 10, "y1": 10, "x2": 60, "y2": 30}, "confidence": 0.95, "text": "B 1234 CD"}
        ],
        "processing_time": 0.123
    }
    report = temp_dir / "outputs" / "reports" / "report.html"
    path = generate_html_report(results, str(report))
    content = Path(path).read_text(encoding='utf-8')
    assert "License Plate Detection Report" in content
    assert "Total Detections" in content
    assert "Processing Time" in content

