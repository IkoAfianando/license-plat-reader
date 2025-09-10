"""
Visualization and Output Utilities
Draw detections, save annotated images, and generate simple HTML reports.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import os

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def draw_detections(image: Any, detections: List[Dict[str, Any]], color: tuple = (0, 255, 0)) -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV not available for visualization")
    if image is None:
        raise ValueError("Image is None")

    img = image.copy()
    for det in detections or []:
        bbox = det.get('bbox')
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        else:
            # list/tuple [x1,y1,x2,y2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
        conf = det.get('confidence', 0.0)
        label = det.get('text', '')
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}".strip()
        cv2.putText(img, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def save_annotated_image(image: Any, detections: List[Dict[str, Any]], output_path: str) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV not available for saving images")
    ensure_dir(str(Path(output_path).parent))
    annotated = draw_detections(image, detections)
    cv2.imwrite(str(output_path), annotated)
    return str(output_path)


def generate_html_report(results: Dict[str, Any], output_path: str) -> str:
    """Generate a minimal HTML report summarizing detection results."""
    ensure_dir(str(Path(output_path).parent))
    total = len(results.get('detections', results.get('license_plates', [])))
    processing_time = results.get('processing_time', 0.0)

    html = [
        "<html><head><meta charset='utf-8'><title>LPR Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:20px} .det{margin:8px 0;padding:8px;border:1px solid #eee}</style>",
        "</head><body>",
        f"<h1>License Plate Detection Report</h1>",
        f"<p><b>Total Detections:</b> {total}</p>",
        f"<p><b>Processing Time:</b> {processing_time:.3f}s</p>",
    ]

    dets = results.get('detections', results.get('license_plates', [])) or []
    for i, d in enumerate(dets):
        bbox = d.get('bbox', {})
        conf = d.get('confidence', 0.0)
        txt = d.get('text', '')
        html.append("<div class='det'>")
        html.append(f"<div><b>Detection {i+1}</b></div>")
        html.append(f"<div>Confidence: {conf:.2f}</div>")
        if bbox:
            if isinstance(bbox, dict):
                coords = f"({bbox.get('x1')},{bbox.get('y1')})-({bbox.get('x2')},{bbox.get('y2')})"
            else:
                coords = f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"
            html.append(f"<div>BBox: {coords}</div>")
        if txt:
            html.append(f"<div>Text: {txt}</div>")
        html.append("</div>")

    html.append("</body></html>")
    Path(output_path).write_text("\n".join(html), encoding='utf-8')
    return str(output_path)

