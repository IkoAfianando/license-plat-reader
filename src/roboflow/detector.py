"""
Roboflow Detector Adapter
Provides sync and async detection methods compatible with tests.
"""

from typing import Any, Dict, List, Optional
import base64
import io
import json

try:
    import numpy as np
    import cv2
except Exception:  # pragma: no cover - tests use mocks
    np = None
    cv2 = None


class RoboflowDetector:
    """Simple Roboflow detector client used in tests."""

    def __init__(self, api_key: str, project: str, version: int, base_url: str = "https://api.roboflow.com"):
        self.api_key = api_key
        self.project = project
        self.version = version
        self.base_url = base_url.rstrip("/")

    def _encode_image(self, image: Any) -> str:
        """Encode numpy/PIL image into base64 string."""
        # Accept numpy array (BGR/RGB or grayscale)
        if np is not None and isinstance(image, np.ndarray):
            img = image
            # Convert to BGR -> RGB if looks like OpenCV image
            if img.ndim == 3 and img.shape[2] == 3:
                # Assume BGR from OpenCV
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
            # Encode as PNG
            success, buf = cv2.imencode('.png', img) if cv2 is not None else (False, None)
            if success:
                return base64.b64encode(buf.tobytes()).decode('utf-8')
            # Fallback via PIL-like buffer
            bio = io.BytesIO()
            try:
                from PIL import Image
                mode = 'RGB' if img.ndim == 3 else 'L'
                Image.fromarray(img.astype('uint8'), mode=mode).save(bio, format='PNG')
                return base64.b64encode(bio.getvalue()).decode('utf-8')
            except Exception:
                raise ValueError("Failed to encode image")

        # Already a base64 string
        if isinstance(image, str):
            return image

        # Unknown type
        raise TypeError("Unsupported image type for encoding")

    def detect(self, image: Any) -> Dict[str, Any]:
        """Synchronous detection using requests.post. Tests patch requests.post."""
        try:
            import requests  # Imported here so tests can patch it easily
            payload = {"image": self._encode_image(image)}
            url = f"{self.base_url}/{self.project}/{self.version}?api_key={self.api_key}"
            resp = requests.post(url, json=payload)
            if getattr(resp, 'status_code', 200) != 200:
                # Handle common error cases
                if resp.status_code == 429:
                    retry_after = getattr(resp, 'headers', {}).get('Retry-After', '')
                    return {"error": f"rate_limit: retry_after={retry_after}"}
                data = resp.json() if hasattr(resp, 'json') else {}
                return {"error": data.get('error', f"HTTP {resp.status_code}")}

            data = resp.json()
            return self._parse_predictions(data)
        except Exception as e:  # Network or other error
            return {"error": str(e)}

    async def detect_async(self, image: Any) -> Dict[str, Any]:
        """Async detection using aiohttp. Tests patch aiohttp.ClientSession.post."""
        try:
            import aiohttp  # Imported here so tests can patch it
            payload = {"image": self._encode_image(image)}
            url = f"{self.base_url}/{self.project}/{self.version}?api_key={self.api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if getattr(resp, 'status', 200) != 200:
                        return {"error": f"HTTP {resp.status}"}
                    data = await resp.json()
                    return self._parse_predictions(data)
        except Exception as e:
            return {"error": str(e)}

    def _parse_predictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        predictions = data.get('predictions', [])
        image_info = data.get('image', {})
        width = image_info.get('width')
        height = image_info.get('height')

        license_plates: List[Dict[str, Any]] = []
        for pred in predictions:
            # Roboflow returns center x,y and width,height typically
            x = pred.get('x'); y = pred.get('y')
            w = pred.get('width'); h = pred.get('height')
            if None not in (x, y, w, h):
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            else:
                # Fallback if absolute coords provided
                bbox = pred.get('bbox') or {}

            license_plates.append({
                "bbox": bbox,
                "confidence": float(pred.get('confidence', 0.0)),
                "class": pred.get('class', 'license-plate')
            })

        return {
            "license_plates": license_plates,
            "processing_time": data.get('processing_time', 0.0),
            "image": {"width": width, "height": height}
        }

