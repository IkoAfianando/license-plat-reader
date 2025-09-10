"""
OCR Engine for License Plate Text Recognition
Supports PaddleOCR, EasyOCR, and Tesseract with regional format validation
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
import logging

# Import OCR libraries with fallbacks
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASY_AVAILABLE = True
except ImportError:
    EASY_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCREngine:
    """OCR processing for license plates with multi-engine support"""
    
    def __init__(self, 
                 engine: str = 'paddleocr',
                 language: str = 'en',
                 region_config: Optional[Dict] = None):
        """
        Initialize OCR engine
        
        Args:
            engine: OCR engine to use ('paddleocr', 'easyocr', 'tesseract')
            language: Language code for OCR
            region_config: Regional format validation rules
        """
        self.engine_name = engine
        self.language = language
        self.region_config = region_config or self._load_default_regions()
        self.ocr_engine = None
        
        # Initialize selected engine
        self._initialize_engine(engine)
    
    def _initialize_engine(self, engine: str):
        """Initialize the specified OCR engine"""
        try:
            if engine == 'paddleocr' and PADDLE_AVAILABLE:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.language,
                    show_log=False
                )
                logger.info("PaddleOCR initialized")
                
            elif engine == 'easyocr' and EASY_AVAILABLE:
                self.ocr_engine = easyocr.Reader([self.language])
                logger.info("EasyOCR initialized")
                
            elif engine == 'tesseract' and TESSERACT_AVAILABLE:
                # Tesseract doesn't need initialization
                self.ocr_engine = 'tesseract'
                logger.info("Tesseract initialized")
                
            else:
                # Fallback to available engine
                if PADDLE_AVAILABLE:
                    self.engine_name = 'paddleocr'
                    self._initialize_engine('paddleocr')
                elif EASY_AVAILABLE:
                    self.engine_name = 'easyocr'
                    self._initialize_engine('easyocr')
                elif TESSERACT_AVAILABLE:
                    self.engine_name = 'tesseract'
                    self._initialize_engine('tesseract')
                else:
                    raise RuntimeError("No OCR engine available")
                    
        except Exception as e:
            logger.error(f"Failed to initialize {engine}: {e}")
            raise
    
    def extract_text(self, 
                    plate_image: np.ndarray,
                    preprocess: bool = True) -> Dict:
        """
        Extract text from license plate image
        
        Args:
            plate_image: Cropped license plate image
            preprocess: Apply image preprocessing
            
        Returns:
            Dictionary with extracted text, confidence, and metadata
        """
        if preprocess:
            processed_image = self._preprocess_image(plate_image)
        else:
            processed_image = plate_image
        
        # Extract text using selected engine
        if self.engine_name == 'paddleocr':
            result = self._extract_paddleocr(processed_image)
        elif self.engine_name == 'easyocr':
            result = self._extract_easyocr(processed_image)
        elif self.engine_name == 'tesseract':
            result = self._extract_tesseract(processed_image)
        else:
            raise ValueError(f"Unknown engine: {self.engine_name}")
        
        # Post-process and validate result
        if result['text']:
            result = self._post_process_result(result)
        
        return result
    
    def _extract_paddleocr(self, image: np.ndarray) -> Dict:
        """Extract text using PaddleOCR"""
        try:
            results = self.ocr_engine.ocr(image, cls=True)
            
            if not results or not results[0]:
                return {'text': '', 'confidence': 0.0, 'engine': 'paddleocr'}
            
            # Combine all detected text
            texts = []
            confidences = []
            
            for line in results[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    texts.append(text)
                    confidences.append(confidence)
            
            combined_text = ''.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'engine': 'paddleocr',
                'raw_results': results[0]
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'engine': 'paddleocr', 'error': str(e)}
    
    def _extract_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR"""
        try:
            results = self.ocr_engine.readtext(image)
            
            if not results:
                return {'text': '', 'confidence': 0.0, 'engine': 'easyocr'}
            
            # Combine all detected text
            texts = []
            confidences = []
            
            for result in results:
                bbox, text, confidence = result
                texts.append(text)
                confidences.append(confidence)
            
            combined_text = ''.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'engine': 'easyocr',
                'raw_results': results
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'engine': 'easyocr', 'error': str(e)}
    
    def _extract_tesseract(self, image: np.ndarray) -> Dict:
        """Extract text using Tesseract"""
        try:
            # Configure Tesseract for license plates
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            text = pytesseract.image_to_string(image, config=config).strip()
            
            # Get confidence data
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'engine': 'tesseract',
                'raw_results': data
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'engine': 'tesseract', 'error': str(e)}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve OCR accuracy"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize for better OCR (height ~32-48 pixels works well)
        height, width = gray.shape
        if height < 32:
            scale = 32 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 32), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Threshold
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        return gray
    
    def _post_process_result(self, result: Dict) -> Dict:
        """Post-process OCR result for better accuracy"""
        text = result['text']
        
        # Clean up text
        text = self._cleanup_text(text)
        
        # Apply common character fixes
        text = self._fix_common_misreads(text)
        
        # Validate against regional formats
        validation = self._validate_plate_format(text)
        result.update(validation)
        
        result['text'] = text
        return result
    
    def _cleanup_text(self, text: str) -> str:
        """Clean up extracted text"""
        # Remove spaces, dots, dashes
        text = re.sub(r'[.\s\-_]', '', text)
        
        # Convert to uppercase
        text = text.upper()
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        return text
    
    def _fix_common_misreads(self, text: str) -> str:
        """Fix common OCR misreads"""
        replacements = {
            'O': '0',  # O -> 0 in most cases for plates
            'I': '1',  # I -> 1
            'L': '1',  # L -> 1
            'S': '5',  # S -> 5 sometimes
            'G': '6',  # G -> 6 sometimes
            'B': '8',  # B -> 8 sometimes
        }
        
        # Apply character-level fixes with context
        fixed_text = text
        for old, new in replacements.items():
            # Only replace if it makes format sense
            test_text = fixed_text.replace(old, new)
            if self._is_better_format(test_text, fixed_text):
                fixed_text = test_text
        
        return fixed_text
    
    def _validate_plate_format(self, text: str, region: str = 'US') -> Dict:
        """Validate text against regional license plate formats"""
        if not text or region not in self.region_config:
            return {'format_valid': False, 'region': region, 'format_confidence': 0.0}
        
        region_data = self.region_config[region]
        
        # Check length
        min_len = region_data.get('min_length', 5)
        max_len = region_data.get('max_length', 8)
        if not (min_len <= len(text) <= max_len):
            return {'format_valid': False, 'region': region, 'format_confidence': 0.0}
        
        # Check patterns
        patterns = region_data.get('formats', [])
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            if re.match(pattern, text):
                return {
                    'format_valid': True, 
                    'region': region,
                    'format_confidence': 1.0,
                    'matched_pattern': pattern_info['description']
                }
        
        return {'format_valid': False, 'region': region, 'format_confidence': 0.0}
    
    def _is_better_format(self, new_text: str, old_text: str) -> bool:
        """Check if new text has better format validity"""
        new_valid = self._validate_plate_format(new_text)['format_valid']
        old_valid = self._validate_plate_format(old_text)['format_valid']
        return new_valid and not old_valid
    
    def _load_default_regions(self) -> Dict:
        """Load default regional format configurations"""
        return {
            'US': {
                'formats': [
                    {'pattern': r'^[A-Z0-9]{2,3}[A-Z0-9]{3,4}$', 'description': 'Standard US format'},
                    {'pattern': r'^[A-Z]{3}[0-9]{3}$', 'description': 'Three letter, three number'},
                    {'pattern': r'^[0-9]{3}[A-Z]{3}$', 'description': 'Three number, three letter'}
                ],
                'min_length': 5,
                'max_length': 8
            }
        }
    
    def batch_extract(self, plate_images: List[np.ndarray]) -> List[Dict]:
        """Extract text from multiple plate images"""
        results = []
        for image in plate_images:
            result = self.extract_text(image)
            results.append(result)
        return results
    
    def get_engine_info(self) -> Dict:
        """Get information about the OCR engine"""
        return {
            'engine': self.engine_name,
            'language': self.language,
            'available_engines': {
                'paddleocr': PADDLE_AVAILABLE,
                'easyocr': EASY_AVAILABLE, 
                'tesseract': TESSERACT_AVAILABLE
            }
        }