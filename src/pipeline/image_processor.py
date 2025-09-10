"""
Image Processing Pipeline for License Plate Reader
Advanced image preprocessing and enhancement for better detection accuracy
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
from pathlib import Path
from enum import Enum
import time

try:
    from skimage import filters, morphology, exposure, restoration
    from scipy import ndimage
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False
    logging.warning("Advanced processing libraries not available. Install with: pip install scikit-image scipy")

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Image processing modes"""
    FAST = "fast"           # Basic preprocessing for speed
    BALANCED = "balanced"   # Balance between speed and quality
    QUALITY = "quality"     # Maximum quality processing
    CUSTOM = "custom"       # User-defined processing chain

class ImageProcessor:
    """Advanced image processing pipeline for license plate detection"""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.BALANCED):
        """
        Initialize image processor
        
        Args:
            mode: Processing mode (fast, balanced, quality, custom)
        """
        self.mode = mode
        self.processing_stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'preprocessing_steps': []
        }
        
        # Define processing chains for each mode
        self.processing_chains = {
            ProcessingMode.FAST: [
                'resize_if_needed',
                'convert_color_space',
                'basic_enhancement'
            ],
            ProcessingMode.BALANCED: [
                'resize_if_needed',
                'convert_color_space',
                'noise_reduction',
                'contrast_enhancement',
                'sharpening'
            ],
            ProcessingMode.QUALITY: [
                'resize_if_needed',
                'convert_color_space',
                'advanced_denoising',
                'histogram_equalization',
                'unsharp_masking',
                'morphological_operations',
                'edge_enhancement'
            ]
        }
        
        self.current_chain = self.processing_chains.get(mode, self.processing_chains[ProcessingMode.BALANCED])
        
    def process_image(self, 
                     image: np.ndarray,
                     target_size: Optional[Tuple[int, int]] = None,
                     custom_steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process image through the pipeline
        
        Args:
            image: Input image as numpy array
            target_size: Target size for resizing (width, height)
            custom_steps: Custom processing steps (overrides mode)
            
        Returns:
            Dictionary with processed image and metadata
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        # Use custom steps if provided
        processing_steps = custom_steps if custom_steps else self.current_chain
        
        # Initialize result
        result = {
            'original_image': image.copy(),
            'processed_image': image.copy(),
            'processing_steps': [],
            'processing_time': 0.0,
            'image_info': self._get_image_info(image),
            'enhancement_metrics': {}
        }
        
        current_image = image.copy()
        
        # Apply each processing step
        for step_name in processing_steps:
            try:
                step_start = time.time()
                
                if hasattr(self, f'_step_{step_name}'):
                    step_func = getattr(self, f'_step_{step_name}')
                    current_image, step_info = step_func(current_image, target_size)
                    
                    step_time = time.time() - step_start
                    
                    result['processing_steps'].append({
                        'name': step_name,
                        'processing_time': step_time,
                        'info': step_info
                    })
                    
                else:
                    logger.warning(f"Processing step '{step_name}' not found")
                    
            except Exception as e:
                logger.error(f"Error in processing step '{step_name}': {e}")
                # Continue with next step rather than failing completely
                continue
        
        result['processed_image'] = current_image
        result['processing_time'] = time.time() - start_time
        result['enhancement_metrics'] = self._calculate_enhancement_metrics(
            image, current_image
        )
        
        # Update statistics
        self.processing_stats['images_processed'] += 1
        self.processing_stats['total_processing_time'] += result['processing_time']
        
        return result
    
    def _get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get basic image information"""
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'dtype': str(image.dtype),
            'size_bytes': image.nbytes,
            'aspect_ratio': width / height
        }
    
    # Processing Steps
    def _step_resize_if_needed(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Resize image if needed"""
        if target_size is None:
            # Auto-resize based on image size for optimal processing
            height, width = image.shape[:2]
            
            # If image is too small, upscale
            if height < 200 or width < 300:
                scale_factor = max(200 / height, 300 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                target_size = (new_width, new_height)
            
            # If image is too large, downscale for processing speed
            elif height > 1200 or width > 1600:
                scale_factor = min(1200 / height, 1600 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                target_size = (new_width, new_height)
            else:
                return image, {'resized': False, 'original_size': (width, height)}
        
        # Perform resize
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        return resized_image, {
            'resized': True,
            'original_size': (image.shape[1], image.shape[0]),
            'new_size': target_size,
            'interpolation': 'INTER_AREA'
        }
    
    def _step_convert_color_space(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Convert color space for optimal processing"""
        if len(image.shape) == 3:
            # Convert BGR to RGB for consistency
            if image.shape[2] == 3:
                # Assume input is BGR (OpenCV default), convert to RGB
                converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return converted, {'conversion': 'BGR_to_RGB', 'channels': 3}
            else:
                return image, {'conversion': 'none', 'channels': image.shape[2]}
        else:
            return image, {'conversion': 'none', 'channels': 1}
    
    def _step_basic_enhancement(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Basic image enhancement"""
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced, {
            'enhancement_type': 'CLAHE',
            'clip_limit': 2.0,
            'tile_size': (8, 8)
        }
    
    def _step_noise_reduction(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Apply noise reduction"""
        if len(image.shape) == 3:
            # Color image denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            method = 'fastNlMeansDenoisingColored'
        else:
            # Grayscale image denoising
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            method = 'fastNlMeansDenoising'
        
        return denoised, {
            'method': method,
            'filter_strength': 10,
            'template_window': 7,
            'search_window': 21
        }
    
    def _step_contrast_enhancement(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Enhance image contrast"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced, {
            'method': 'CLAHE',
            'clip_limit': 3.0,
            'tile_size': (8, 8)
        }
    
    def _step_sharpening(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Apply image sharpening"""
        # Unsharp mask filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1], 
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        return sharpened, {
            'method': 'unsharp_mask',
            'kernel_size': (3, 3)
        }
    
    def _step_advanced_denoising(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Advanced denoising using multiple methods"""
        if not ADVANCED_PROCESSING_AVAILABLE:
            return self._step_noise_reduction(image, target_size)
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Wiener filter for noise reduction
        try:
            # Estimate noise variance
            noise_var = restoration.estimate_sigma(gray, average_sigmas=True) ** 2
            
            # Apply Wiener deconvolution
            denoised = restoration.wiener(gray, noise=noise_var)
            denoised = (denoised * 255).astype(np.uint8)
            
            return denoised, {
                'method': 'wiener_filter',
                'estimated_noise_variance': noise_var
            }
        except Exception as e:
            logger.warning(f"Advanced denoising failed: {e}, falling back to basic method")
            return self._step_noise_reduction(image, target_size)
    
    def _step_histogram_equalization(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Apply histogram equalization"""
        if not ADVANCED_PROCESSING_AVAILABLE:
            return self._step_contrast_enhancement(image, target_size)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive histogram equalization
        equalized = exposure.equalize_adapthist(gray, clip_limit=0.03)
        equalized = (equalized * 255).astype(np.uint8)
        
        return equalized, {
            'method': 'adaptive_histogram_equalization',
            'clip_limit': 0.03
        }
    
    def _step_unsharp_masking(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Apply unsharp masking for edge enhancement"""
        if not ADVANCED_PROCESSING_AVAILABLE:
            return self._step_sharpening(image, target_size)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply unsharp mask
        enhanced = filters.unsharp_mask(gray, radius=1, amount=1)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced, {
            'method': 'unsharp_mask',
            'radius': 1,
            'amount': 1
        }
    
    def _step_morphological_operations(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Apply morphological operations"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Apply closing to fill small gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed, {
            'operations': ['opening', 'closing'],
            'kernel_shape': 'rectangle',
            'kernel_size': (3, 3)
        }
    
    def _step_edge_enhancement(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Enhance edges in the image"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Laplacian for edge enhancement
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Combine with original image
        enhanced = cv2.addWeighted(gray, 0.8, laplacian, 0.2, 0)
        
        return enhanced, {
            'method': 'laplacian_enhancement',
            'original_weight': 0.8,
            'laplacian_weight': 0.2
        }
    
    def _calculate_enhancement_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Calculate metrics to measure enhancement quality"""
        metrics = {}
        
        # Convert both images to grayscale for comparison
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original.copy()
            
        if len(processed.shape) == 3:
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            proc_gray = processed.copy()
        
        # Ensure same size for comparison
        if orig_gray.shape != proc_gray.shape:
            proc_gray = cv2.resize(proc_gray, (orig_gray.shape[1], orig_gray.shape[0]))
        
        # Calculate contrast improvement
        orig_contrast = np.std(orig_gray)
        proc_contrast = np.std(proc_gray)
        metrics['contrast_improvement'] = proc_contrast / orig_contrast if orig_contrast > 0 else 1.0
        
        # Calculate sharpness (using variance of Laplacian)
        orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        proc_sharpness = cv2.Laplacian(proc_gray, cv2.CV_64F).var()
        metrics['sharpness_improvement'] = proc_sharpness / orig_sharpness if orig_sharpness > 0 else 1.0
        
        # Calculate brightness change
        orig_brightness = np.mean(orig_gray)
        proc_brightness = np.mean(proc_gray)
        metrics['brightness_change'] = proc_brightness / orig_brightness if orig_brightness > 0 else 1.0
        
        # Calculate histogram entropy (measure of detail)
        orig_entropy = self._calculate_entropy(orig_gray)
        proc_entropy = self._calculate_entropy(proc_gray)
        metrics['entropy_improvement'] = proc_entropy / orig_entropy if orig_entropy > 0 else 1.0
        
        return metrics
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate histogram entropy"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        histogram = histogram / histogram.sum()  # Normalize
        
        # Calculate entropy
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        return entropy
    
    def create_custom_pipeline(self, steps: List[str]) -> None:
        """Create a custom processing pipeline"""
        # Validate steps
        valid_steps = []
        for step in steps:
            if hasattr(self, f'_step_{step}'):
                valid_steps.append(step)
            else:
                logger.warning(f"Invalid processing step: {step}")
        
        if valid_steps:
            self.current_chain = valid_steps
            self.mode = ProcessingMode.CUSTOM
            logger.info(f"Custom pipeline created with {len(valid_steps)} steps")
        else:
            logger.error("No valid steps provided for custom pipeline")
    
    def batch_process_images(self, 
                           image_paths: List[str],
                           output_dir: Optional[str] = None,
                           target_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Process multiple images in batch"""
        results = {
            'total_images': len(image_paths),
            'processed_images': 0,
            'failed_images': 0,
            'processing_results': [],
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    results['failed_images'] += 1
                    continue
                
                # Process image
                result = self.process_image(image, target_size)
                
                # Save processed image if output directory provided
                if output_dir:
                    output_path = Path(output_dir) / f"processed_{Path(image_path).name}"
                    cv2.imwrite(str(output_path), result['processed_image'])
                    result['output_path'] = str(output_path)
                
                result['input_path'] = image_path
                result['batch_index'] = i
                
                results['processing_results'].append(result)
                results['processed_images'] += 1
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                results['failed_images'] += 1
        
        results['total_processing_time'] = time.time() - start_time
        results['average_processing_time'] = (
            results['total_processing_time'] / results['processed_images'] 
            if results['processed_images'] > 0 else 0
        )
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['images_processed'] > 0:
            stats['average_processing_time'] = (
                stats['total_processing_time'] / stats['images_processed']
            )
        else:
            stats['average_processing_time'] = 0.0
        
        stats['current_mode'] = self.mode.value
        stats['current_pipeline'] = self.current_chain
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'preprocessing_steps': []
        }

class LicensePlateImageProcessor(ImageProcessor):
    """Specialized image processor for license plates"""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.BALANCED):
        super().__init__(mode)
        
        # Override with license plate specific processing chains
        self.processing_chains[ProcessingMode.QUALITY] = [
            'resize_if_needed',
            'convert_color_space', 
            'noise_reduction',
            'contrast_enhancement',
            'plate_specific_enhancement',
            'text_enhancement',
            'final_sharpening'
        ]
    
    def _step_plate_specific_enhancement(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Enhancement specifically for license plates"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply gamma correction to enhance readability
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(gray, lookup_table)
        
        # Apply bilateral filter to smooth while preserving edges
        bilateral = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)
        
        return bilateral, {
            'gamma_correction': gamma,
            'bilateral_filter': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
        }
    
    def _step_text_enhancement(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Enhance text readability on license plates"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to create high contrast
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned, {
            'thresholding': 'OTSU',
            'morphology': 'closing',
            'kernel_size': (2, 2)
        }
    
    def _step_final_sharpening(self, image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Final sharpening step for license plates"""
        # Enhanced sharpening kernel for text
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        return sharpened, {
            'method': 'enhanced_sharpening',
            'kernel': 'text_optimized'
        }

def main():
    """Demo script for image processing pipeline"""
    print("🖼️  Image Processing Pipeline Demo")
    print("=" * 40)
    
    # Create sample test image
    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    # Test different processing modes
    modes = [ProcessingMode.FAST, ProcessingMode.BALANCED, ProcessingMode.QUALITY]
    
    for mode in modes:
        print(f"\\n🔧 Testing {mode.value.upper()} mode:")
        
        processor = ImageProcessor(mode)
        result = processor.process_image(test_image)
        
        print(f"  Processing time: {result['processing_time']:.3f}s")
        print(f"  Steps executed: {len(result['processing_steps'])}")
        print(f"  Enhancement metrics:")
        for metric, value in result['enhancement_metrics'].items():
            print(f"    {metric}: {value:.3f}")
    
    # Test license plate specific processor
    print(f"\\n🚗 Testing License Plate Processor:")
    lp_processor = LicensePlateImageProcessor(ProcessingMode.QUALITY)
    result = lp_processor.process_image(test_image)
    
    print(f"  Processing time: {result['processing_time']:.3f}s")
    print(f"  Steps executed: {len(result['processing_steps'])}")
    
    # Test custom pipeline
    print(f"\\n⚙️  Testing Custom Pipeline:")
    processor = ImageProcessor()
    processor.create_custom_pipeline(['resize_if_needed', 'noise_reduction', 'sharpening'])
    result = processor.process_image(test_image)
    
    print(f"  Custom steps: {processor.current_chain}")
    print(f"  Processing time: {result['processing_time']:.3f}s")
    
    # Show statistics
    stats = processor.get_processing_statistics()
    print(f"\\n📊 Processing Statistics:")
    print(f"  Images processed: {stats['images_processed']}")
    print(f"  Average time: {stats['average_processing_time']:.3f}s")
    print(f"  Current mode: {stats['current_mode']}")
    
    print(f"\\n✅ Image processing pipeline demo completed!")

if __name__ == "__main__":
    main()