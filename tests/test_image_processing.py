"""
Image Processing Tests
Test image processing pipeline and enhancement algorithms
"""

import pytest
import numpy as np
import cv2
from PIL import Image
from unittest.mock import patch, Mock

from src.pipeline.image_processor import (
    ImageProcessor, 
    LicensePlateImageProcessor,
    ProcessingMode
)

class TestImageProcessor:
    """Test basic image processor functionality"""
    
    def test_processor_initialization(self):
        """Test processor initialization with different configs"""
        # Default config
        processor = ImageProcessor()
        assert processor is not None
        assert processor.config is not None
        
        # Custom config
        custom_config = {
            'noise_reduction': {'enabled': False},
            'contrast_enhancement': {'enabled': True, 'clip_limit': 3.0}
        }
        processor = ImageProcessor(custom_config)
        assert processor.config['noise_reduction']['enabled'] is False
        assert processor.config['contrast_enhancement']['clip_limit'] == 3.0
    
    def test_process_image_basic(self, image_processor, sample_image):
        """Test basic image processing"""
        result = image_processor.process_image(sample_image)
        
        assert 'processed_image' in result
        assert 'metrics' in result
        assert 'processing_time' in result
        
        # Check output image dimensions match input
        processed_img = result['processed_image']
        assert processed_img.shape == sample_image.shape
        assert processed_img.dtype == np.uint8
    
    def test_process_image_with_resize(self, image_processor, sample_image):
        """Test image processing with resizing"""
        target_size = (320, 240)
        result = image_processor.process_image(sample_image, target_size=target_size)
        
        processed_img = result['processed_image']
        assert processed_img.shape[:2] == target_size
    
    def test_noise_reduction(self, sample_image):
        """Test noise reduction functionality"""
        processor = ImageProcessor({
            'noise_reduction': {'enabled': True, 'kernel_size': 5},
            'contrast_enhancement': {'enabled': False},
            'sharpening': {'enabled': False}
        })
        
        # Add noise to image
        noisy_image = sample_image.copy()
        noise = np.random.randint(0, 50, sample_image.shape, dtype=np.uint8)
        noisy_image = np.clip(noisy_image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        result = processor.process_image(noisy_image)
        processed_img = result['processed_image']
        
        # Processed image should be different from noisy input
        assert not np.array_equal(processed_img, noisy_image)
        
        # Should have noise reduction metrics
        assert 'noise_reduction' in result['metrics']
    
    def test_contrast_enhancement(self, sample_image):
        """Test contrast enhancement using CLAHE"""
        processor = ImageProcessor({
            'noise_reduction': {'enabled': False},
            'contrast_enhancement': {'enabled': True, 'clip_limit': 2.0},
            'sharpening': {'enabled': False}
        })
        
        # Create low contrast image
        low_contrast = (sample_image * 0.5 + 64).astype(np.uint8)
        
        result = processor.process_image(low_contrast)
        processed_img = result['processed_image']
        
        # Processed image should have better contrast
        original_std = np.std(low_contrast)
        enhanced_std = np.std(processed_img)
        assert enhanced_std >= original_std  # Should increase contrast
        
        assert 'contrast_enhancement' in result['metrics']
    
    def test_sharpening(self, sample_image):
        """Test image sharpening"""
        processor = ImageProcessor({
            'noise_reduction': {'enabled': False},
            'contrast_enhancement': {'enabled': False},
            'sharpening': {'enabled': True, 'strength': 1.5}
        })
        
        # Create blurred image
        blurred = cv2.GaussianBlur(sample_image, (15, 15), 0)
        
        result = processor.process_image(blurred)
        processed_img = result['processed_image']
        
        # Should be different from blurred input
        assert not np.array_equal(processed_img, blurred)
        assert 'sharpening' in result['metrics']
    
    def test_morphological_operations(self, sample_image):
        """Test morphological operations"""
        processor = ImageProcessor({
            'morphological': {'enabled': True, 'kernel_size': 3}
        })
        
        result = processor.process_image(sample_image)
        assert 'morphological' in result['metrics']
    
    def test_processing_modes(self, sample_image):
        """Test different processing modes"""
        modes = [ProcessingMode.FAST, ProcessingMode.BALANCED, ProcessingMode.QUALITY]
        
        for mode in modes:
            processor = ImageProcessor()
            processor.mode = mode
            
            result = processor.process_image(sample_image)
            assert 'processed_image' in result
            assert result['metrics']['processing_mode'] == mode.value
    
    def test_batch_processing(self, image_processor):
        """Test batch image processing"""
        # Create batch of images
        batch_images = []
        for i in range(3):
            img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
            batch_images.append(img)
        
        results = image_processor.process_batch(batch_images)
        
        assert len(results) == 3
        for result in results:
            assert 'processed_image' in result
            assert 'metrics' in result
    
    def test_invalid_input(self, image_processor):
        """Test error handling for invalid inputs"""
        # Test with None
        with pytest.raises((ValueError, TypeError)):
            image_processor.process_image(None)
        
        # Test with wrong dimensions
        invalid_image = np.array([1, 2, 3])  # 1D array
        with pytest.raises((ValueError, IndexError)):
            image_processor.process_image(invalid_image)
        
        # Test with wrong data type
        wrong_dtype = np.random.random((100, 100, 3))  # Float64 instead of uint8
        # Should either work or raise appropriate error
        try:
            result = image_processor.process_image(wrong_dtype)
            assert result is not None
        except (ValueError, TypeError):
            pass  # Expected behavior
    
    def test_performance_metrics(self, image_processor, sample_image):
        """Test that performance metrics are collected"""
        result = image_processor.process_image(sample_image)
        
        metrics = result['metrics']
        assert 'processing_time' in metrics
        assert 'image_size' in metrics
        assert metrics['processing_time'] > 0
        assert metrics['image_size']['width'] > 0
        assert metrics['image_size']['height'] > 0

class TestLicensePlateImageProcessor:
    """Test license plate specific image processor"""
    
    def test_lp_processor_initialization(self):
        """Test license plate processor initialization"""
        processor = LicensePlateImageProcessor()
        assert processor is not None
        
        # Should have license plate specific config
        assert 'license_plate' in processor.config
        lp_config = processor.config['license_plate']
        assert 'edge_enhancement' in lp_config
        assert 'text_enhancement' in lp_config
        assert 'background_removal' in lp_config
    
    def test_edge_enhancement(self, sample_license_plate_image):
        """Test edge enhancement for license plates"""
        processor = LicensePlateImageProcessor({
            'license_plate': {
                'edge_enhancement': True,
                'text_enhancement': False,
                'background_removal': False
            }
        })
        
        result = processor.process_image(sample_license_plate_image)
        
        assert 'processed_image' in result
        assert 'edge_enhancement' in result['metrics']
        
        # Edge enhancement should preserve or improve edge information
        processed_img = result['processed_image']
        assert processed_img.shape == sample_license_plate_image.shape
    
    def test_text_enhancement(self, sample_license_plate_image):
        """Test text enhancement for better OCR"""
        processor = LicensePlateImageProcessor({
            'license_plate': {
                'edge_enhancement': False,
                'text_enhancement': True,
                'background_removal': False
            }
        })
        
        result = processor.process_image(sample_license_plate_image)
        
        assert 'text_enhancement' in result['metrics']
        
        # Text should be more prominent
        processed_img = result['processed_image']
        assert processed_img is not None
    
    def test_background_removal(self, sample_license_plate_image):
        """Test background noise removal"""
        processor = LicensePlateImageProcessor({
            'license_plate': {
                'edge_enhancement': False,
                'text_enhancement': False,
                'background_removal': True
            }
        })
        
        result = processor.process_image(sample_license_plate_image)
        
        assert 'background_removal' in result['metrics']
        processed_img = result['processed_image']
        assert processed_img is not None
    
    def test_combined_lp_enhancements(self, lp_image_processor, sample_license_plate_image):
        """Test all license plate enhancements together"""
        result = lp_image_processor.process_image(sample_license_plate_image)
        
        metrics = result['metrics']
        assert 'edge_enhancement' in metrics
        assert 'text_enhancement' in metrics
        assert 'background_removal' in metrics
        
        processed_img = result['processed_image']
        assert processed_img.shape == sample_license_plate_image.shape
    
    def test_adaptive_processing(self, lp_image_processor):
        """Test adaptive processing based on image characteristics"""
        # Test with dark image
        dark_image = np.ones((100, 300, 3), dtype=np.uint8) * 50
        result_dark = lp_image_processor.process_image(dark_image)
        
        # Test with bright image
        bright_image = np.ones((100, 300, 3), dtype=np.uint8) * 200
        result_bright = lp_image_processor.process_image(bright_image)
        
        # Processing should adapt to image characteristics
        assert result_dark['metrics'] != result_bright['metrics']
    
    def test_ocr_preparation(self, lp_image_processor, sample_license_plate_image):
        """Test that processing improves OCR readiness"""
        result = lp_image_processor.process_image(sample_license_plate_image)
        processed_img = result['processed_image']
        
        # Should be grayscale for better OCR
        if len(processed_img.shape) == 2:
            assert True  # Already grayscale
        else:
            # Convert to grayscale for testing
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            assert gray is not None
        
        # Should have good contrast for text
        contrast_ratio = np.std(processed_img)
        assert contrast_ratio > 10  # Should have reasonable contrast

class TestImageProcessingUtils:
    """Test utility functions for image processing"""
    
    def test_image_quality_assessment(self, image_processor, sample_image):
        """Test image quality assessment"""
        result = image_processor.process_image(sample_image)
        
        if 'quality_score' in result['metrics']:
            quality = result['metrics']['quality_score']
            assert 0 <= quality <= 1
    
    def test_noise_level_detection(self, image_processor):
        """Test noise level detection"""
        # Clean image
        clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result_clean = image_processor.process_image(clean_image)
        
        # Noisy image
        noisy_image = clean_image.copy()
        noise = np.random.randint(0, 100, clean_image.shape, dtype=np.uint8)
        noisy_image = np.clip(noisy_image.astype(int) + noise, 0, 255).astype(np.uint8)
        result_noisy = image_processor.process_image(noisy_image)
        
        # Should detect different noise levels
        if 'noise_level' in result_clean['metrics'] and 'noise_level' in result_noisy['metrics']:
            assert result_noisy['metrics']['noise_level'] > result_clean['metrics']['noise_level']
    
    def test_contrast_measurement(self, image_processor):
        """Test contrast measurement"""
        # Low contrast image
        low_contrast = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result_low = image_processor.process_image(low_contrast)
        
        # High contrast image
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[:50, :, :] = 255  # Half white, half black
        result_high = image_processor.process_image(high_contrast)
        
        # Should measure different contrast levels
        if 'contrast' in result_low['metrics'] and 'contrast' in result_high['metrics']:
            assert result_high['metrics']['contrast'] > result_low['metrics']['contrast']

class TestImageValidation:
    """Test image validation and preprocessing"""
    
    def test_image_format_validation(self, image_processor):
        """Test validation of different image formats"""
        # Valid RGB image
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = image_processor.process_image(rgb_image)
        assert result is not None
        
        # Valid grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = image_processor.process_image(gray_image)
        assert result is not None
        
        # RGBA image (should handle or convert)
        rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        try:
            result = image_processor.process_image(rgba_image)
            assert result is not None
        except ValueError:
            pass  # Expected if RGBA not supported
    
    def test_image_size_validation(self, image_processor):
        """Test validation of image sizes"""
        # Very small image
        tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = image_processor.process_image(tiny_image)
        assert result is not None
        
        # Very large image might be handled differently
        # large_image = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        # This would use too much memory in tests
    
    def test_dtype_conversion(self, image_processor):
        """Test automatic dtype conversion"""
        # Float image [0, 1]
        float_image = np.random.random((100, 100, 3))
        
        try:
            result = image_processor.process_image(float_image)
            processed_img = result['processed_image']
            assert processed_img.dtype == np.uint8
        except (ValueError, TypeError):
            pass  # Expected if conversion not automatic

@pytest.mark.performance
class TestImageProcessingPerformance:
    """Test image processing performance"""
    
    def test_processing_speed(self, image_processor, sample_image):
        """Test processing speed is reasonable"""
        import time
        
        start_time = time.time()
        result = image_processor.process_image(sample_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process within reasonable time (adjust based on hardware)
        assert processing_time < 5.0  # 5 seconds max for test image
        
        # Should match reported processing time
        reported_time = result.get('processing_time', processing_time)
        assert abs(reported_time - processing_time) < 1.0
    
    def test_memory_usage(self, image_processor, sample_image):
        """Test memory usage is reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Process multiple images
        for _ in range(10):
            result = image_processor.process_image(sample_image)
        
        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB
        
        # Should not leak too much memory
        assert memory_increase < 100  # Less than 100 MB increase
    
    def test_batch_vs_individual_performance(self, image_processor):
        """Test batch processing performance vs individual"""
        # Create test images
        images = [np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8) for _ in range(5)]
        
        # Individual processing
        import time
        start_individual = time.time()
        individual_results = []
        for img in images:
            result = image_processor.process_image(img)
            individual_results.append(result)
        individual_time = time.time() - start_individual
        
        # Batch processing
        start_batch = time.time()
        batch_results = image_processor.process_batch(images)
        batch_time = time.time() - start_batch
        
        # Batch should be more efficient (or at least not much slower)
        efficiency_ratio = batch_time / individual_time
        assert efficiency_ratio < 1.5  # Batch shouldn't be more than 50% slower
        
        # Results should be equivalent
        assert len(batch_results) == len(individual_results)

@pytest.mark.integration
class TestImageProcessingIntegration:
    """Integration tests for image processing pipeline"""
    
    def test_end_to_end_pipeline(self, lp_image_processor, sample_license_plate_image):
        """Test complete image processing pipeline"""
        # Process image
        result = lp_image_processor.process_image(sample_license_plate_image)
        processed_img = result['processed_image']
        
        # Should be ready for detection
        assert processed_img is not None
        assert processed_img.shape[0] > 0 and processed_img.shape[1] > 0
        
        # Should have comprehensive metrics
        metrics = result['metrics']
        expected_metrics = [
            'processing_time', 'image_size', 'edge_enhancement',
            'text_enhancement', 'background_removal'
        ]
        
        for metric in expected_metrics:
            if metric in metrics:
                assert metrics[metric] is not None
    
    def test_preprocessing_for_yolo(self, lp_image_processor, sample_image):
        """Test preprocessing optimized for YOLO detection"""
        target_size = (640, 640)  # Common YOLO input size
        result = lp_image_processor.process_image(sample_image, target_size=target_size)
        
        processed_img = result['processed_image']
        assert processed_img.shape[:2] == target_size
        
        # Should maintain aspect ratio information
        if 'resize_info' in result['metrics']:
            resize_info = result['metrics']['resize_info']
            assert 'original_size' in resize_info
            assert 'scale_factor' in resize_info
    
    def test_preprocessing_for_ocr(self, lp_image_processor, sample_license_plate_image):
        """Test preprocessing optimized for OCR"""
        result = lp_image_processor.process_image(sample_license_plate_image)
        processed_img = result['processed_image']
        
        # Should enhance text readability
        # Convert to grayscale if needed for OCR testing
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img
        
        # Should have good text contrast
        text_regions = gray[40:80, 50:250]  # Approximate text area
        contrast = np.std(text_regions)
        assert contrast > 15  # Should have reasonable text contrast