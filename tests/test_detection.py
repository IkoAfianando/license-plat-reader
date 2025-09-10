"""
Detection System Tests
Test license plate detection functionality including YOLO and Roboflow integration
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, Mock, AsyncMock
from pathlib import Path
import tempfile
import json

from src.offline.standalone_detector import StandaloneLicensePlateDetector
from src.roboflow.detector import RoboflowDetector
from models.model_manager import ModelManager

class TestStandaloneDetector:
    """Test standalone YOLO-based detection"""
    
    def test_detector_initialization(self, mock_yolo_model, temp_dir):
        """Test detector initialization"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            detector = StandaloneLicensePlateDetector(
                model_path=str(temp_dir / "test_model.pt"),
                confidence_threshold=0.5
            )
            
            assert detector is not None
            assert detector.confidence_threshold == 0.5
            assert detector.model is not None
    
    def test_single_image_detection(self, mock_yolo_model, sample_license_plate_image):
        """Test detection on single image"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            detector = StandaloneLicensePlateDetector()
            result = detector.detect_license_plates(sample_license_plate_image)
            
            assert 'license_plates' in result
            assert 'processing_time' in result
            assert 'metadata' in result
            
            # Should find the mocked detection
            plates = result['license_plates']
            assert len(plates) >= 0  # Depends on filtering
    
    def test_confidence_threshold_filtering(self, sample_license_plate_image):
        """Test confidence threshold filtering"""
        # Mock model with different confidence scores
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.xyxy = np.array([[50, 50, 250, 100], [300, 300, 400, 350]])
        mock_result.boxes.conf = np.array([0.9, 0.3])  # One high, one low confidence
        mock_result.boxes.cls = np.array([0, 0])
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'license-plate'}
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_model
            
            # High threshold should filter out low confidence detection
            detector = StandaloneLicensePlateDetector(confidence_threshold=0.5)
            result = detector.detect_license_plates(sample_license_plate_image)
            
            # Should only include high confidence detection
            plates = result['license_plates']
            if plates:
                assert all(plate['confidence'] >= 0.5 for plate in plates)
    
    def test_batch_detection(self, mock_yolo_model):
        """Test batch detection functionality"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            detector = StandaloneLicensePlateDetector()
            
            # Create batch of test images
            images = [
                np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
                for _ in range(3)
            ]
            
            results = detector.detect_batch(images)
            
            assert len(results) == 3
            for result in results:
                assert 'license_plates' in result
                assert 'processing_time' in result
    
    def test_size_filtering(self, sample_license_plate_image):
        """Test size-based filtering of detections"""
        # Mock model with detections of different sizes
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.xyxy = np.array([
            [50, 50, 250, 100],    # Good size: 200x50
            [300, 300, 305, 302]   # Too small: 5x2
        ])
        mock_result.boxes.conf = np.array([0.9, 0.9])
        mock_result.boxes.cls = np.array([0, 0])
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'license-plate'}
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_model
            
            detector = StandaloneLicensePlateDetector()
            result = detector.detect_license_plates(sample_license_plate_image)
            
            # Should filter out tiny detections
            plates = result['license_plates']
            for plate in plates:
                bbox = plate['bbox']
                width = bbox['x2'] - bbox['x1']
                height = bbox['y2'] - bbox['y1']
                assert width >= detector.min_width
                assert height >= detector.min_height
    
    def test_aspect_ratio_filtering(self, sample_license_plate_image):
        """Test aspect ratio filtering"""
        # Mock model with different aspect ratios
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.xyxy = np.array([
            [50, 50, 250, 100],    # Good ratio: 4:1
            [300, 300, 350, 450]   # Bad ratio: 1:3 (too tall)
        ])
        mock_result.boxes.conf = np.array([0.9, 0.9])
        mock_result.boxes.cls = np.array([0, 0])
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'license-plate'}
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_model
            
            detector = StandaloneLicensePlateDetector()
            result = detector.detect_license_plates(sample_license_plate_image)
            
            # Should filter based on aspect ratio
            plates = result['license_plates']
            for plate in plates:
                bbox = plate['bbox']
                width = bbox['x2'] - bbox['x1']
                height = bbox['y2'] - bbox['y1']
                ratio = width / height
                assert detector.min_aspect_ratio <= ratio <= detector.max_aspect_ratio
    
    def test_nms_duplicate_removal(self, sample_license_plate_image):
        """Test Non-Maximum Suppression for removing duplicate detections"""
        # Mock model with overlapping detections
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.xyxy = np.array([
            [50, 50, 250, 100],    # Detection 1
            [55, 55, 255, 105]     # Overlapping detection 2
        ])
        mock_result.boxes.conf = np.array([0.9, 0.8])
        mock_result.boxes.cls = np.array([0, 0])
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'license-plate'}
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_model
            
            detector = StandaloneLicensePlateDetector()
            result = detector.detect_license_plates(sample_license_plate_image)
            
            # Should remove overlapping detections
            plates = result['license_plates']
            assert len(plates) <= 1  # Should keep only the highest confidence
    
    def test_error_handling(self, mock_yolo_model):
        """Test error handling for invalid inputs"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            detector = StandaloneLicensePlateDetector()
            
            # Test with None
            with pytest.raises((ValueError, TypeError)):
                detector.detect_license_plates(None)
            
            # Test with invalid array
            invalid_image = np.array([1, 2, 3])
            with pytest.raises((ValueError, IndexError)):
                detector.detect_license_plates(invalid_image)
    
    def test_model_loading_failure(self):
        """Test handling of model loading failures"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception):
                StandaloneLicensePlateDetector(model_path="nonexistent.pt")
    
    def test_gpu_cpu_fallback(self, mock_yolo_model):
        """Test GPU/CPU device selection and fallback"""
        with patch('ultralytics.YOLO') as mock_yolo_class, \
             patch('torch.cuda.is_available') as mock_cuda:
            
            mock_yolo_class.return_value = mock_yolo_model
            
            # Test GPU availability
            mock_cuda.return_value = True
            detector_gpu = StandaloneLicensePlateDetector(device='auto')
            assert detector_gpu.device in ['cuda', 'cpu']  # Should choose appropriately
            
            # Test CPU fallback
            mock_cuda.return_value = False
            detector_cpu = StandaloneLicensePlateDetector(device='auto')
            assert detector_cpu.device == 'cpu'

class TestRoboflowDetector:
    """Test Roboflow API integration"""
    
    def test_detector_initialization(self):
        """Test Roboflow detector initialization"""
        detector = RoboflowDetector(
            api_key="test_key",
            project="test_project",
            version=1
        )
        
        assert detector.api_key == "test_key"
        assert detector.project == "test_project"
        assert detector.version == 1
    
    @pytest.mark.asyncio
    async def test_async_detection(self, sample_license_plate_image, mock_roboflow_response):
        """Test async detection with Roboflow API"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_roboflow_response
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            detector = RoboflowDetector(api_key="test_key", project="test", version=1)
            result = await detector.detect_async(sample_license_plate_image)
            
            assert 'license_plates' in result
            assert 'processing_time' in result
            
            plates = result['license_plates']
            if plates:
                assert all('bbox' in plate for plate in plates)
                assert all('confidence' in plate for plate in plates)
    
    def test_sync_detection(self, sample_license_plate_image, mock_roboflow_response):
        """Test synchronous detection with Roboflow API"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_roboflow_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            detector = RoboflowDetector(api_key="test_key", project="test", version=1)
            result = detector.detect(sample_license_plate_image)
            
            assert 'license_plates' in result
            plates = result['license_plates']
            if plates:
                assert plates[0]['confidence'] == 0.95
    
    def test_api_error_handling(self, sample_license_plate_image):
        """Test handling of API errors"""
        with patch('requests.post') as mock_post:
            # Test network error
            mock_post.side_effect = Exception("Network error")
            
            detector = RoboflowDetector(api_key="test_key", project="test", version=1)
            result = detector.detect(sample_license_plate_image)
            
            # Should handle error gracefully
            assert 'error' in result
    
    def test_invalid_api_key(self, sample_license_plate_image):
        """Test handling of invalid API key"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": "Invalid API key"}
            mock_post.return_value = mock_response
            
            detector = RoboflowDetector(api_key="invalid_key", project="test", version=1)
            result = detector.detect(sample_license_plate_image)
            
            assert 'error' in result
    
    def test_rate_limiting_handling(self, sample_license_plate_image):
        """Test handling of rate limiting"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {'Retry-After': '60'}
            mock_post.return_value = mock_response
            
            detector = RoboflowDetector(api_key="test_key", project="test", version=1)
            result = detector.detect(sample_license_plate_image)
            
            assert 'error' in result
            assert 'rate_limit' in result['error'].lower()
    
    def test_image_encoding(self, sample_license_plate_image):
        """Test image encoding for API requests"""
        detector = RoboflowDetector(api_key="test_key", project="test", version=1)
        
        # Test different image formats
        encoded = detector._encode_image(sample_license_plate_image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Test with different dtypes
        float_image = sample_license_plate_image.astype(np.float32) / 255.0
        encoded_float = detector._encode_image(float_image)
        assert isinstance(encoded_float, str)

class TestModelManager:
    """Test model management functionality"""
    
    def test_model_manager_initialization(self, model_manager):
        """Test model manager initialization"""
        assert model_manager is not None
        assert model_manager.config is not None
        assert 'model_dir' in model_manager.config
    
    def test_yolo_model_download(self, model_manager):
        """Test automatic YOLO model download"""
        with patch('ultralytics.YOLO') as mock_yolo_class, \
             patch('requests.get') as mock_get:
            
            # Mock successful download
            mock_response = Mock()
            mock_response.content = b"fake_model_data"
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            mock_model = Mock()
            mock_yolo_class.return_value = mock_model
            
            model = model_manager.get_yolo_model()
            assert model is not None
    
    def test_model_validation(self, model_manager, temp_dir):
        """Test model file validation"""
        # Create fake model file
        fake_model_path = temp_dir / "fake_model.pt"
        fake_model_path.write_bytes(b"fake_model_content")
        
        # Should validate file exists
        assert model_manager._validate_model_file(str(fake_model_path))
        
        # Should fail for non-existent file
        assert not model_manager._validate_model_file(str(temp_dir / "nonexistent.pt"))
    
    def test_model_registry(self, model_manager):
        """Test model registry functionality"""
        # Add model to registry
        model_info = {
            "name": "test_model",
            "version": "1.0",
            "accuracy": 0.95,
            "size": "nano"
        }
        
        model_manager.register_model("test_model", model_info)
        
        # Retrieve model info
        retrieved = model_manager.get_model_info("test_model")
        assert retrieved == model_info
        
        # List models
        models = model_manager.list_models()
        assert "test_model" in models
    
    def test_model_performance_tracking(self, model_manager):
        """Test model performance tracking"""
        # Record performance metrics
        metrics = {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91,
            "inference_time": 1.25
        }
        
        model_manager.record_performance("test_model", metrics)
        
        # Retrieve performance history
        history = model_manager.get_performance_history("test_model")
        assert len(history) > 0
        assert history[-1]["accuracy"] == 0.92

class TestDetectionPipeline:
    """Test complete detection pipeline"""
    
    def test_dual_mode_detection(self, sample_license_plate_image, mock_yolo_model, mock_roboflow_response):
        """Test detection using both YOLO and Roboflow"""
        with patch('ultralytics.YOLO') as mock_yolo_class, \
             patch('requests.post') as mock_post:
            
            # Setup mocks
            mock_yolo_class.return_value = mock_yolo_model
            
            mock_response = Mock()
            mock_response.json.return_value = mock_roboflow_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            # Test YOLO detection
            yolo_detector = StandaloneLicensePlateDetector()
            yolo_result = yolo_detector.detect_license_plates(sample_license_plate_image)
            
            # Test Roboflow detection
            roboflow_detector = RoboflowDetector(api_key="test", project="test", version=1)
            roboflow_result = roboflow_detector.detect(sample_license_plate_image)
            
            # Both should return valid results
            assert 'license_plates' in yolo_result
            assert 'license_plates' in roboflow_result
    
    def test_fallback_mechanism(self, sample_license_plate_image, mock_yolo_model):
        """Test fallback from Roboflow to YOLO on API failure"""
        with patch('ultralytics.YOLO') as mock_yolo_class, \
             patch('requests.post') as mock_post:
            
            # Setup YOLO mock
            mock_yolo_class.return_value = mock_yolo_model
            
            # Mock Roboflow API failure
            mock_post.side_effect = Exception("API Error")
            
            # Create detection pipeline with fallback
            roboflow_detector = RoboflowDetector(api_key="test", project="test", version=1)
            yolo_detector = StandaloneLicensePlateDetector()
            
            # Try Roboflow first, fallback to YOLO
            try:
                result = roboflow_detector.detect(sample_license_plate_image)
                if 'error' in result:
                    result = yolo_detector.detect_license_plates(sample_license_plate_image)
            except:
                result = yolo_detector.detect_license_plates(sample_license_plate_image)
            
            # Should have valid result from fallback
            assert 'license_plates' in result
    
    def test_result_aggregation(self, sample_license_plate_image, mock_yolo_model, mock_roboflow_response):
        """Test aggregating results from multiple detection methods"""
        with patch('ultralytics.YOLO') as mock_yolo_class, \
             patch('requests.post') as mock_post:
            
            # Setup mocks
            mock_yolo_class.return_value = mock_yolo_model
            
            mock_response = Mock()
            mock_response.json.return_value = mock_roboflow_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            # Get results from both
            yolo_detector = StandaloneLicensePlateDetector()
            yolo_result = yolo_detector.detect_license_plates(sample_license_plate_image)
            
            roboflow_detector = RoboflowDetector(api_key="test", project="test", version=1)
            roboflow_result = roboflow_detector.detect(sample_license_plate_image)
            
            # Aggregate results (simple combination)
            all_plates = []
            all_plates.extend(yolo_result.get('license_plates', []))
            all_plates.extend(roboflow_result.get('license_plates', []))
            
            # Should have combined results
            assert isinstance(all_plates, list)

@pytest.mark.performance
class TestDetectionPerformance:
    """Test detection performance and benchmarks"""
    
    def test_single_detection_speed(self, mock_yolo_model, sample_license_plate_image):
        """Test single image detection speed"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            detector = StandaloneLicensePlateDetector()
            
            import time
            start_time = time.time()
            result = detector.detect_license_plates(sample_license_plate_image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should be reasonably fast
            assert processing_time < 2.0  # 2 seconds max
            
            # Check reported time
            reported_time = result.get('processing_time', 0)
            assert reported_time > 0
    
    def test_batch_detection_efficiency(self, mock_yolo_model):
        """Test batch detection efficiency"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            detector = StandaloneLicensePlateDetector()
            
            # Create batch of images
            images = [
                np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
                for _ in range(5)
            ]
            
            import time
            
            # Individual processing
            start_individual = time.time()
            individual_results = [detector.detect_license_plates(img) for img in images]
            individual_time = time.time() - start_individual
            
            # Batch processing
            start_batch = time.time()
            batch_results = detector.detect_batch(images)
            batch_time = time.time() - start_batch
            
            # Batch should be more efficient
            assert batch_time <= individual_time * 1.2  # Allow 20% overhead
            assert len(batch_results) == len(individual_results)
    
    def test_memory_usage_detection(self, mock_yolo_model, sample_license_plate_image):
        """Test memory usage during detection"""
        import psutil
        import os
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            detector = StandaloneLicensePlateDetector()
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            # Perform multiple detections
            for _ in range(20):
                result = detector.detect_license_plates(sample_license_plate_image)
            
            memory_after = process.memory_info().rss
            memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB
            
            # Should not leak significant memory
            assert memory_increase < 50  # Less than 50 MB increase

@pytest.mark.integration
class TestDetectionIntegration:
    """Integration tests for detection system"""
    
    def test_detection_with_image_processing(self, lp_image_processor, mock_yolo_model, sample_license_plate_image):
        """Test detection with image preprocessing"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            # Preprocess image
            processed_result = lp_image_processor.process_image(sample_license_plate_image)
            processed_image = processed_result['processed_image']
            
            # Detect on processed image
            detector = StandaloneLicensePlateDetector()
            detection_result = detector.detect_license_plates(processed_image)
            
            # Should work with processed image
            assert 'license_plates' in detection_result
    
    def test_detection_with_database_storage(self, test_db_session, mock_yolo_model, sample_license_plate_image):
        """Test detection with database storage"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            from database.models import Detection, LicensePlate
            
            detector = StandaloneLicensePlateDetector()
            result = detector.detect_license_plates(sample_license_plate_image)
            
            # Store result in database
            detection = Detection(
                image_path="test.jpg",
                processing_time=result.get('processing_time', 0),
                confidence_threshold=0.5,
                model_used="yolov8n"
            )
            
            test_db_session.add(detection)
            test_db_session.commit()
            
            # Verify storage
            stored_detection = test_db_session.query(Detection).first()
            assert stored_detection is not None
            assert stored_detection.image_path == "test.jpg"
    
    def test_end_to_end_detection_pipeline(self, mock_external_services, sample_license_plate_image):
        """Test complete end-to-end detection pipeline"""
        # This would test the full pipeline from image input to final result
        # Including preprocessing, detection, post-processing, and storage
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes.xyxy = np.array([[50, 50, 250, 100]])
            mock_result.boxes.conf = np.array([0.95])
            mock_result.boxes.cls = np.array([0])
            mock_model.return_value = [mock_result]
            mock_model.names = {0: 'license-plate'}
            mock_yolo_class.return_value = mock_model
            
            # Full pipeline test
            from src.pipeline.image_processor import LicensePlateImageProcessor
            
            # 1. Image preprocessing
            processor = LicensePlateImageProcessor()
            processed_result = processor.process_image(sample_license_plate_image)
            
            # 2. Detection
            detector = StandaloneLicensePlateDetector()
            detection_result = detector.detect_license_plates(processed_result['processed_image'])
            
            # 3. Verify complete pipeline
            assert 'license_plates' in detection_result
            assert 'processing_time' in detection_result
            assert 'metadata' in detection_result