"""
Performance Tests
Comprehensive performance testing for the license plate reader system
"""

import pytest
import time
import psutil
import os
import concurrent.futures
import threading
from typing import List, Dict, Any
import numpy as np
import statistics
from unittest.mock import patch, Mock

# Performance test markers
pytestmark = pytest.mark.performance

class TestSystemPerformance:
    """Test overall system performance"""
    
    def test_memory_usage_baseline(self):
        """Test baseline memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Baseline memory should be reasonable
        memory_mb = memory_info.rss / 1024 / 1024
        assert memory_mb < 500  # Less than 500 MB for test process
    
    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring"""
        # Monitor CPU usage during light operations
        cpu_before = psutil.cpu_percent(interval=1)
        
        # Perform some work
        for _ in range(1000):
            result = sum(range(100))
        
        cpu_after = psutil.cpu_percent(interval=1)
        
        # CPU usage should be measurable but not excessive
        assert cpu_after >= 0
        assert cpu_after <= 100
    
    def test_disk_io_performance(self, temp_dir):
        """Test disk I/O performance"""
        import time
        
        # Test file write performance
        test_file = temp_dir / "performance_test.txt"
        data = "x" * 1024 * 1024  # 1MB of data
        
        start_time = time.time()
        test_file.write_text(data)
        write_time = time.time() - start_time
        
        # Test file read performance
        start_time = time.time()
        read_data = test_file.read_text()
        read_time = time.time() - start_time
        
        # I/O should be reasonably fast
        assert write_time < 5.0  # 5 seconds max
        assert read_time < 2.0   # 2 seconds max
        assert len(read_data) == len(data)

class TestImageProcessingPerformance:
    """Test image processing performance"""
    
    def test_single_image_processing_speed(self, image_processor, sample_image):
        """Test single image processing speed"""
        processing_times = []
        
        # Run multiple iterations for consistent measurement
        for _ in range(10):
            start_time = time.time()
            result = image_processor.process_image(sample_image)
            end_time = time.time()
            
            processing_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        # Performance expectations
        assert avg_time < 2.0    # Average under 2 seconds
        assert max_time < 5.0    # Max under 5 seconds
        assert min_time > 0.001  # Minimum reasonable time
        
        # Consistency check (standard deviation should be reasonable)
        std_dev = statistics.stdev(processing_times)
        assert std_dev < avg_time * 0.5  # Std dev less than 50% of mean
    
    def test_batch_processing_efficiency(self, image_processor):
        """Test batch processing efficiency vs individual processing"""
        # Create test images
        batch_size = 10
        images = [
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]
        
        # Time individual processing
        start_time = time.time()
        individual_results = []
        for img in images:
            result = image_processor.process_image(img)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Time batch processing
        start_time = time.time()
        batch_results = image_processor.process_batch(images)
        batch_time = time.time() - start_time
        
        # Batch should be more efficient
        efficiency_ratio = batch_time / individual_time
        assert efficiency_ratio < 1.2  # Batch shouldn't be more than 20% slower
        
        # Verify results are equivalent
        assert len(batch_results) == len(individual_results)
        assert len(batch_results) == batch_size
    
    def test_memory_usage_during_processing(self, image_processor):
        """Test memory usage during image processing"""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Process multiple large images
        large_images = [
            np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        for img in large_images:
            result = image_processor.process_image(img)
            del result  # Explicit cleanup
        
        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB
        
        # Should not leak significant memory
        assert memory_increase < 100  # Less than 100 MB increase
    
    def test_different_image_sizes_performance(self, image_processor):
        """Test performance with different image sizes"""
        sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
        performance_data = []
        
        for width, height in sizes:
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            start_time = time.time()
            result = image_processor.process_image(img)
            processing_time = time.time() - start_time
            
            performance_data.append({
                'size': (width, height),
                'pixels': width * height,
                'time': processing_time
            })
        
        # Processing time should scale reasonably with image size
        for i in range(1, len(performance_data)):
            prev_data = performance_data[i-1]
            curr_data = performance_data[i]
            
            # Larger images should take more time, but not exponentially more
            time_ratio = curr_data['time'] / prev_data['time']
            pixel_ratio = curr_data['pixels'] / prev_data['pixels']
            
            # Time should scale roughly linearly with pixels (allow some overhead)
            assert time_ratio <= pixel_ratio * 2

class TestDetectionPerformance:
    """Test license plate detection performance"""
    
    def test_yolo_detection_speed(self, mock_yolo_model, sample_license_plate_image):
        """Test YOLO detection speed"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            from src.offline.standalone_detector import StandaloneLicensePlateDetector
            detector = StandaloneLicensePlateDetector()
            
            detection_times = []
            
            # Run multiple detections
            for _ in range(20):
                start_time = time.time()
                result = detector.detect_license_plates(sample_license_plate_image)
                detection_time = time.time() - start_time
                detection_times.append(detection_time)
            
            # Calculate performance metrics
            avg_time = statistics.mean(detection_times)
            max_time = max(detection_times)
            min_time = min(detection_times)
            
            # Performance expectations
            assert avg_time < 1.0   # Average under 1 second
            assert max_time < 3.0   # Max under 3 seconds
            assert min_time > 0.01  # Minimum reasonable time
    
    def test_roboflow_api_response_time(self, sample_license_plate_image, mock_roboflow_response):
        """Test Roboflow API response time simulation"""
        with patch('requests.post') as mock_post:
            # Simulate network delay
            def delayed_response(*args, **kwargs):
                time.sleep(0.1)  # 100ms network delay
                mock_response = Mock()
                mock_response.json.return_value = mock_roboflow_response
                mock_response.status_code = 200
                return mock_response
            
            mock_post.side_effect = delayed_response
            
            from src.roboflow.detector import RoboflowDetector
            detector = RoboflowDetector(api_key="test", project="test", version=1)
            
            response_times = []
            
            for _ in range(5):
                start_time = time.time()
                result = detector.detect(sample_license_plate_image)
                response_time = time.time() - start_time
                response_times.append(response_time)
            
            avg_response_time = statistics.mean(response_times)
            
            # Should include network delay but not be excessive
            assert 0.05 < avg_response_time < 2.0  # Between 50ms and 2 seconds
    
    def test_concurrent_detection_performance(self, mock_yolo_model, sample_license_plate_image):
        """Test concurrent detection performance"""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            
            from src.offline.standalone_detector import StandaloneLicensePlateDetector
            detector = StandaloneLicensePlateDetector()
            
            def detect_image():
                start_time = time.time()
                result = detector.detect_license_plates(sample_license_plate_image)
                return time.time() - start_time
            
            # Test sequential processing
            start_time = time.time()
            sequential_times = [detect_image() for _ in range(10)]
            sequential_total = time.time() - start_time
            
            # Test concurrent processing
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                concurrent_times = list(executor.map(lambda _: detect_image(), range(10)))
            concurrent_total = time.time() - start_time
            
            # Concurrent should be faster (or at least not much slower)
            speedup_ratio = sequential_total / concurrent_total
            assert speedup_ratio >= 0.8  # Allow some overhead

class TestAPIPerformance:
    """Test API endpoint performance"""
    
    def test_health_endpoint_response_time(self, test_client):
        """Test health endpoint response time"""
        response_times = []
        
        for _ in range(50):
            start_time = time.time()
            response = test_client.get("/health")
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            assert response.status_code == 200
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        # Health checks should be very fast
        assert avg_response_time < 0.1  # 100ms average
        assert max_response_time < 0.5  # 500ms max
    
    def test_detection_endpoint_performance(self, test_client, sample_image, auth_headers):
        """Test detection endpoint performance"""
        from PIL import Image
        import io
        
        # Convert image to bytes
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {
                "license_plates": [],
                "processing_time": 0.5
            }
            
            response_times = []
            
            for _ in range(10):
                img_bytes.seek(0)
                
                start_time = time.time()
                response = test_client.post(
                    "/detect/image",
                    files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                assert response.status_code == 200
            
            avg_response_time = statistics.mean(response_times)
            
            # API should respond quickly (including network overhead)
            assert avg_response_time < 2.0  # 2 seconds max including processing
    
    def test_concurrent_api_requests(self, test_client, sample_image, auth_headers):
        """Test concurrent API request handling"""
        from PIL import Image
        import io
        
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {
                "license_plates": [],
                "processing_time": 0.1
            }
            
            def make_request():
                start_time = time.time()
                response = test_client.post(
                    "/detect/image",
                    files={"image": ("test.jpg", io.BytesIO(img_data), "image/jpeg")},
                    headers=auth_headers
                )
                request_time = time.time() - start_time
                return response.status_code, request_time
            
            # Test concurrent requests
            num_requests = 20
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(lambda _: make_request(), range(num_requests)))
            
            total_time = time.time() - start_time
            
            # Verify all requests succeeded
            status_codes = [result[0] for result in results]
            request_times = [result[1] for result in results]
            
            successful_requests = sum(1 for code in status_codes if code in [200, 429])
            assert successful_requests >= num_requests * 0.8  # At least 80% success
            
            # Average request time should be reasonable
            avg_request_time = statistics.mean(request_times)
            assert avg_request_time < 3.0

class TestDatabasePerformance:
    """Test database performance"""
    
    def test_bulk_insert_performance(self, test_db_session):
        """Test bulk insert performance"""
        from database.models import User, Detection
        
        # Create users for bulk insert
        users = [
            User(username=f"user{i}", email=f"user{i}@test.com", hashed_password="hash")
            for i in range(1000)
        ]
        
        start_time = time.time()
        test_db_session.bulk_save_objects(users)
        test_db_session.commit()
        bulk_time = time.time() - start_time
        
        # Bulk insert should be fast
        assert bulk_time < 5.0  # Less than 5 seconds for 1000 records
        
        # Verify all records were inserted
        user_count = test_db_session.query(User).count()
        assert user_count == 1000
    
    def test_query_performance_with_indexes(self, test_db_session):
        """Test query performance with proper indexing"""
        from database.models import User, Detection
        
        # Create test data
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Create many detections
        detections = [
            Detection(
                user_id=user.id,
                image_path=f"/img{i}.jpg",
                processing_time=1.0 + (i * 0.001)
            )
            for i in range(10000)
        ]
        
        test_db_session.bulk_save_objects(detections)
        test_db_session.commit()
        
        # Test indexed query performance
        start_time = time.time()
        user_detections = test_db_session.query(Detection).filter(
            Detection.user_id == user.id
        ).limit(100).all()
        query_time = time.time() - start_time
        
        # Indexed query should be fast
        assert query_time < 1.0  # Less than 1 second
        assert len(user_detections) == 100
    
    def test_complex_query_performance(self, test_db_session):
        """Test complex query performance"""
        from database.models import User, Detection, LicensePlate
        from sqlalchemy import func
        from datetime import datetime, timedelta
        
        # Create test data
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Create detections and plates
        for i in range(100):
            detection = Detection(
                user_id=user.id,
                image_path=f"/img{i}.jpg",
                processing_time=1.0,
                created_at=datetime.utcnow() - timedelta(days=i % 30)
            )
            test_db_session.add(detection)
            test_db_session.commit()
            
            # Add plates to some detections
            if i % 3 == 0:
                plate = LicensePlate(
                    detection_id=detection.id,
                    text=f"ABC{i:03d}",
                    confidence=0.9,
                    bbox_x1=100, bbox_y1=100, bbox_x2=200, bbox_y2=150,
                    region="US"
                )
                test_db_session.add(plate)
        
        test_db_session.commit()
        
        # Complex query: Get detection stats by day with plate counts
        start_time = time.time()
        results = test_db_session.query(
            func.date(Detection.created_at).label('date'),
            func.count(Detection.id).label('detection_count'),
            func.count(LicensePlate.id).label('plate_count')
        ).outerjoin(LicensePlate).group_by(
            func.date(Detection.created_at)
        ).all()
        query_time = time.time() - start_time
        
        # Complex query should complete reasonably fast
        assert query_time < 2.0  # Less than 2 seconds
        assert len(results) > 0

class TestMemoryLeakDetection:
    """Test for memory leaks"""
    
    def test_repeated_processing_memory_stability(self, image_processor, sample_image):
        """Test memory stability during repeated processing"""
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss
        memory_samples = [initial_memory]
        
        # Perform repeated processing
        for i in range(50):
            result = image_processor.process_image(sample_image)
            del result  # Explicit cleanup
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                memory_samples.append(process.memory_info().rss)
        
        # Check for memory leaks
        memory_growth = [(memory_samples[i] - memory_samples[i-1]) / 1024 / 1024 
                        for i in range(1, len(memory_samples))]
        
        # Should not continuously grow
        avg_growth = statistics.mean(memory_growth)
        assert avg_growth < 5.0  # Less than 5 MB average growth per 10 iterations
    
    def test_session_cleanup_memory(self, test_db_session):
        """Test database session memory cleanup"""
        from database.models import User
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and query many objects
        for batch in range(10):
            # Create objects
            users = [
                User(username=f"user{batch}_{i}", email=f"user{batch}_{i}@test.com", hashed_password="hash")
                for i in range(100)
            ]
            test_db_session.bulk_save_objects(users)
            test_db_session.commit()
            
            # Query objects
            all_users = test_db_session.query(User).all()
            del all_users
            
            # Explicit cleanup
            test_db_session.expunge_all()
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024
        
        # Should not accumulate excessive memory
        assert memory_increase < 50  # Less than 50 MB increase

class TestStressTest:
    """Stress tests for system limits"""
    
    @pytest.mark.slow
    def test_high_volume_processing(self, image_processor):
        """Test processing high volume of images"""
        # Create many small images
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(500)
        ]
        
        start_time = time.time()
        
        # Process all images
        results = []
        for img in images:
            try:
                result = image_processor.process_image(img)
                results.append(result)
            except Exception as e:
                pytest.fail(f"Failed to process image: {e}")
        
        total_time = time.time() - start_time
        
        # Verify all processed successfully
        assert len(results) == 500
        
        # Should complete within reasonable time
        assert total_time < 60  # 1 minute for 500 small images
        
        # Average processing time should be reasonable
        avg_time = total_time / len(images)
        assert avg_time < 0.2  # Less than 200ms per small image
    
    @pytest.mark.slow
    def test_long_running_stability(self, image_processor, sample_image):
        """Test system stability during long running operations"""
        process = psutil.Process(os.getpid())
        
        start_time = time.time()
        run_duration = 30  # 30 seconds
        iterations = 0
        errors = 0
        
        while time.time() - start_time < run_duration:
            try:
                result = image_processor.process_image(sample_image)
                iterations += 1
                
                # Check memory occasionally
                if iterations % 50 == 0:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    assert memory_mb < 1000  # Less than 1GB
                
            except Exception:
                errors += 1
        
        # Should process many iterations successfully
        assert iterations > 100  # At least 100 iterations in 30 seconds
        
        # Error rate should be very low
        error_rate = errors / iterations if iterations > 0 else 1
        assert error_rate < 0.01  # Less than 1% error rate
    
    def test_resource_exhaustion_handling(self, image_processor):
        """Test handling of resource exhaustion"""
        # Try to create very large images to test memory limits
        large_images = []
        
        try:
            for i in range(10):
                # Create progressively larger images
                size = 1000 + (i * 500)
                img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                large_images.append(img)
                
                # Try to process
                result = image_processor.process_image(img)
                assert result is not None
                
        except MemoryError:
            # Expected behavior - system should handle gracefully
            assert len(large_images) > 0  # Should process at least some images
        except Exception as e:
            # Other exceptions should be handled gracefully
            assert "memory" in str(e).lower() or "size" in str(e).lower()

@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance comparison"""
    
    def test_processing_pipeline_benchmark(self, performance_test_config):
        """Benchmark the complete processing pipeline"""
        from src.pipeline.image_processor import LicensePlateImageProcessor
        
        processor = LicensePlateImageProcessor()
        
        # Standard test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(5):
            processor.process_image(test_image)
        
        # Benchmark runs
        times = []
        for _ in range(50):
            start_time = time.time()
            result = processor.process_image(test_image)
            processing_time = time.time() - start_time
            times.append(processing_time)
        
        # Calculate benchmark metrics
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        
        # Log benchmark results
        print(f"\nProcessing Pipeline Benchmark:")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Median time: {median_time:.3f}s")
        print(f"95th percentile: {p95_time:.3f}s")
        
        # Performance assertions
        assert avg_time < performance_test_config['max_response_time']
        assert p95_time < performance_test_config['max_response_time'] * 2
    
    def test_throughput_benchmark(self, performance_test_config):
        """Benchmark system throughput"""
        from src.pipeline.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        test_images = [
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
            for _ in range(100)
        ]
        
        start_time = time.time()
        
        # Process all images
        for img in test_images:
            result = processor.process_image(img)
        
        total_time = time.time() - start_time
        throughput = len(test_images) / total_time
        
        print(f"\nThroughput Benchmark:")
        print(f"Images processed: {len(test_images)}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Throughput: {throughput:.2f} images/second")
        
        # Throughput should be reasonable
        assert throughput > 10  # At least 10 images per second