"""
API Endpoint Tests
Test all FastAPI endpoints for the license plate reader system
"""

import pytest
import json
import io
from PIL import Image
import numpy as np
from unittest.mock import patch, Mock

class TestHealthEndpoints:
    """Test health check and status endpoints"""
    
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data
        assert "version" in data
    
    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        # Prometheus metrics should be text/plain
        assert "text/plain" in response.headers.get("content-type", "")

class TestAuthenticationEndpoints:
    """Test authentication endpoints"""
    
    def test_login_success(self, test_client):
        """Test successful login"""
        with patch('src.api.auth.verify_password') as mock_verify, \
             patch('src.api.auth.get_user_by_username') as mock_get_user:
            
            mock_user = Mock()
            mock_user.username = "testuser"
            mock_user.role = "user"
            mock_user.is_active = True
            
            mock_verify.return_value = True
            mock_get_user.return_value = mock_user
            
            response = test_client.post("/auth/login", json={
                "username": "testuser",
                "password": "testpass"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials"""
        with patch('src.api.auth.verify_password') as mock_verify:
            mock_verify.return_value = False
            
            response = test_client.post("/auth/login", json={
                "username": "testuser",
                "password": "wrongpass"
            })
            
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
    
    def test_protected_endpoint_without_token(self, test_client):
        """Test accessing protected endpoint without token"""
        response = test_client.get("/auth/me")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_valid_token(self, test_client, auth_headers):
        """Test accessing protected endpoint with valid token"""
        with patch('src.api.auth.decode_token') as mock_decode:
            mock_decode.return_value = {"username": "testuser", "role": "user"}
            
            response = test_client.get("/auth/me", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["username"] == "testuser"

class TestDetectionEndpoints:
    """Test license plate detection endpoints"""
    
    def test_detect_image_success(self, test_client, sample_image, auth_headers):
        """Test successful image detection"""
        # Convert numpy array to bytes
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {
                "license_plates": [
                    {
                        "text": "ABC123",
                        "confidence": 0.95,
                        "bbox": {"x1": 200, "y1": 200, "x2": 440, "y2": 280}
                    }
                ],
                "processing_time": 1.25
            }
            
            response = test_client.post(
                "/detect/image",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "license_plates" in data
            assert len(data["license_plates"]) == 1
            assert data["license_plates"][0]["text"] == "ABC123"
    
    def test_detect_image_no_auth(self, test_client, sample_image):
        """Test image detection without authentication"""
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = test_client.post(
            "/detect/image",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 401
    
    def test_detect_image_invalid_file(self, test_client, auth_headers):
        """Test image detection with invalid file"""
        response = test_client.post(
            "/detect/image",
            files={"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
            headers=auth_headers
        )
        
        assert response.status_code == 422
    
    def test_detect_image_with_options(self, test_client, sample_image, auth_headers):
        """Test image detection with various options"""
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {"license_plates": [], "processing_time": 1.0}
            
            response = test_client.post(
                "/detect/image",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                data={
                    "confidence_threshold": "0.8",
                    "use_roboflow": "false",
                    "enhance_image": "true"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            # Verify options were passed to processor
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["confidence_threshold"] == 0.8
            assert call_kwargs["use_roboflow"] is False
    
    def test_batch_detection(self, test_client, sample_batch_images, auth_headers):
        """Test batch image detection"""
        files = []
        for img_path in sample_batch_images[:2]:  # Test with 2 images
            with open(img_path, 'rb') as f:
                files.append(("images", (f.name, f.read(), "image/jpeg")))
        
        with patch('src.api.detection.process_batch_detection') as mock_process:
            mock_process.return_value = {
                "results": [
                    {"filename": "test1.jpg", "license_plates": [], "processing_time": 1.0},
                    {"filename": "test2.jpg", "license_plates": [], "processing_time": 1.1}
                ],
                "total_processing_time": 2.1,
                "successful_detections": 2,
                "failed_detections": 0
            }
            
            response = test_client.post(
                "/detect/batch",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2
    
    def test_video_detection_start(self, test_client, auth_headers):
        """Test starting video detection job"""
        video_data = b"fake_video_data"
        
        with patch('src.api.detection.start_video_processing') as mock_start:
            mock_start.return_value = {
                "job_id": "test_job_123",
                "status": "processing",
                "estimated_completion": "2024-01-01T00:05:00Z"
            }
            
            response = test_client.post(
                "/detect/video",
                files={"video": ("test.mp4", io.BytesIO(video_data), "video/mp4")},
                headers=auth_headers
            )
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "processing"
    
    def test_video_detection_status(self, test_client, auth_headers):
        """Test getting video detection job status"""
        job_id = "test_job_123"
        
        with patch('src.api.detection.get_job_status') as mock_status:
            mock_status.return_value = {
                "job_id": job_id,
                "status": "completed",
                "progress": 100,
                "results": {
                    "total_frames": 100,
                    "frames_with_plates": 25,
                    "unique_plates": 5
                }
            }
            
            response = test_client.get(
                f"/detect/video/{job_id}/status",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["progress"] == 100

class TestAnalyticsEndpoints:
    """Test analytics and statistics endpoints"""
    
    def test_get_detection_stats(self, test_client, auth_headers):
        """Test getting detection statistics"""
        with patch('src.api.analytics.get_detection_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_detections": 150,
                "successful_detections": 140,
                "failed_detections": 10,
                "average_confidence": 0.85,
                "detections_by_hour": [10, 15, 20, 25],
                "top_regions": ["US", "EU", "ASIA"]
            }
            
            response = test_client.get("/analytics/stats", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "total_detections" in data
            assert data["total_detections"] == 150
    
    def test_get_detection_stats_with_filters(self, test_client, auth_headers):
        """Test getting detection statistics with date filters"""
        response = test_client.get(
            "/analytics/stats?start_date=2024-01-01&end_date=2024-01-31",
            headers=auth_headers
        )
        
        # Should not fail even if no data
        assert response.status_code in [200, 404]
    
    def test_get_system_metrics(self, test_client, auth_headers):
        """Test getting system performance metrics"""
        with patch('src.api.analytics.get_system_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage": 45.2,
                "memory_usage": 60.8,
                "disk_usage": 25.5,
                "active_connections": 12,
                "requests_per_minute": 45.3,
                "average_response_time": 1.25
            }
            
            response = test_client.get("/analytics/system", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "cpu_usage" in data
            assert "memory_usage" in data

class TestDataManagementEndpoints:
    """Test data management endpoints"""
    
    def test_upload_training_data(self, test_client, sample_image, auth_headers):
        """Test uploading training data"""
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        annotation_data = {
            "annotations": [
                {
                    "bbox": [200, 200, 240, 80],
                    "label": "ABC123",
                    "confidence": 1.0
                }
            ]
        }
        
        with patch('src.api.data.save_training_sample') as mock_save:
            mock_save.return_value = {"success": True, "sample_id": "123"}
            
            response = test_client.post(
                "/data/upload",
                files={
                    "image": ("train.jpg", img_bytes, "image/jpeg"),
                    "annotations": ("annotations.json", json.dumps(annotation_data), "application/json")
                },
                headers=auth_headers
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["success"] is True
    
    def test_get_dataset_info(self, test_client, auth_headers):
        """Test getting dataset information"""
        with patch('src.api.data.get_dataset_summary') as mock_summary:
            mock_summary.return_value = {
                "total_images": 1000,
                "total_annotations": 1200,
                "train_split": 800,
                "val_split": 150,
                "test_split": 50,
                "label_distribution": {
                    "US": 600,
                    "EU": 300,
                    "ASIA": 300
                }
            }
            
            response = test_client.get("/data/dataset/info", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "total_images" in data
            assert data["total_images"] == 1000

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_exceeded(self, test_client, sample_image, auth_headers):
        """Test rate limiting behavior"""
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {"license_plates": [], "processing_time": 0.1}
            
            # Make multiple requests quickly
            responses = []
            for i in range(25):  # Exceed the typical limit of 20 per minute
                img_bytes.seek(0)
                response = test_client.post(
                    "/detect/image",
                    files={"image": (f"test{i}.jpg", img_bytes, "image/jpeg")},
                    headers=auth_headers
                )
                responses.append(response)
            
            # Some responses should be rate limited (429)
            status_codes = [r.status_code for r in responses]
            assert 429 in status_codes  # At least one should be rate limited
    
    def test_rate_limit_headers(self, test_client, sample_image, auth_headers):
        """Test rate limiting headers are present"""
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {"license_plates": [], "processing_time": 0.1}
            
            response = test_client.post(
                "/detect/image",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                headers=auth_headers
            )
            
            # Check for rate limiting headers
            if response.status_code == 200:
                assert "X-RateLimit-Limit" in response.headers
                assert "X-RateLimit-Remaining" in response.headers
                assert "X-RateLimit-Reset" in response.headers

class TestWebSocketEndpoints:
    """Test WebSocket endpoints"""
    
    @pytest.mark.asyncio
    async def test_websocket_detection_stream(self, async_test_client):
        """Test WebSocket detection streaming"""
        with patch('src.api.websocket.process_websocket_detection') as mock_ws:
            async with async_test_client.websocket_connect("/ws/detect") as websocket:
                # Send test message
                await websocket.send_json({
                    "type": "image_data",
                    "data": "base64_encoded_image_data"
                })
                
                # Should receive response
                response = await websocket.receive_json()
                assert "type" in response

class TestErrorHandling:
    """Test error handling across all endpoints"""
    
    def test_404_endpoint(self, test_client):
        """Test 404 error for non-existent endpoint"""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, test_client):
        """Test 405 error for wrong HTTP method"""
        response = test_client.put("/health")
        assert response.status_code == 405
    
    def test_validation_error(self, test_client, auth_headers):
        """Test 422 validation errors"""
        response = test_client.post(
            "/detect/image",
            json={"invalid": "data"},  # Should be multipart/form-data
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_internal_server_error(self, test_client, sample_image, auth_headers):
        """Test 500 error handling"""
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            response = test_client.post(
                "/detect/image",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data

@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API workflows"""
    
    def test_full_detection_workflow(self, test_client, sample_image, auth_headers):
        """Test complete detection workflow"""
        # 1. Login (already done via auth_headers fixture)
        # 2. Upload and detect image
        img_pil = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {
                "license_plates": [{"text": "ABC123", "confidence": 0.95}],
                "processing_time": 1.0
            }
            
            response = test_client.post(
                "/detect/image",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            detection_result = response.json()
            
            # 3. Get statistics (should include this detection)
            with patch('src.api.analytics.get_detection_statistics') as mock_stats:
                mock_stats.return_value = {"total_detections": 1}
                
                stats_response = test_client.get("/analytics/stats", headers=auth_headers)
                assert stats_response.status_code == 200
    
    @pytest.mark.slow
    def test_concurrent_requests(self, test_client, sample_image, auth_headers):
        """Test handling concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            img_pil = Image.fromarray(sample_image)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            return test_client.post(
                "/detect/image",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                headers=auth_headers
            )
        
        with patch('src.api.detection.process_image_detection') as mock_process:
            mock_process.return_value = {"license_plates": [], "processing_time": 0.1}
            
            # Make 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(5)]
                responses = [future.result() for future in futures]
            
            # All should succeed or be rate limited
            for response in responses:
                assert response.status_code in [200, 429]