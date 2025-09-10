"""
Database Tests
Test database models, schemas, and operations
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

from database.models import (
    Base, User, Detection, LicensePlate, ProcessingJob,
    TrainingData, ModelPerformance, SystemMetrics
)

class TestDatabaseModels:
    """Test database model functionality"""
    
    def test_user_model(self, test_db_session):
        """Test User model operations"""
        # Create user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123",
            role="user"
        )
        
        test_db_session.add(user)
        test_db_session.commit()
        
        # Verify creation
        retrieved_user = test_db_session.query(User).filter_by(username="testuser").first()
        assert retrieved_user is not None
        assert retrieved_user.email == "test@example.com"
        assert retrieved_user.role == "user"
        assert retrieved_user.is_active is True
        assert retrieved_user.created_at is not None
    
    def test_user_unique_constraints(self, test_db_session):
        """Test user unique constraints"""
        # Create first user
        user1 = User(username="testuser", email="test@example.com", hashed_password="hash1")
        test_db_session.add(user1)
        test_db_session.commit()
        
        # Try to create user with same username
        user2 = User(username="testuser", email="different@example.com", hashed_password="hash2")
        test_db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
        
        test_db_session.rollback()
        
        # Try to create user with same email
        user3 = User(username="different", email="test@example.com", hashed_password="hash3")
        test_db_session.add(user3)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_detection_model(self, test_db_session):
        """Test Detection model operations"""
        # Create user first
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Create detection
        detection = Detection(
            user_id=user.id,
            image_path="/path/to/image.jpg",
            image_hash="abc123hash",
            processing_time=1.25,
            confidence_threshold=0.5,
            model_used="yolov8n",
            detection_count=2
        )
        
        test_db_session.add(detection)
        test_db_session.commit()
        
        # Verify creation
        retrieved = test_db_session.query(Detection).first()
        assert retrieved is not None
        assert retrieved.user_id == user.id
        assert retrieved.image_path == "/path/to/image.jpg"
        assert retrieved.processing_time == 1.25
        assert retrieved.detection_count == 2
        assert retrieved.created_at is not None
    
    def test_license_plate_model(self, test_db_session):
        """Test LicensePlate model operations"""
        # Create dependencies
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        detection = Detection(
            user_id=user.id,
            image_path="/path/to/image.jpg",
            processing_time=1.0,
            model_used="yolov8n"
        )
        test_db_session.add(detection)
        test_db_session.commit()
        
        # Create license plate
        plate = LicensePlate(
            detection_id=detection.id,
            text="ABC123",
            confidence=0.95,
            bbox_x1=100,
            bbox_y1=200,
            bbox_x2=300,
            bbox_y2=250,
            region="US",
            state="CA"
        )
        
        test_db_session.add(plate)
        test_db_session.commit()
        
        # Verify creation and relationships
        retrieved_plate = test_db_session.query(LicensePlate).first()
        assert retrieved_plate is not None
        assert retrieved_plate.text == "ABC123"
        assert retrieved_plate.confidence == 0.95
        assert retrieved_plate.region == "US"
        
        # Test relationship
        assert retrieved_plate.detection is not None
        assert retrieved_plate.detection.user.username == "testuser"
    
    def test_processing_job_model(self, test_db_session):
        """Test ProcessingJob model operations"""
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Create processing job
        job = ProcessingJob(
            user_id=user.id,
            job_type="video_processing",
            status="processing",
            progress=50.0,
            input_path="/path/to/video.mp4",
            estimated_completion=datetime.utcnow() + timedelta(minutes=5)
        )
        
        test_db_session.add(job)
        test_db_session.commit()
        
        # Verify creation
        retrieved_job = test_db_session.query(ProcessingJob).first()
        assert retrieved_job is not None
        assert retrieved_job.job_type == "video_processing"
        assert retrieved_job.status == "processing"
        assert retrieved_job.progress == 50.0
        assert retrieved_job.estimated_completion is not None
    
    def test_training_data_model(self, test_db_session):
        """Test TrainingData model operations"""
        training_data = TrainingData(
            image_path="/training/images/001.jpg",
            annotation_path="/training/annotations/001.txt",
            dataset_split="train",
            region="US",
            verified=True
        )
        
        test_db_session.add(training_data)
        test_db_session.commit()
        
        # Verify creation
        retrieved = test_db_session.query(TrainingData).first()
        assert retrieved is not None
        assert retrieved.dataset_split == "train"
        assert retrieved.verified is True
        assert retrieved.region == "US"
    
    def test_model_performance_tracking(self, test_db_session):
        """Test ModelPerformance model operations"""
        performance = ModelPerformance(
            model_name="yolov8n",
            model_version="1.0",
            accuracy=0.92,
            precision=0.89,
            recall=0.94,
            f1_score=0.91,
            inference_time=1.25,
            dataset_size=1000,
            test_date=datetime.utcnow()
        )
        
        test_db_session.add(performance)
        test_db_session.commit()
        
        # Verify creation
        retrieved = test_db_session.query(ModelPerformance).first()
        assert retrieved is not None
        assert retrieved.model_name == "yolov8n"
        assert retrieved.accuracy == 0.92
        assert retrieved.f1_score == 0.91
    
    def test_system_metrics_model(self, test_db_session):
        """Test SystemMetrics model operations"""
        metrics = SystemMetrics(
            cpu_usage=45.2,
            memory_usage=1024.5,
            disk_usage=75.8,
            active_connections=12,
            requests_per_minute=45.3,
            average_response_time=1.25,
            error_rate=0.02
        )
        
        test_db_session.add(metrics)
        test_db_session.commit()
        
        # Verify creation
        retrieved = test_db_session.query(SystemMetrics).first()
        assert retrieved is not None
        assert retrieved.cpu_usage == 45.2
        assert retrieved.memory_usage == 1024.5
        assert retrieved.error_rate == 0.02

class TestDatabaseRelationships:
    """Test relationships between models"""
    
    def test_user_detection_relationship(self, test_db_session):
        """Test User-Detection relationship"""
        # Create user
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Create multiple detections
        detection1 = Detection(user_id=user.id, image_path="/img1.jpg", processing_time=1.0)
        detection2 = Detection(user_id=user.id, image_path="/img2.jpg", processing_time=1.5)
        
        test_db_session.add_all([detection1, detection2])
        test_db_session.commit()
        
        # Test relationship
        user_detections = user.detections
        assert len(user_detections) == 2
        assert all(d.user_id == user.id for d in user_detections)
    
    def test_detection_license_plate_relationship(self, test_db_session):
        """Test Detection-LicensePlate relationship"""
        # Create dependencies
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        detection = Detection(user_id=user.id, image_path="/img.jpg", processing_time=1.0)
        test_db_session.add(detection)
        test_db_session.commit()
        
        # Create multiple license plates for one detection
        plate1 = LicensePlate(
            detection_id=detection.id, text="ABC123", confidence=0.95,
            bbox_x1=100, bbox_y1=100, bbox_x2=200, bbox_y2=150
        )
        plate2 = LicensePlate(
            detection_id=detection.id, text="XYZ789", confidence=0.88,
            bbox_x1=300, bbox_y1=200, bbox_x2=400, bbox_y2=250
        )
        
        test_db_session.add_all([plate1, plate2])
        test_db_session.commit()
        
        # Test relationship
        detection_plates = detection.license_plates
        assert len(detection_plates) == 2
        assert all(p.detection_id == detection.id for p in detection_plates)
        
        # Test reverse relationship
        assert plate1.detection.id == detection.id
    
    def test_cascade_deletion(self, test_db_session):
        """Test cascade deletion behavior"""
        # Create user with detection and plates
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        detection = Detection(user_id=user.id, image_path="/img.jpg", processing_time=1.0)
        test_db_session.add(detection)
        test_db_session.commit()
        
        plate = LicensePlate(
            detection_id=detection.id, text="ABC123", confidence=0.95,
            bbox_x1=100, bbox_y1=100, bbox_x2=200, bbox_y2=150
        )
        test_db_session.add(plate)
        test_db_session.commit()
        
        # Delete detection should cascade to license plates
        test_db_session.delete(detection)
        test_db_session.commit()
        
        # License plate should be deleted
        remaining_plates = test_db_session.query(LicensePlate).count()
        assert remaining_plates == 0

class TestDatabaseQueries:
    """Test complex database queries"""
    
    def test_detection_statistics(self, test_db_session):
        """Test detection statistics queries"""
        # Create test data
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Create detections with different dates
        today = datetime.utcnow()
        yesterday = today - timedelta(days=1)
        
        detection1 = Detection(
            user_id=user.id, image_path="/img1.jpg", processing_time=1.0,
            detection_count=2, created_at=today
        )
        detection2 = Detection(
            user_id=user.id, image_path="/img2.jpg", processing_time=1.5,
            detection_count=1, created_at=yesterday
        )
        
        test_db_session.add_all([detection1, detection2])
        test_db_session.commit()
        
        # Test queries
        total_detections = test_db_session.query(Detection).count()
        assert total_detections == 2
        
        today_detections = test_db_session.query(Detection).filter(
            Detection.created_at >= today.date()
        ).count()
        assert today_detections == 1
        
        total_plates = test_db_session.query(Detection).with_entities(
            text("SUM(detection_count)")
        ).scalar()
        assert total_plates == 3
    
    def test_user_activity_queries(self, test_db_session):
        """Test user activity tracking queries"""
        # Create users with different activity levels
        user1 = User(username="user1", email="user1@test.com", hashed_password="hash")
        user2 = User(username="user2", email="user2@test.com", hashed_password="hash")
        test_db_session.add_all([user1, user2])
        test_db_session.commit()
        
        # User1 has more detections
        for i in range(5):
            detection = Detection(
                user_id=user1.id, image_path=f"/img{i}.jpg", processing_time=1.0
            )
            test_db_session.add(detection)
        
        # User2 has fewer detections
        detection = Detection(user_id=user2.id, image_path="/img.jpg", processing_time=1.0)
        test_db_session.add(detection)
        test_db_session.commit()
        
        # Query most active users
        from sqlalchemy import func
        most_active = test_db_session.query(
            User.username,
            func.count(Detection.id).label('detection_count')
        ).join(Detection).group_by(User.id).order_by(
            func.count(Detection.id).desc()
        ).all()
        
        assert len(most_active) == 2
        assert most_active[0].username == "user1"
        assert most_active[0].detection_count == 5
    
    def test_performance_trending(self, test_db_session):
        """Test performance trending queries"""
        # Create performance records over time
        base_date = datetime.utcnow() - timedelta(days=10)
        
        for i in range(5):
            performance = ModelPerformance(
                model_name="yolov8n",
                model_version="1.0",
                accuracy=0.9 + (i * 0.01),  # Improving accuracy
                test_date=base_date + timedelta(days=i * 2)
            )
            test_db_session.add(performance)
        
        test_db_session.commit()
        
        # Query performance trend
        from sqlalchemy import func
        performance_trend = test_db_session.query(
            func.avg(ModelPerformance.accuracy).label('avg_accuracy'),
            func.max(ModelPerformance.accuracy).label('max_accuracy'),
            func.min(ModelPerformance.accuracy).label('min_accuracy')
        ).filter(ModelPerformance.model_name == "yolov8n").first()
        
        assert performance_trend.max_accuracy > performance_trend.min_accuracy
        assert 0.9 <= performance_trend.avg_accuracy <= 0.95
    
    def test_regional_distribution(self, test_db_session):
        """Test regional distribution queries"""
        # Create test data with different regions
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        detection = Detection(user_id=user.id, image_path="/img.jpg", processing_time=1.0)
        test_db_session.add(detection)
        test_db_session.commit()
        
        # Create plates from different regions
        regions = ["US", "US", "EU", "ASIA", "EU"]
        for i, region in enumerate(regions):
            plate = LicensePlate(
                detection_id=detection.id,
                text=f"ABC{i}",
                confidence=0.9,
                bbox_x1=100, bbox_y1=100, bbox_x2=200, bbox_y2=150,
                region=region
            )
            test_db_session.add(plate)
        
        test_db_session.commit()
        
        # Query regional distribution
        from sqlalchemy import func
        regional_dist = test_db_session.query(
            LicensePlate.region,
            func.count(LicensePlate.id).label('count')
        ).group_by(LicensePlate.region).order_by(
            func.count(LicensePlate.id).desc()
        ).all()
        
        assert len(regional_dist) == 3
        assert regional_dist[0].region in ["US", "EU"]  # Top regions
        assert regional_dist[0].count == 2

class TestDatabaseConstraints:
    """Test database constraints and validation"""
    
    def test_foreign_key_constraints(self, test_db_session):
        """Test foreign key constraints"""
        # Try to create detection without valid user
        detection = Detection(
            user_id=999,  # Non-existent user
            image_path="/img.jpg",
            processing_time=1.0
        )
        test_db_session.add(detection)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
        
        test_db_session.rollback()
        
        # Try to create license plate without valid detection
        plate = LicensePlate(
            detection_id=999,  # Non-existent detection
            text="ABC123",
            confidence=0.9,
            bbox_x1=100, bbox_y1=100, bbox_x2=200, bbox_y2=150
        )
        test_db_session.add(plate)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_check_constraints(self, test_db_session):
        """Test check constraints (if implemented)"""
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Test invalid confidence (should be between 0 and 1)
        detection = Detection(user_id=user.id, image_path="/img.jpg", processing_time=1.0)
        test_db_session.add(detection)
        test_db_session.commit()
        
        invalid_plate = LicensePlate(
            detection_id=detection.id,
            text="ABC123",
            confidence=1.5,  # Invalid confidence > 1
            bbox_x1=100, bbox_y1=100, bbox_x2=200, bbox_y2=150
        )
        test_db_session.add(invalid_plate)
        
        # Depending on implementation, this might raise an error
        try:
            test_db_session.commit()
        except IntegrityError:
            test_db_session.rollback()
            # Expected behavior if check constraints are implemented
    
    def test_nullable_constraints(self, test_db_session):
        """Test NOT NULL constraints"""
        # Try to create user without required fields
        incomplete_user = User(username="testuser")  # Missing email and password
        test_db_session.add(incomplete_user)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()

class TestDatabasePerformance:
    """Test database performance and indexing"""
    
    def test_index_usage(self, test_db_session):
        """Test that indexes are used effectively"""
        # Create test data
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # Create many detections
        for i in range(100):
            detection = Detection(
                user_id=user.id,
                image_path=f"/img{i}.jpg",
                processing_time=1.0 + (i * 0.01)
            )
            test_db_session.add(detection)
        
        test_db_session.commit()
        
        # Query with indexed field should be efficient
        import time
        start_time = time.time()
        
        user_detections = test_db_session.query(Detection).filter(
            Detection.user_id == user.id
        ).all()
        
        query_time = time.time() - start_time
        
        assert len(user_detections) == 100
        assert query_time < 1.0  # Should be fast with proper indexing
    
    def test_bulk_operations(self, test_db_session):
        """Test bulk insert/update operations"""
        # Test bulk insert
        users = [
            User(username=f"user{i}", email=f"user{i}@test.com", hashed_password="hash")
            for i in range(50)
        ]
        
        import time
        start_time = time.time()
        
        test_db_session.bulk_save_objects(users)
        test_db_session.commit()
        
        bulk_time = time.time() - start_time
        
        # Verify all users were created
        user_count = test_db_session.query(User).count()
        assert user_count == 50
        
        # Bulk operations should be reasonably fast
        assert bulk_time < 2.0

@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def test_full_detection_workflow(self, test_db_session):
        """Test complete detection workflow with database"""
        # 1. Create user
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        # 2. Process image and store detection
        detection = Detection(
            user_id=user.id,
            image_path="/test/image.jpg",
            image_hash="abc123",
            processing_time=1.25,
            confidence_threshold=0.5,
            model_used="yolov8n",
            detection_count=2
        )
        test_db_session.add(detection)
        test_db_session.commit()
        
        # 3. Store license plate results
        plates = [
            LicensePlate(
                detection_id=detection.id,
                text="ABC123",
                confidence=0.95,
                bbox_x1=100, bbox_y1=200, bbox_x2=300, bbox_y2=250,
                region="US", state="CA"
            ),
            LicensePlate(
                detection_id=detection.id,
                text="XYZ789",
                confidence=0.88,
                bbox_x1=400, bbox_y1=150, bbox_x2=600, bbox_y2=200,
                region="US", state="NY"
            )
        ]
        test_db_session.add_all(plates)
        test_db_session.commit()
        
        # 4. Verify complete workflow
        stored_detection = test_db_session.query(Detection).filter_by(
            image_path="/test/image.jpg"
        ).first()
        
        assert stored_detection is not None
        assert stored_detection.detection_count == 2
        assert len(stored_detection.license_plates) == 2
        assert stored_detection.user.username == "testuser"
        
        # Verify individual plates
        plate_texts = [p.text for p in stored_detection.license_plates]
        assert "ABC123" in plate_texts
        assert "XYZ789" in plate_texts
    
    def test_concurrent_access(self, test_db_session):
        """Test concurrent database access"""
        import threading
        import time
        
        # Create shared user
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        test_db_session.add(user)
        test_db_session.commit()
        
        results = []
        errors = []
        
        def create_detection(thread_id):
            try:
                # Each thread creates its own session
                from sqlalchemy.orm import sessionmaker
                Session = sessionmaker(bind=test_db_session.bind)
                session = Session()
                
                detection = Detection(
                    user_id=user.id,
                    image_path=f"/img_{thread_id}.jpg",
                    processing_time=1.0
                )
                session.add(detection)
                session.commit()
                session.close()
                
                results.append(thread_id)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_detection, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 5
        assert len(errors) == 0
        
        # Verify all detections were created
        total_detections = test_db_session.query(Detection).count()
        assert total_detections == 5