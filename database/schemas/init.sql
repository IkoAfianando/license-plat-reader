-- License Plate Reader Database Schema
-- PostgreSQL initialization script
-- Creates all necessary tables, indexes, and functions

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE detection_status AS ENUM ('success', 'failed', 'processing');
CREATE TYPE model_type AS ENUM ('offline', 'roboflow', 'custom');
CREATE TYPE ocr_engine AS ENUM ('paddleocr', 'easyocr', 'tesseract');
CREATE TYPE alert_severity AS ENUM ('info', 'warning', 'critical');
CREATE TYPE job_status AS ENUM ('queued', 'processing', 'completed', 'failed', 'cancelled');

-- Users table for authentication and authorization
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    api_key VARCHAR(255) UNIQUE,
    rate_limit_tier VARCHAR(20) DEFAULT 'standard'
);

-- API sessions/tokens
CREATE TABLE IF NOT EXISTS api_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_agent TEXT,
    ip_address INET
);

-- Detection models registry
CREATE TABLE IF NOT EXISTS detection_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    type model_type NOT NULL,
    version VARCHAR(20),
    file_path TEXT,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64),
    accuracy_map50 DECIMAL(5,4),
    accuracy_map95 DECIMAL(5,4),
    inference_time_ms INTEGER,
    is_active BOOLEAN DEFAULT true,
    description TEXT,
    training_data_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- Images table for uploaded/processed images
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    mime_type VARCHAR(50) NOT NULL,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    channels INTEGER DEFAULT 3,
    uploaded_by UUID REFERENCES users(id),
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Detection sessions (groups of related detections)
CREATE TABLE IF NOT EXISTS detection_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_batch BOOLEAN DEFAULT false,
    total_images INTEGER DEFAULT 0,
    completed_images INTEGER DEFAULT 0,
    status job_status DEFAULT 'queued'
);

-- Main detections table
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES detection_sessions(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    model_id UUID REFERENCES detection_models(id),
    
    -- Detection results
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    plate_score REAL CHECK (plate_score >= 0 AND plate_score <= 1),
    class_id INTEGER DEFAULT 0,
    class_name VARCHAR(50) DEFAULT 'license_plate',
    
    -- OCR results
    ocr_text VARCHAR(20),
    ocr_confidence REAL CHECK (ocr_confidence >= 0 AND ocr_confidence <= 1),
    ocr_engine ocr_engine,
    text_format_valid BOOLEAN DEFAULT false,
    detected_region VARCHAR(10),
    
    -- Processing metadata
    processing_time_ms INTEGER,
    model_version VARCHAR(20),
    status detection_status DEFAULT 'success',
    error_message TEXT,
    
    -- Timestamps
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT valid_bbox CHECK (bbox_x2 > bbox_x1 AND bbox_y2 > bbox_y1)
);

-- License plates registry (unique plates with first/last seen)
CREATE TABLE IF NOT EXISTS license_plates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_text VARCHAR(20) UNIQUE NOT NULL,
    normalized_text VARCHAR(20) NOT NULL, -- Cleaned/standardized version
    detected_region VARCHAR(10),
    format_pattern VARCHAR(50),
    confidence_avg REAL,
    detection_count INTEGER DEFAULT 1,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_flagged BOOLEAN DEFAULT false,
    flag_reason TEXT,
    notes TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Link detections to license plates
CREATE TABLE IF NOT EXISTS detection_license_plates (
    detection_id UUID NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    license_plate_id UUID NOT NULL REFERENCES license_plates(id) ON DELETE CASCADE,
    PRIMARY KEY (detection_id, license_plate_id)
);

-- Video processing jobs
CREATE TABLE IF NOT EXISTS video_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    
    -- Video metadata
    duration_seconds REAL,
    fps REAL,
    total_frames INTEGER,
    width_px INTEGER,
    height_px INTEGER,
    
    -- Processing parameters
    frame_skip INTEGER DEFAULT 1,
    max_frames INTEGER,
    confidence_threshold REAL DEFAULT 0.5,
    model_id UUID REFERENCES detection_models(id),
    
    -- Job status
    status job_status DEFAULT 'queued',
    progress_percent REAL DEFAULT 0,
    frames_processed INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    unique_plates_count INTEGER DEFAULT 0,
    
    -- Output
    output_video_path TEXT,
    results_json JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

-- System metrics for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(50) NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit VARCHAR(20),
    tags JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(100),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms REAL,
    user_id UUID REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alert history
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name VARCHAR(100) NOT NULL,
    severity alert_severity NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    
    -- Alert lifecycle
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by UUID REFERENCES users(id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Context
    trigger_data JSONB DEFAULT '{}'::jsonb,
    affected_components TEXT[],
    
    -- Notification status
    email_sent BOOLEAN DEFAULT false,
    webhook_sent BOOLEAN DEFAULT false,
    slack_sent BOOLEAN DEFAULT false
);

-- Rate limiting tracking
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(64) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    window_end TIMESTAMP WITH TIME ZONE,
    is_blocked BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite index for efficient lookups
    UNIQUE(key_hash, endpoint, window_start)
);

-- Audit log for important system events
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id UUID
);

-- Data processing pipelines
CREATE TABLE IF NOT EXISTS processing_pipelines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Dataset management
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    version VARCHAR(20) DEFAULT '1.0',
    
    -- Dataset statistics
    total_images INTEGER DEFAULT 0,
    total_annotations INTEGER DEFAULT 0,
    size_bytes BIGINT DEFAULT 0,
    
    -- Splits
    train_split REAL DEFAULT 0.8,
    val_split REAL DEFAULT 0.1,
    test_split REAL DEFAULT 0.1,
    
    -- Export information
    export_format VARCHAR(20),
    export_path TEXT,
    last_exported TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Link images to datasets
CREATE TABLE IF NOT EXISTS dataset_images (
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    split_type VARCHAR(10) CHECK (split_type IN ('train', 'val', 'test')),
    annotation_path TEXT,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (dataset_id, image_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id);
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections(detected_at);
CREATE INDEX IF NOT EXISTS idx_detections_ocr_text ON detections(ocr_text) WHERE ocr_text IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence);
CREATE INDEX IF NOT EXISTS idx_detections_session_id ON detections(session_id);

CREATE INDEX IF NOT EXISTS idx_license_plates_text ON license_plates(plate_text);
CREATE INDEX IF NOT EXISTS idx_license_plates_normalized ON license_plates(normalized_text);
CREATE INDEX IF NOT EXISTS idx_license_plates_last_seen ON license_plates(last_seen);
CREATE INDEX IF NOT EXISTS idx_license_plates_flagged ON license_plates(is_flagged) WHERE is_flagged = true;

CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_images_uploaded_by ON images(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_images_upload_timestamp ON images(upload_timestamp);

CREATE INDEX IF NOT EXISTS idx_video_jobs_user_id ON video_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_video_jobs_status ON video_jobs(status);
CREATE INDEX IF NOT EXISTS idx_video_jobs_created_at ON video_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_endpoint_time ON performance_metrics(endpoint, timestamp);

CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at ON alerts(triggered_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved_at) WHERE resolved_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_rate_limits_key_endpoint ON rate_limits(key_hash, endpoint);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_start, window_end);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_timestamp ON audit_log(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON audit_log(resource_type, resource_id);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_license_plates_text_gin ON license_plates USING gin(plate_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_detections_ocr_text_gin ON detections USING gin(ocr_text gin_trgm_ops);

-- JSON indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_detections_metadata ON detections USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_images_metadata ON images USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_system_metrics_tags ON system_metrics USING gin(tags);

-- Functions for data processing
CREATE OR REPLACE FUNCTION normalize_license_plate_text(input_text TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Remove spaces, dashes, dots and convert to uppercase
    RETURN UPPER(REGEXP_REPLACE(input_text, '[^A-Z0-9]', '', 'g'));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION update_license_plate_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update or insert license plate record
    INSERT INTO license_plates (
        plate_text, 
        normalized_text, 
        detected_region,
        confidence_avg,
        detection_count,
        first_seen,
        last_seen
    ) VALUES (
        NEW.ocr_text,
        normalize_license_plate_text(NEW.ocr_text),
        NEW.detected_region,
        NEW.ocr_confidence,
        1,
        NEW.detected_at,
        NEW.detected_at
    )
    ON CONFLICT (plate_text) DO UPDATE SET
        detection_count = license_plates.detection_count + 1,
        confidence_avg = (license_plates.confidence_avg * license_plates.detection_count + NEW.ocr_confidence) / (license_plates.detection_count + 1),
        last_seen = NEW.detected_at
    WHERE license_plates.plate_text = NEW.ocr_text;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update license plate statistics
CREATE TRIGGER update_license_plate_stats_trigger
    AFTER INSERT ON detections
    FOR EACH ROW
    WHEN (NEW.ocr_text IS NOT NULL AND NEW.status = 'success')
    EXECUTE FUNCTION update_license_plate_stats();

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detection_models_updated_at BEFORE UPDATE ON detection_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detection_sessions_updated_at BEFORE UPDATE ON detection_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE OR REPLACE VIEW detection_summary AS
SELECT 
    d.id,
    d.detected_at,
    i.filename,
    d.ocr_text,
    d.confidence,
    d.ocr_confidence,
    dm.name as model_name,
    u.username,
    lp.detection_count as plate_frequency
FROM detections d
JOIN images i ON d.image_id = i.id
LEFT JOIN detection_models dm ON d.model_id = dm.id
LEFT JOIN users u ON i.uploaded_by = u.id
LEFT JOIN license_plates lp ON d.ocr_text = lp.plate_text
WHERE d.status = 'success';

CREATE OR REPLACE VIEW daily_detection_stats AS
SELECT 
    DATE(detected_at) as date,
    COUNT(*) as total_detections,
    COUNT(DISTINCT ocr_text) as unique_plates,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time_ms,
    COUNT(*) FILTER (WHERE ocr_text IS NOT NULL) as successful_ocr
FROM detections
WHERE status = 'success'
GROUP BY DATE(detected_at)
ORDER BY date DESC;

CREATE OR REPLACE VIEW top_detected_plates AS
SELECT 
    plate_text,
    detection_count,
    first_seen,
    last_seen,
    confidence_avg,
    EXTRACT(days FROM (last_seen - first_seen)) as days_span
FROM license_plates
ORDER BY detection_count DESC, last_seen DESC
LIMIT 100;

CREATE OR REPLACE VIEW system_health AS
SELECT 
    'detections_last_24h' as metric,
    COUNT(*)::text as value,
    'count' as unit
FROM detections 
WHERE detected_at > NOW() - INTERVAL '24 hours'
UNION ALL
SELECT 
    'active_users_last_7d' as metric,
    COUNT(DISTINCT uploaded_by)::text as value,
    'count' as unit
FROM images
WHERE upload_timestamp > NOW() - INTERVAL '7 days'
UNION ALL
SELECT 
    'avg_processing_time_ms' as metric,
    ROUND(AVG(processing_time_ms), 2)::text as value,
    'milliseconds' as unit
FROM detections
WHERE detected_at > NOW() - INTERVAL '1 hour' AND processing_time_ms IS NOT NULL;

-- Insert default admin user (password: 'admin123' - CHANGE IN PRODUCTION!)
INSERT INTO users (username, email, password_hash, full_name, is_superuser)
VALUES (
    'admin',
    'admin@lpr-system.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/uOCxOiY.AoTGR1B7G', -- bcrypt hash of 'admin123'
    'System Administrator',
    true
) ON CONFLICT (username) DO NOTHING;

-- Insert default detection models
INSERT INTO detection_models (name, type, version, description, is_active)
VALUES 
    ('YOLOv8n Offline', 'offline', '8.0', 'Fast offline detection model', true),
    ('YOLOv8s Offline', 'offline', '8.0', 'Balanced offline detection model', true),
    ('Roboflow API', 'roboflow', '4.0', 'High accuracy cloud-based detection', true)
ON CONFLICT (name) DO NOTHING;

-- Insert sample dataset
INSERT INTO datasets (name, description, created_by)
SELECT 
    'Default Dataset',
    'Default dataset for license plate detection',
    id
FROM users WHERE username = 'admin'
LIMIT 1
ON CONFLICT (name) DO NOTHING;

COMMIT;