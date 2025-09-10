"""
Metrics Collection System for License Plate Reader
Collects performance metrics, system stats, and application metrics
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import influxdb_client
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DetectionMetric:
    """Detection performance metric"""
    timestamp: float
    image_size: tuple
    detections_count: int
    processing_time: float
    confidence_avg: float
    model_used: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class SystemMetric:
    """System performance metric"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_utilization: Optional[float] = None

@dataclass
class APIMetric:
    """API endpoint metric"""
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    response_time: float
    user_agent: Optional[str] = None

class MetricsCollector:
    """Central metrics collection and monitoring system"""
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 enable_prometheus: bool = True,
                 enable_influxdb: bool = False):
        """
        Initialize metrics collector
        
        Args:
            config: Configuration dictionary
            enable_prometheus: Enable Prometheus metrics
            enable_influxdb: Enable InfluxDB metrics
        """
        self.config = config or {}
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_influxdb = enable_influxdb and INFLUXDB_AVAILABLE
        
        # In-memory storage for recent metrics
        self.detection_metrics = deque(maxlen=1000)
        self.system_metrics = deque(maxlen=1000)
        self.api_metrics = deque(maxlen=1000)
        
        # Thread-safe locks
        self.detection_lock = threading.Lock()
        self.system_lock = threading.Lock()
        self.api_lock = threading.Lock()
        
        # Initialize Prometheus metrics
        if self.enable_prometheus:
            self._init_prometheus_metrics()
        
        # Initialize InfluxDB client
        if self.enable_influxdb:
            self._init_influxdb_client()
        
        # System monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        logger.info("MetricsCollector initialized")
        logger.info(f"Prometheus enabled: {self.enable_prometheus}")
        logger.info(f"InfluxDB enabled: {self.enable_influxdb}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.registry = CollectorRegistry()
        
        # Detection metrics
        self.detection_counter = Counter(
            'lpr_detections_total',
            'Total number of license plate detections',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.detection_histogram = Histogram(
            'lpr_detection_duration_seconds',
            'Time spent on license plate detection',
            ['model'],
            registry=self.registry
        )
        
        self.detection_confidence = Histogram(
            'lpr_detection_confidence',
            'Detection confidence scores',
            ['model'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'lpr_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'lpr_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'lpr_gpu_memory_used_bytes',
            'GPU memory usage in bytes',
            registry=self.registry
        )
        
        # API metrics
        self.api_requests = Counter(
            'lpr_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.api_duration = Histogram(
            'lpr_api_duration_seconds',
            'API request duration',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
    
    def _init_influxdb_client(self):
        """Initialize InfluxDB client"""
        try:
            influx_config = self.config.get('influxdb', {})
            
            self.influx_client = InfluxDBClient(
                url=influx_config.get('url', 'http://localhost:8086'),
                token=influx_config.get('token', ''),
                org=influx_config.get('org', 'lpr_org')
            )
            
            self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            self.influx_bucket = influx_config.get('bucket', 'lpr_analytics')
            
            logger.info("InfluxDB client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB: {e}")
            self.enable_influxdb = False
    
    def record_detection(self, 
                        image_size: tuple,
                        detections_count: int,
                        processing_time: float,
                        confidence_avg: float,
                        model_used: str,
                        success: bool = True,
                        error_message: Optional[str] = None):
        """Record detection performance metric"""
        metric = DetectionMetric(
            timestamp=time.time(),
            image_size=image_size,
            detections_count=detections_count,
            processing_time=processing_time,
            confidence_avg=confidence_avg,
            model_used=model_used,
            success=success,
            error_message=error_message
        )
        
        with self.detection_lock:
            self.detection_metrics.append(metric)
        
        # Update counters
        status = 'success' if success else 'error'
        self.counters[f'detections_{status}'] += 1
        self.timers['detection_times'].append(processing_time)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.detection_counter.labels(model=model_used, status=status).inc()
            self.detection_histogram.labels(model=model_used).observe(processing_time)
            if confidence_avg > 0:
                self.detection_confidence.labels(model=model_used).observe(confidence_avg)
        
        # InfluxDB metrics
        if self.enable_influxdb:
            self._write_detection_to_influxdb(metric)
    
    def record_system_metrics(self):
        """Record current system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics (if available)
            gpu_memory_used = None
            gpu_utilization = None
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_memory_used = gpu_info.used / (1024 ** 3)  # GB
                gpu_utilization = gpu_util.gpu
                
            except (ImportError, pynvml.NVMLError):
                pass  # GPU monitoring not available
            
            metric = SystemMetric(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024 ** 3),
                disk_usage_percent=(disk.used / disk.total) * 100,
                gpu_memory_used=gpu_memory_used,
                gpu_utilization=gpu_utilization
            )
            
            with self.system_lock:
                self.system_metrics.append(metric)
            
            # Prometheus metrics
            if self.enable_prometheus:
                self.cpu_usage.set(cpu_percent)
                self.memory_usage.set(memory.percent)
                if gpu_memory_used:
                    self.gpu_memory.set(gpu_memory_used * (1024 ** 3))  # Convert back to bytes
            
            # InfluxDB metrics
            if self.enable_influxdb:
                self._write_system_to_influxdb(metric)
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def record_api_request(self,
                          endpoint: str,
                          method: str,
                          status_code: int,
                          response_time: float,
                          user_agent: Optional[str] = None):
        """Record API request metric"""
        metric = APIMetric(
            timestamp=time.time(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            user_agent=user_agent
        )
        
        with self.api_lock:
            self.api_metrics.append(metric)
        
        # Update counters
        self.counters[f'api_requests_{status_code}'] += 1
        self.timers['api_response_times'].append(response_time)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.api_requests.labels(
                endpoint=endpoint, 
                method=method, 
                status=str(status_code)
            ).inc()
            self.api_duration.labels(endpoint=endpoint, method=method).observe(response_time)
        
        # InfluxDB metrics
        if self.enable_influxdb:
            self._write_api_to_influxdb(metric)
    
    def _write_detection_to_influxdb(self, metric: DetectionMetric):
        """Write detection metric to InfluxDB"""
        try:
            point = Point("detection") \
                .tag("model", metric.model_used) \
                .tag("success", str(metric.success)) \
                .field("detections_count", metric.detections_count) \
                .field("processing_time", metric.processing_time) \
                .field("confidence_avg", metric.confidence_avg) \
                .field("image_width", metric.image_size[0]) \
                .field("image_height", metric.image_size[1]) \
                .time(int(metric.timestamp * 1000), WritePrecision.MS)
            
            if metric.error_message:
                point = point.tag("error", metric.error_message)
            
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
            
        except Exception as e:
            logger.error(f"Failed to write detection metric to InfluxDB: {e}")
    
    def _write_system_to_influxdb(self, metric: SystemMetric):
        """Write system metric to InfluxDB"""
        try:
            point = Point("system") \
                .field("cpu_percent", metric.cpu_percent) \
                .field("memory_percent", metric.memory_percent) \
                .field("memory_used_gb", metric.memory_used_gb) \
                .field("disk_usage_percent", metric.disk_usage_percent) \
                .time(int(metric.timestamp * 1000), WritePrecision.MS)
            
            if metric.gpu_memory_used:
                point = point.field("gpu_memory_used", metric.gpu_memory_used)
            if metric.gpu_utilization:
                point = point.field("gpu_utilization", metric.gpu_utilization)
            
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
            
        except Exception as e:
            logger.error(f"Failed to write system metric to InfluxDB: {e}")
    
    def _write_api_to_influxdb(self, metric: APIMetric):
        """Write API metric to InfluxDB"""
        try:
            point = Point("api") \
                .tag("endpoint", metric.endpoint) \
                .tag("method", metric.method) \
                .tag("status_code", str(metric.status_code)) \
                .field("response_time", metric.response_time) \
                .time(int(metric.timestamp * 1000), WritePrecision.MS)
            
            if metric.user_agent:
                point = point.tag("user_agent", metric.user_agent[:100])  # Limit length
            
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
            
        except Exception as e:
            logger.error(f"Failed to write API metric to InfluxDB: {e}")
    
    def start_system_monitoring(self, interval: int = 30):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            logger.warning("System monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                self.record_system_metrics()
                time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"System monitoring started (interval: {interval}s)")
    
    def stop_system_monitoring(self):
        """Stop system monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def get_metrics_summary(self, last_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of metrics from last N minutes"""
        cutoff_time = time.time() - (last_minutes * 60)
        
        summary = {
            'period_minutes': last_minutes,
            'generated_at': datetime.now().isoformat(),
            'detection': {},
            'system': {},
            'api': {},
            'counters': dict(self.counters)
        }
        
        # Detection metrics summary
        with self.detection_lock:
            recent_detections = [m for m in self.detection_metrics if m.timestamp > cutoff_time]
        
        if recent_detections:
            successful = [m for m in recent_detections if m.success]
            processing_times = [m.processing_time for m in successful]
            confidences = [m.confidence_avg for m in successful if m.confidence_avg > 0]
            detection_counts = [m.detections_count for m in successful]
            
            summary['detection'] = {
                'total_requests': len(recent_detections),
                'successful_requests': len(successful),
                'success_rate': len(successful) / len(recent_detections) if recent_detections else 0,
                'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'total_detections': sum(detection_counts),
                'avg_detections_per_image': sum(detection_counts) / len(detection_counts) if detection_counts else 0
            }
        
        # System metrics summary
        with self.system_lock:
            recent_system = [m for m in self.system_metrics if m.timestamp > cutoff_time]
        
        if recent_system:
            cpu_values = [m.cpu_percent for m in recent_system]
            memory_values = [m.memory_percent for m in recent_system]
            
            summary['system'] = {
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                'max_cpu_percent': max(cpu_values),
                'avg_memory_percent': sum(memory_values) / len(memory_values),
                'max_memory_percent': max(memory_values),
                'samples': len(recent_system)
            }
        
        # API metrics summary
        with self.api_lock:
            recent_api = [m for m in self.api_metrics if m.timestamp > cutoff_time]
        
        if recent_api:
            response_times = [m.response_time for m in recent_api]
            status_codes = defaultdict(int)
            for m in recent_api:
                status_codes[str(m.status_code)] += 1
            
            summary['api'] = {
                'total_requests': len(recent_api),
                'avg_response_time': sum(response_times) / len(response_times),
                'max_response_time': max(response_times),
                'status_codes': dict(status_codes)
            }
        
        return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.enable_prometheus:
            return "# Prometheus not enabled"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def export_metrics_json(self, last_minutes: int = 60) -> str:
        """Export metrics summary as JSON"""
        summary = self.get_metrics_summary(last_minutes)
        return json.dumps(summary, indent=2)
    
    def save_metrics_report(self, 
                           output_path: str,
                           last_minutes: int = 60,
                           format: str = 'json'):
        """Save metrics report to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'json':
            content = self.export_metrics_json(last_minutes)
        elif format.lower() == 'prometheus':
            content = self.export_prometheus_metrics()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Metrics report saved: {output_path}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        # Get recent system metrics
        current_time = time.time()
        recent_system = None
        
        with self.system_lock:
            if self.system_metrics:
                recent_system = self.system_metrics[-1]
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # System resource checks
        if recent_system:
            health_status['checks']['cpu'] = {
                'status': 'ok' if recent_system.cpu_percent < 80 else 'warning',
                'value': recent_system.cpu_percent,
                'threshold': 80
            }
            
            health_status['checks']['memory'] = {
                'status': 'ok' if recent_system.memory_percent < 85 else 'warning',
                'value': recent_system.memory_percent,
                'threshold': 85
            }
            
            health_status['checks']['disk'] = {
                'status': 'ok' if recent_system.disk_usage_percent < 90 else 'warning',
                'value': recent_system.disk_usage_percent,
                'threshold': 90
            }
        
        # Detection performance checks
        summary = self.get_metrics_summary(5)  # Last 5 minutes
        if 'detection' in summary and summary['detection']:
            success_rate = summary['detection']['success_rate']
            health_status['checks']['detection_success_rate'] = {
                'status': 'ok' if success_rate > 0.95 else 'warning',
                'value': success_rate,
                'threshold': 0.95
            }
        
        # Overall health status
        warning_checks = [check for check in health_status['checks'].values() 
                         if check['status'] == 'warning']
        if warning_checks:
            health_status['status'] = 'warning'
        
        return health_status
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_system_monitoring()
        
        if self.enable_influxdb and hasattr(self, 'influx_client'):
            try:
                self.influx_client.close()
            except:
                pass
        
        logger.info("MetricsCollector cleanup completed")


def main():
    """Demo script for metrics collector"""
    print("📊 License Plate Reader Metrics Collector Demo")
    print("=" * 50)
    
    # Initialize metrics collector
    collector = MetricsCollector(
        enable_prometheus=True,
        enable_influxdb=False  # Disable for demo
    )
    
    # Start system monitoring
    collector.start_system_monitoring(interval=5)
    
    try:
        # Simulate some detection metrics
        print("🎭 Simulating detection metrics...")
        
        import random
        for i in range(10):
            collector.record_detection(
                image_size=(640, 480),
                detections_count=random.randint(0, 3),
                processing_time=random.uniform(0.1, 0.5),
                confidence_avg=random.uniform(0.7, 0.95),
                model_used=random.choice(['yolov8n', 'yolov8s', 'roboflow']),
                success=random.random() > 0.1  # 90% success rate
            )
            
            # Simulate API requests
            collector.record_api_request(
                endpoint='/detect/image',
                method='POST',
                status_code=random.choice([200, 200, 200, 400, 500]),
                response_time=random.uniform(0.2, 1.0)
            )
            
            time.sleep(1)
        
        print("✅ Simulated metrics recorded")
        
        # Wait for some system metrics
        print("⏱️  Collecting system metrics (10 seconds)...")
        time.sleep(10)
        
        # Get metrics summary
        summary = collector.get_metrics_summary(last_minutes=5)
        print("\\n📈 Metrics Summary:")
        print(f"  Detection requests: {summary.get('detection', {}).get('total_requests', 0)}")
        print(f"  Success rate: {summary.get('detection', {}).get('success_rate', 0):.2%}")
        print(f"  Avg processing time: {summary.get('detection', {}).get('avg_processing_time', 0):.3f}s")
        print(f"  System CPU: {summary.get('system', {}).get('avg_cpu_percent', 0):.1f}%")
        print(f"  System Memory: {summary.get('system', {}).get('avg_memory_percent', 0):.1f}%")
        
        # Get health status
        health = collector.get_health_status()
        print(f"\\n🏥 Health Status: {health['status']}")
        for check_name, check_info in health['checks'].items():
            status_icon = "✅" if check_info['status'] == 'ok' else "⚠️"
            print(f"  {status_icon} {check_name}: {check_info['value']:.1f}")
        
        # Export metrics
        print("\\n💾 Exporting metrics...")
        
        # Save JSON report
        os.makedirs("outputs/reports", exist_ok=True)
        collector.save_metrics_report(
            "outputs/reports/metrics_demo.json",
            last_minutes=5,
            format='json'
        )
        
        # Save Prometheus metrics
        collector.save_metrics_report(
            "outputs/reports/metrics_demo.prom",
            format='prometheus'
        )
        
        print("✅ Metrics exported to outputs/reports/")
        
        # Display Prometheus metrics sample
        if collector.enable_prometheus:
            prom_metrics = collector.export_prometheus_metrics()
            print("\\n📊 Prometheus Metrics Sample:")
            print(prom_metrics[:500] + "..." if len(prom_metrics) > 500 else prom_metrics)
        
    except KeyboardInterrupt:
        print("\\n⏹️  Demo interrupted")
    finally:
        collector.cleanup()
        print("✅ Metrics collector demo completed!")


if __name__ == "__main__":
    main()