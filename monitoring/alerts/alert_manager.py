"""
Alert Manager for License Plate Reader System
Monitors metrics and sends alerts when thresholds are exceeded
"""

import smtplib
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    severity: AlertSeverity
    condition_func: str  # Function name to evaluate
    threshold: float
    duration_minutes: int = 5  # Alert if condition persists for this long
    cooldown_minutes: int = 30  # Don't re-alert for this period
    enabled: bool = True

@dataclass
class Alert:
    """Active alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[float] = None
    metadata: Optional[Dict] = None

class AlertManager:
    """Manages alerts and notifications for the LPR system"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize alert manager
        
        Args:
            config: Configuration dictionary with alert settings
        """
        self.config = config or {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Notification channels
        self.email_config = self.config.get('email', {})
        self.webhook_config = self.config.get('webhooks', {})
        self.slack_config = self.config.get('slack', {})
        
        # State tracking
        self.condition_states = {}  # Track how long conditions have been true
        self.last_alerts = {}  # Track last alert time for cooldown
        
        # Threading
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Load default rules
        self._load_default_rules()
        
        logger.info("AlertManager initialized")
    
    def _load_default_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                description="CPU usage is high",
                severity=AlertSeverity.WARNING,
                condition_func="check_cpu_usage",
                threshold=80.0,
                duration_minutes=5,
                cooldown_minutes=30
            ),
            AlertRule(
                name="high_memory_usage", 
                description="Memory usage is high",
                severity=AlertSeverity.WARNING,
                condition_func="check_memory_usage",
                threshold=85.0,
                duration_minutes=5,
                cooldown_minutes=30
            ),
            AlertRule(
                name="low_detection_success_rate",
                description="Detection success rate is low",
                severity=AlertSeverity.CRITICAL,
                condition_func="check_detection_success_rate",
                threshold=0.8,
                duration_minutes=10,
                cooldown_minutes=60
            ),
            AlertRule(
                name="high_api_error_rate",
                description="API error rate is high",
                severity=AlertSeverity.WARNING,
                condition_func="check_api_error_rate",
                threshold=0.1,
                duration_minutes=5,
                cooldown_minutes=30
            ),
            AlertRule(
                name="slow_detection_processing",
                description="Detection processing is slow",
                severity=AlertSeverity.WARNING,
                condition_func="check_detection_processing_time",
                threshold=2.0,
                duration_minutes=10,
                cooldown_minutes=30
            ),
            AlertRule(
                name="disk_usage_high",
                description="Disk usage is high",
                severity=AlertSeverity.CRITICAL,
                condition_func="check_disk_usage",
                threshold=90.0,
                duration_minutes=15,
                cooldown_minutes=60
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
        
        logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        with self.lock:
            self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                # Also remove any tracking state
                if rule_name in self.condition_states:
                    del self.condition_states[rule_name]
                if rule_name in self.last_alerts:
                    del self.last_alerts[rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def evaluate_conditions(self, metrics_data: Dict[str, Any]):
        """
        Evaluate alert conditions against metrics data
        
        Args:
            metrics_data: Current metrics data from MetricsCollector
        """
        current_time = time.time()
        
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check if we're in cooldown period
                if rule_name in self.last_alerts:
                    time_since_last = current_time - self.last_alerts[rule_name]
                    if time_since_last < (rule.cooldown_minutes * 60):
                        continue
                
                # Evaluate condition
                try:
                    condition_met = self._evaluate_condition(rule, metrics_data)
                    
                    if condition_met:
                        # Track how long condition has been true
                        if rule_name not in self.condition_states:
                            self.condition_states[rule_name] = current_time
                        
                        # Check if duration threshold is met
                        duration = current_time - self.condition_states[rule_name]
                        if duration >= (rule.duration_minutes * 60):
                            self._trigger_alert(rule, metrics_data, current_time)
                            self.last_alerts[rule_name] = current_time
                    else:
                        # Condition not met, reset state
                        if rule_name in self.condition_states:
                            del self.condition_states[rule_name]
                        
                        # Check if we should resolve existing alert
                        if rule_name in self.alerts:
                            self._resolve_alert(rule_name, current_time)
                
                except Exception as e:
                    logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, rule: AlertRule, metrics_data: Dict[str, Any]) -> bool:
        """Evaluate a specific alert condition"""
        condition_func = rule.condition_func
        threshold = rule.threshold
        
        if condition_func == "check_cpu_usage":
            system_data = metrics_data.get('system', {})
            current_cpu = system_data.get('avg_cpu_percent', 0)
            return current_cpu > threshold
        
        elif condition_func == "check_memory_usage":
            system_data = metrics_data.get('system', {})
            current_memory = system_data.get('avg_memory_percent', 0)
            return current_memory > threshold
        
        elif condition_func == "check_detection_success_rate":
            detection_data = metrics_data.get('detection', {})
            success_rate = detection_data.get('success_rate', 1.0)
            return success_rate < threshold
        
        elif condition_func == "check_api_error_rate":
            api_data = metrics_data.get('api', {})
            total_requests = api_data.get('total_requests', 0)
            if total_requests == 0:
                return False
            
            status_codes = api_data.get('status_codes', {})
            error_requests = sum(count for code, count in status_codes.items() 
                               if int(code) >= 400)
            error_rate = error_requests / total_requests
            return error_rate > threshold
        
        elif condition_func == "check_detection_processing_time":
            detection_data = metrics_data.get('detection', {})
            avg_processing_time = detection_data.get('avg_processing_time', 0)
            return avg_processing_time > threshold
        
        elif condition_func == "check_disk_usage":
            # This would need to be provided by system metrics
            system_data = metrics_data.get('system', {})
            # Assuming disk usage is added to system metrics
            disk_usage = system_data.get('disk_usage_percent', 0)
            return disk_usage > threshold
        
        else:
            logger.warning(f"Unknown condition function: {condition_func}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, metrics_data: Dict[str, Any], timestamp: float):
        """Trigger an alert"""
        # Generate alert message
        message = self._generate_alert_message(rule, metrics_data)
        
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            timestamp=timestamp,
            metadata=metrics_data
        )
        
        # Store alert
        self.alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {rule.name} - {message}")
    
    def _resolve_alert(self, rule_name: str, timestamp: float):
        """Resolve an active alert"""
        if rule_name in self.alerts:
            alert = self.alerts[rule_name]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = timestamp
            
            # Remove from active alerts
            del self.alerts[rule_name]
            
            # Send resolution notification
            self._send_resolution_notification(alert)
            
            logger.info(f"Alert resolved: {rule_name}")
    
    def _generate_alert_message(self, rule: AlertRule, metrics_data: Dict[str, Any]) -> str:
        """Generate alert message"""
        message = f"ALERT: {rule.description}\\n\\n"
        
        # Add specific details based on rule type
        if rule.condition_func == "check_cpu_usage":
            cpu_usage = metrics_data.get('system', {}).get('avg_cpu_percent', 0)
            message += f"Current CPU usage: {cpu_usage:.1f}% (threshold: {rule.threshold}%)\\n"
        
        elif rule.condition_func == "check_memory_usage":
            memory_usage = metrics_data.get('system', {}).get('avg_memory_percent', 0)
            message += f"Current memory usage: {memory_usage:.1f}% (threshold: {rule.threshold}%)\\n"
        
        elif rule.condition_func == "check_detection_success_rate":
            success_rate = metrics_data.get('detection', {}).get('success_rate', 1.0)
            message += f"Current success rate: {success_rate:.2%} (threshold: {rule.threshold:.2%})\\n"
        
        elif rule.condition_func == "check_api_error_rate":
            api_data = metrics_data.get('api', {})
            message += f"API error rate exceeded threshold of {rule.threshold:.2%}\\n"
            message += f"Total requests: {api_data.get('total_requests', 0)}\\n"
        
        elif rule.condition_func == "check_detection_processing_time":
            avg_time = metrics_data.get('detection', {}).get('avg_processing_time', 0)
            message += f"Average processing time: {avg_time:.2f}s (threshold: {rule.threshold}s)\\n"
        
        message += f"\\nSeverity: {rule.severity.value.upper()}\\n"
        message += f"Timestamp: {datetime.fromtimestamp(time.time()).isoformat()}\\n"
        
        return message
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        # Send email notification
        if self.email_config.get('enabled', False):
            self._send_email_notification(alert)
        
        # Send webhook notification
        if self.webhook_config.get('enabled', False):
            self._send_webhook_notification(alert)
        
        # Send Slack notification
        if self.slack_config.get('enabled', False):
            self._send_slack_notification(alert)
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            smtp_config = self.email_config
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email', '')
            msg['To'] = ', '.join(smtp_config.get('to_emails', []))
            msg['Subject'] = f"LPR Alert: {alert.rule_name} ({alert.severity.value.upper()})"
            
            # Email body
            body = alert.message
            body += "\\n\\n--- Alert Details ---\\n"
            body += f"Rule: {alert.rule_name}\\n"
            body += f"Severity: {alert.severity.value}\\n"
            body += f"Status: {alert.status.value}\\n"
            body += f"Timestamp: {datetime.fromtimestamp(alert.timestamp).isoformat()}\\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config.get('smtp_server', ''), smtp_config.get('smtp_port', 587))
            server.starttls()
            server.login(smtp_config.get('username', ''), smtp_config.get('password', ''))
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for webhook notifications")
            return
        
        try:
            webhook_url = self.webhook_config.get('url', '')
            if not webhook_url:
                return
            
            payload = {
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'status': alert.status.value,
                'metadata': alert.metadata
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert: {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for Slack notifications")
            return
        
        try:
            webhook_url = self.slack_config.get('webhook_url', '')
            if not webhook_url:
                return
            
            # Color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "title": f"LPR Alert: {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Status", 
                                "value": alert.status.value,
                                "short": True
                            }
                        ],
                        "timestamp": int(alert.timestamp)
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert: {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_resolution_notification(self, alert: Alert):
        """Send notification when alert is resolved"""
        # Create resolution message
        resolution_message = f"RESOLVED: {alert.rule_name}\\n\\n"
        resolution_message += f"Alert was active for {(alert.resolved_at - alert.timestamp) / 60:.1f} minutes\\n"
        resolution_message += f"Resolved at: {datetime.fromtimestamp(alert.resolved_at).isoformat()}\\n"
        
        # Send notifications (similar to alert notifications but with resolved status)
        if self.email_config.get('enabled', False):
            # Send resolution email (simplified implementation)
            logger.info(f"Would send resolution email for: {alert.rule_name}")
        
        logger.info(f"Alert resolution notification sent: {alert.rule_name}")
    
    def acknowledge_alert(self, rule_name: str, acknowledged_by: str):
        """Acknowledge an active alert"""
        with self.lock:
            if rule_name in self.alerts:
                alert = self.alerts[rule_name]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert acknowledged: {rule_name} by {acknowledged_by}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        with self.lock:
            return list(self.alerts.values())
    
    def get_alert_history(self, last_hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours"""
        cutoff_time = time.time() - (last_hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_alerts = self.get_active_alerts()
        recent_history = self.get_alert_history(24)
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            'active_alerts': len(active_alerts),
            'alerts_last_24h': len(recent_history),
            'severity_breakdown': severity_counts,
            'alert_rules': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled])
        }
    
    def start_monitoring(self, metrics_collector, check_interval: int = 60):
        """Start alert monitoring loop"""
        if self.monitoring_active:
            logger.warning("Alert monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Get current metrics
                    metrics_data = metrics_collector.get_metrics_summary(last_minutes=10)
                    
                    # Evaluate alert conditions
                    self.evaluate_conditions(metrics_data)
                    
                except Exception as e:
                    logger.error(f"Error in alert monitoring loop: {e}")
                
                time.sleep(check_interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Alert monitoring started (interval: {check_interval}s)")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Alert monitoring stopped")
    
    def save_config(self, config_path: str):
        """Save alert configuration to file"""
        config_data = {
            'alert_rules': {name: asdict(rule) for name, rule in self.alert_rules.items()},
            'email_config': self.email_config,
            'webhook_config': self.webhook_config,
            'slack_config': self.slack_config
        }
        
        # Convert enums to strings for JSON serialization
        for rule_data in config_data['alert_rules'].values():
            rule_data['severity'] = rule_data['severity'].value
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Alert configuration saved to: {config_path}")
    
    def load_config(self, config_path: str):
        """Load alert configuration from file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Load alert rules
        self.alert_rules = {}
        for name, rule_data in config_data.get('alert_rules', {}).items():
            rule_data['severity'] = AlertSeverity(rule_data['severity'])
            rule = AlertRule(**rule_data)
            self.alert_rules[name] = rule
        
        # Load notification configs
        self.email_config = config_data.get('email_config', {})
        self.webhook_config = config_data.get('webhook_config', {})
        self.slack_config = config_data.get('slack_config', {})
        
        logger.info(f"Alert configuration loaded from: {config_path}")


def main():
    """Demo script for alert manager"""
    print("🚨 License Plate Reader Alert Manager Demo")
    print("=" * 50)
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Display alert rules
    print("📋 Alert Rules:")
    for name, rule in alert_manager.alert_rules.items():
        status = "✅" if rule.enabled else "❌"
        print(f"  {status} {name}: {rule.description} ({rule.severity.value})")
    
    # Simulate some metrics that would trigger alerts
    print("\\n🎭 Simulating alert conditions...")
    
    # High CPU usage
    high_cpu_metrics = {
        'system': {
            'avg_cpu_percent': 85.0,  # Above 80% threshold
            'avg_memory_percent': 60.0
        },
        'detection': {
            'success_rate': 0.95,
            'avg_processing_time': 0.5
        },
        'api': {
            'total_requests': 100,
            'status_codes': {'200': 90, '500': 10}  # 10% error rate
        }
    }
    
    # Evaluate conditions
    alert_manager.evaluate_conditions(high_cpu_metrics)
    
    # Check for active alerts
    active_alerts = alert_manager.get_active_alerts()
    if active_alerts:
        print(f"\\n🚨 Active Alerts ({len(active_alerts)}):")
        for alert in active_alerts:
            print(f"  {alert.severity.value.upper()}: {alert.rule_name}")
            print(f"    {alert.message[:100]}...")
    else:
        print("\\n✅ No active alerts (conditions not met for duration threshold)")
    
    # Get alert summary
    summary = alert_manager.get_alert_summary()
    print(f"\\n📊 Alert Summary:")
    print(f"  Active alerts: {summary['active_alerts']}")
    print(f"  Enabled rules: {summary['enabled_rules']}/{summary['alert_rules']}")
    
    print("\\n✅ Alert manager demo completed!")
    print("\\n💡 Next Steps:")
    print("  1. Configure email/webhook/Slack notifications")
    print("  2. Integrate with MetricsCollector for real-time monitoring")
    print("  3. Customize alert rules for your environment")


if __name__ == "__main__":
    main()