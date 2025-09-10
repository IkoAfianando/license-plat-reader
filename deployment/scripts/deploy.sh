set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="license-plate-reader"
DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
ENV_FILE=".env"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Please edit .env file with your configurations before continuing."
            read -p "Press enter to continue after editing .env file..."
        else
            print_error ".env.example file not found. Cannot create .env file."
            exit 1
        fi
    fi
    
    print_status "Prerequisites check completed."
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/{raw,processed,annotations,models,exports}
    mkdir -p models/{pretrained,custom,exports}
    mkdir -p outputs/{images,videos,reports,logs}
    mkdir -p monitoring/{prometheus,grafana/dashboards,grafana/provisioning}
    mkdir -p deployment/nginx/ssl
    
    print_status "Directories created successfully."
}

# Function to download models
download_models() {
    print_status "Downloading pre-trained models..."
    
    # Create models directory
    mkdir -p models/pretrained
    
    # Download YOLOv8 models (they will be downloaded automatically on first use)
    print_status "YOLOv8 models will be downloaded automatically on first use."
    
    print_status "Model setup completed."
}

# Function to setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring configuration..."
    
    # Create Prometheus config
    cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'lpr-app'
    static_configs:
      - targets: ['lpr-app:9090']
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093
EOF

    # Create Grafana provisioning
    mkdir -p monitoring/grafana/provisioning/{dashboards,datasources}
    
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    print_status "Monitoring configuration completed."
}

# Function to setup Nginx
setup_nginx() {
    print_status "Setting up Nginx configuration..."
    
    cat > deployment/nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream lpr_backend {
        server lpr-app:8000;
    }
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name localhost;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # Main API
        location /api/ {
            proxy_pass http://lpr_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            
            # Timeout settings
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
        
        # Health check
        location /health {
            proxy_pass http://lpr_backend/health;
        }
        
        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Monitoring endpoints
        location /metrics {
            proxy_pass http://lpr_backend/metrics;
            allow 127.0.0.1;
            allow 172.16.0.0/12;  # Docker networks
            deny all;
        }
    }
}
EOF

    print_status "Nginx configuration completed."
}

# Function to build and start services
deploy_services() {
    print_status "Building and deploying services..."
    
    # Build the application image
    print_status "Building application image..."
    docker-compose -f $DOCKER_COMPOSE_FILE build
    
    # Start services
    print_status "Starting services..."
    docker-compose -f $DOCKER_COMPOSE_FILE up -d
    
    print_status "Services deployed successfully."
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Wait for services to start
    sleep 10
    
    # Check if services are running
    if docker-compose -f $DOCKER_COMPOSE_FILE ps | grep -q "Up"; then
        print_status "Services are running."
    else
        print_error "Some services failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
    
    # Test API health check
    print_status "Testing API health check..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "API is healthy."
    else
        print_warning "API health check failed. Service may still be starting..."
    fi
    
    print_status "Deployment verification completed."
}

# Function to display deployment info
display_info() {
    print_status "Deployment completed successfully!"
    echo
    echo "🚀 Services are running at:"
    echo "   📡 API Server: http://localhost:8000"
    echo "   🌐 Nginx Proxy: http://localhost:80"
    echo "   📊 Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "   📈 Prometheus: http://localhost:9090"
    echo "   🔍 InfluxDB: http://localhost:8086"
    echo
    echo "📋 Useful commands:"
    echo "   View logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
    echo "   Stop services: docker-compose -f $DOCKER_COMPOSE_FILE down"
    echo "   Restart: docker-compose -f $DOCKER_COMPOSE_FILE restart"
    echo
    echo "📁 Data directories:"
    echo "   Data: ./data/"
    echo "   Models: ./models/"
    echo "   Outputs: ./outputs/"
    echo
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTION]"
    echo "Deploy License Plate Reader system using Docker"
    echo
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --check-only        Only check prerequisites"
    echo "  --setup-only        Only setup configurations"
    echo "  --deploy-only       Only deploy services (skip setup)"
    echo "  --stop              Stop all services"
    echo "  --clean             Stop and remove all containers and volumes"
    echo
}

# Main deployment function
main_deploy() {
    echo "🚗 License Plate Reader Deployment Script"
    echo "=========================================="
    echo
    
    check_prerequisites
    create_directories
    download_models
    setup_monitoring
    setup_nginx
    deploy_services
    verify_deployment
    display_info
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        usage
        exit 0
        ;;
    --check-only)
        check_prerequisites
        exit 0
        ;;
    --setup-only)
        create_directories
        download_models
        setup_monitoring
        setup_nginx
        print_status "Setup completed. Run without --setup-only to deploy."
        exit 0
        ;;
    --deploy-only)
        deploy_services
        verify_deployment
        display_info
        exit 0
        ;;
    --stop)
        print_status "Stopping all services..."
        docker-compose -f $DOCKER_COMPOSE_FILE down
        print_status "All services stopped."
        exit 0
        ;;
    --clean)
        print_status "Stopping and cleaning up all services..."
        docker-compose -f $DOCKER_COMPOSE_FILE down -v --remove-orphans
        docker system prune -f
        print_status "Cleanup completed."
        exit 0
        ;;
    "")
        main_deploy
        ;;
    *)
        print_error "Unknown option: $1"
        usage
        exit 1
        ;;
esac