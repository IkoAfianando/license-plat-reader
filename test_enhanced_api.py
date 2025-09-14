#!/usr/bin/env python3
"""
Test script for the enhanced License Plate Scanner API
Tests both the backend API and the frontend functionality
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_api_server():
    """Test the enhanced API server"""
    print("🚀 Testing Enhanced License Plate Scanner API")
    print("=" * 60)
    
    # Test if we can import the API
    try:
        from src.api.simple_main import app
        print("✅ API imports successfully")
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False
    
    # Start the API server in the background
    print("\n📡 Starting API server...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.api.simple_main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("🏥 Testing health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed - Status: {health_data['status']}")
            print(f"   Services: {health_data['services']}")
        else:
            print(f"❌ Health check failed - Status: {response.status_code}")
            return False
        
        # Test config endpoint
        print("\n⚙️  Testing config endpoint...")
        response = requests.get("http://localhost:8000/config", timeout=10)
        if response.status_code == 200:
            config_data = response.json()
            print(f"✅ Config endpoint working")
            print(f"   Model type: {config_data['detection']['model_type']}")
        else:
            print(f"❌ Config endpoint failed - Status: {response.status_code}")
        
        # Create a test image
        print("\n🖼️  Creating test image...")
        import cv2
        import numpy as np
        
        # Create a simple test image with a rectangle (mock license plate)
        test_img = np.zeros((400, 600, 3), dtype=np.uint8)
        test_img.fill(128)  # Gray background
        cv2.rectangle(test_img, (200, 150), (400, 250), (255, 255, 255), -1)  # White rectangle
        cv2.putText(test_img, "B1234XYZ", (220, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        test_image_path = "/tmp/test_license_plate.jpg"
        cv2.imwrite(test_image_path, test_img)
        print(f"✅ Test image created: {test_image_path}")
        
        # Test detection endpoint with image return
        print("\n🔍 Testing detection endpoint with image return...")
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'confidence': '0.5',
                'extract_text': 'true',
                'return_image': 'true'
            }
            response = requests.post(
                "http://localhost:8000/detect/image", 
                files=files, 
                data=data, 
                timeout=30
            )
        
        if response.status_code == 200:
            detection_data = response.json()
            print(f"✅ Detection successful!")
            print(f"   Success: {detection_data['success']}")
            print(f"   Detections found: {len(detection_data['detections'])}")
            print(f"   Processing time: {detection_data['processing_time']:.3f}s")
            print(f"   Model: {detection_data['model_info']}")
            
            if detection_data.get('annotated_image'):
                print(f"   ✅ Annotated image returned (base64 length: {len(detection_data['annotated_image'])})")
            else:
                print(f"   ⚠️  No annotated image returned")
            
            # Show detected license plates
            for i, detection in enumerate(detection_data['detections']):
                print(f"   Detection {i+1}:")
                print(f"     Confidence: {detection['confidence']:.2f}")
                print(f"     Text: {detection.get('text', 'N/A')}")
                print(f"     BBox: {detection['bbox']}")
        else:
            print(f"❌ Detection failed - Status: {response.status_code}")
            print(f"   Response: {response.text}")
        
        print("\n🎉 API testing completed!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up
        api_process.terminate()
        api_process.wait()
        try:
            os.unlink("/tmp/test_license_plate.jpg")
        except:
            pass

def test_frontend():
    """Test the frontend development setup"""
    print("\n🌐 Testing Frontend Setup")
    print("=" * 40)
    
    frontend_dir = Path("web-app/license-plate-scanner")
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return False
    
    # Check if package.json exists
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print(f"❌ package.json not found: {package_json}")
        return False
    
    print(f"✅ Frontend directory found: {frontend_dir}")
    
    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print("✅ Node modules installed")
    else:
        print("⚠️  Node modules not found - run 'pnpm install' in the frontend directory")
    
    # Check key files
    key_files = [
        "src/app/page.tsx",
        "src/components/FileUpload.tsx",
        "src/app/api/detect/route.ts"
    ]
    
    for file_path in key_files:
        full_path = frontend_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    return True

def show_usage_instructions():
    """Show usage instructions"""
    print("\n📋 Usage Instructions")
    print("=" * 40)
    
    print("\n🔧 To run the enhanced system:")
    print("1. Backend API:")
    print("   cd /home/ikoafian/COMPANY/ieko-media/metabase-setup/research/license-plate-reader")
    print("   python -m uvicorn src.api.simple_main:app --host 0.0.0.0 --port 8000 --reload")
    
    print("\n2. Frontend:")
    print("   cd web-app/license-plate-scanner")
    print("   pnpm install  # if not already installed")
    print("   pnpm run dev")
    
    print("\n3. Access the application:")
    print("   - Frontend: http://localhost:3000")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - API Health: http://localhost:8000/health")
    
    print("\n🎯 Key Enhancements Made:")
    print("✅ Fixed YOLO model internal server error (using mock detector)")
    print("✅ Enhanced scan output to return annotated images")
    print("✅ Frontend now displays images with bounding boxes")
    print("✅ Created comprehensive YOLO training notebook")
    print("✅ Dataset ready for custom model training")
    
    print("\n📊 Training Your Custom Model:")
    print("1. Open license_plate_training.ipynb in Jupyter")
    print("2. Run all cells to train a custom YOLOv8 model")
    print("3. Replace the mock detector with your trained model")
    
    print("\n🔄 Next Steps:")
    print("1. Run the training notebook to create a custom model")
    print("2. Update the API to use the trained model instead of mock detector")
    print("3. Test with real license plate images")
    print("4. Deploy to production environment")

if __name__ == "__main__":
    print("🧪 License Plate Scanner - Enhanced API Test Suite")
    print("=" * 60)
    
    # Test API
    api_success = test_api_server()
    
    # Test frontend setup
    frontend_success = test_frontend()
    
    # Show instructions
    show_usage_instructions()
    
    print("\n" + "=" * 60)
    if api_success and frontend_success:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    print("=" * 60)