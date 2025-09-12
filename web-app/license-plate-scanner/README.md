# License Plate Scanner Web App

A mobile-friendly Next.js web application for scanning license plates using your existing FastAPI server with YOLO and Roboflow detection models.

## Features

- 📱 **Mobile-First Design**: Optimized for smartphone cameras
- 📷 **Camera Integration**: Direct access to device camera with `react-webcam`
- 🤖 **Dual AI Models**: Support for both YOLO local and Roboflow API detection
- 🔗 **Ngrok Ready**: Easy deployment with ngrok tunneling
- ⚡ **Real-time Detection**: Fast license plate scanning and OCR
- 🎨 **Modern UI**: Clean interface built with Tailwind CSS
- **📹 NEW: CCTV Live Mode**: Continuous scanning like security camera
- **🔴 Live Feed**: Real-time detection every 2 seconds with history
- **📊 Detection History**: Keep track of all detected plates in live mode

## Quick Start

### Prerequisites

1. **FastAPI Server**: Make sure your license plate detection API is running on port 8000
2. **Node.js & pnpm**: Install Node.js and pnpm package manager
3. **Ngrok**: For mobile access via internet tunnel

### Installation

```bash
# Install dependencies
pnpm install
```

## 🚀 CARA SUPER MUDAH - SATU PERINTAH SAJA!

### 1. Start FastAPI Server (terminal 1):
```bash
cd ../..
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Start Website Mobile (terminal 2):
```bash
cd license-plate-scanner
./start-mobile.sh
```

**SELESAI!** Script akan memberikan **NGROK URL** yang bisa dibuka di handphone dari mana saja!

### Alternative Scripts:
- `./start.sh` - Local network access  
- `./start-mobile.sh` - **Internet access via ngrok (RECOMMENDED)**

---

### Alternative Methods (jika butuh kontrol lebih)

**Method 2 - Manual ngrok:**
```bash
./scripts/get-urls.sh     # Get URLs instantly
./scripts/dev-start.sh    # Development mode
```

### Manual Development

```bash
# Start development server
pnpm dev

# Open http://localhost:3000 in your browser
```

## Usage on Mobile Phone

1. **Pastikan FastAPI server berjalan** di komputer/server Anda
2. **Jalankan script start**:
   ```bash
   ./start.sh
   ```
3. **Buka URL** yang diberikan di browser handphone Anda
4. **Berikan izin akses kamera** ketika diminta
5. **Pilih model deteksi**:
   - **YOLO Local**: Menggunakan model YOLOv8 lokal Anda
   - **Roboflow API**: Menggunakan Roboflow cloud service

### 📸 Single Capture Mode
6. **Arahkan kamera ke plat nomor** dan tekan "📸 Capture & Scan"
7. **Lihat hasil deteksi** termasuk teks plat nomor dan confidence score

### 📹 CCTV Live Mode (NEW!)
6. **Tekan tombol "📹 Start CCTV Mode"** untuk mulai monitoring
7. **Kamera akan otomatis scan** setiap 2 detik
8. **Semua plat nomor yang terdeteksi** akan muncul di live feed
9. **History detection** tersimpan sampai Anda clear atau stop
10. **Tekan "🔴 Stop CCTV Mode"** untuk berhenti

## Configuration

The app will automatically configure the API connection when using the ngrok script. For manual configuration, update `.env.local`:

```bash
# FastAPI server URL 
LPR_API_URL=http://localhost:8000  # Or your ngrok URL

# Next.js configuration
NODE_ENV=development
```

## API Integration

The web app connects to your existing FastAPI endpoints:

- `POST /detect/image` - Main detection endpoint (supports both YOLO and Roboflow)
- `GET /health` - Health check endpoint

## Troubleshooting

### Camera Issues
- Grant camera permissions in browser
- Use HTTPS (ngrok automatically provides this)
- Try Chrome browser for best compatibility

### API Connection Issues
- Verify FastAPI server is running: `curl http://localhost:8000/health`
- Check if ngrok tunnels are active
- Look at browser console for error messages

### Detection Issues
- Ensure YOLO model (yolov8x.pt) exists in project root
- For Roboflow: Check API key configuration in FastAPI server
- Verify good lighting and clear view of license plate

## Project Structure

```
src/
├── app/
│   ├── api/              # Next.js API routes
│   │   ├── detect/       # Detection proxy endpoint  
│   │   └── health/       # Health check endpoint
│   ├── layout.tsx        # Root layout with mobile optimizations
│   └── page.tsx          # Main page
├── components/
│   └── CameraCapture.tsx # Camera component with detection
└── ...

scripts/
└── start-with-ngrok.sh   # Automated setup script

public/
├── manifest.json         # PWA manifest for mobile app-like experience
└── ...
```
