#!/bin/bash

echo "🚀 LICENSE PLATE SCANNER - SUPER SIMPLE START"
echo "============================================="
echo ""

# Set ngrok token
ngrok config add-authtoken 32bYAL7SXhImqo3AtsnG4RjnbLY_7VxTBCgST1q7LY7RFkUQH > /dev/null 2>&1

# Kill existing
pkill -f ngrok > /dev/null 2>&1 || true
pkill -f next > /dev/null 2>&1 || true
sleep 2

echo "🔍 Checking FastAPI server..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ FastAPI not running!"
    echo ""
    echo "Start FastAPI first:"
    echo "cd ../.. && PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    echo ""
    exit 1
fi
echo "✅ FastAPI OK"

# Install deps if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    pnpm install > /dev/null 2>&1
fi

echo "🌍 Starting tunnels..."

# Start FastAPI ngrok
ngrok http 8000 > /dev/null 2>&1 &
sleep 3
FASTAPI_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n1)

# Start Next.js
echo "LPR_API_URL=$FASTAPI_URL" > .env.local
echo "NODE_ENV=development" >> .env.local
pnpm dev > /dev/null 2>&1 &

# Wait for Next.js to fully start
echo "⏳ Waiting for Next.js to be ready..."
for i in {1..15}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Next.js is ready"
        break
    fi
    sleep 2
done

# Start ngrok for Next.js frontend
echo "🌍 Starting frontend ngrok tunnel..."
ngrok http 3000 --web-addr=localhost:4041 > /dev/null 2>&1 &
sleep 5

# Get frontend ngrok URL
WEBAPP_URL=""
for i in {1..10}; do
    WEBAPP_URL=$(curl -s http://localhost:4041/api/tunnels 2>/dev/null | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n1)
    if [ ! -z "$WEBAPP_URL" ]; then
        break
    fi
    sleep 1
done

echo ""
echo "🎉 READY TO USE!"
echo "=================="
echo ""
if [ ! -z "$WEBAPP_URL" ]; then
    echo "📱 FRONTEND NGROK URL (for phone):"
    echo "   $WEBAPP_URL"
    echo ""
    echo "🏠 Local URL (if on the same network):"  
    echo "   http://$(hostname -I | cut -d' ' -f1):3000"
else
    echo "📱 FRONTEND URL:"  
    echo "   http://$(hostname -I | cut -d' ' -f1):3000"
    echo "   (frontend ngrok failed, use local IP)"
fi
echo ""
echo "🔗 BACKEND API: $FASTAPI_URL"
echo ""
echo "📲 Next steps:"
echo "   1. Copy the FRONTEND NGROK URL above"
echo "   2. Open it in your phone's browser (from anywhere via internet)"
echo "   3. Grant camera permission"
echo "   4. Choose YOLO or Roboflow"  
echo "   5. Test the CCTV Live Mode feature!"
echo ""
echo "🛑 Press Ctrl+C to stop all services"

# Keep running
trap 'echo ""; echo "🧹 Stopping..."; pkill -f ngrok; pkill -f next; exit 0' SIGINT SIGTERM
while true; do sleep 1; done
