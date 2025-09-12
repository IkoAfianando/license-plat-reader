#!/bin/bash

echo "🚀 LICENSE PLATE SCANNER - MOBILE NGROK VERSION"
echo "================================================"
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

# Set environment to use direct FastAPI (since we'll access via ngrok)
echo "LPR_API_URL=http://localhost:8000" > .env.local
echo "NODE_ENV=development" >> .env.local

# Start Next.js
echo "🚀 Starting Next.js..."
pnpm dev > /dev/null 2>&1 &

# Wait for Next.js to be ready
echo "⏳ Waiting for Next.js to be ready..."
for i in {1..15}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Next.js is ready"
        break
    fi
    sleep 2
done

# Start ngrok ONLY for frontend (Next.js will proxy to FastAPI internally)
echo "🌍 Starting ngrok tunnel for frontend..."
ngrok http 3000 > /dev/null 2>&1 &
sleep 5

# Get frontend ngrok URL
WEBAPP_URL=""
for i in {1..10}; do
    WEBAPP_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n1)
    if [ ! -z "$WEBAPP_URL" ]; then
        break
    fi
    sleep 1
done

echo ""
echo "🎉 MOBILE ACCESS READY!"
echo "======================"
echo ""
if [ ! -z "$WEBAPP_URL" ]; then
    echo "📱 BUKA URL INI DI HANDPHONE (dari mana saja via internet):"
    echo "   $WEBAPP_URL"
    echo ""
    echo "✨ URL ini sudah termasuk akses ke FastAPI server!"
else
    echo "❌ Ngrok gagal. Gunakan IP lokal:"
    echo "   http://$(hostname -I | cut -d' ' -f1):3000"
fi
echo ""
echo "🔗 FastAPI server: http://localhost:8000 (internal)"
echo ""
echo "📲 Langkah selanjutnya:"
echo "   1. Copy URL ngrok di atas"
echo "   2. Buka di browser handphone"
echo "   3. Berikan izin kamera"
echo "   4. Pilih Next.js API mode (recommended)"
echo "   5. Test fitur CCTV Live Mode!"
echo ""
echo "💡 Tips: Website akan otomatis proxy ke FastAPI server lokal"
echo ""
echo "🛑 Press Ctrl+C to stop"

# Keep running
trap 'echo ""; echo "🧹 Stopping..."; pkill -f ngrok; pkill -f next; exit 0' SIGINT SIGTERM
while true; do sleep 1; done