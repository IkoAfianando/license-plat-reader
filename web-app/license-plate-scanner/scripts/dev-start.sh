#!/bin/bash

echo "🚀 Starting License Plate Scanner (Development Mode)"
echo "===================================================="

# Set ngrok auth
ngrok config add-authtoken 32bYAL7SXhImqo3AtsnG4RjnbLY_7VxTBCgST1q7LY7RFkUQH

# Kill existing processes
pkill -f ngrok || true
pkill -f next || true
sleep 2

# Check FastAPI
echo "🔍 Checking FastAPI server..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ FastAPI server is running"
else
    echo "❌ FastAPI server not running!"
    echo "   Start it with: cd ../.. && PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    pnpm install
fi

# Start ngrok for FastAPI
echo "🌍 Starting FastAPI tunnel..."
ngrok http 8000 --log=stdout > /tmp/fastapi.log 2>&1 &
sleep 5

# Get FastAPI URL
FASTAPI_URL=""
for i in {1..10}; do
    FASTAPI_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n1)
    if [ ! -z "$FASTAPI_URL" ]; then
        break
    fi
    sleep 1
done

if [ -z "$FASTAPI_URL" ]; then
    echo "❌ Failed to get FastAPI ngrok URL"
    exit 1
fi

echo "✅ FastAPI URL: $FASTAPI_URL"

# Update environment
echo "LPR_API_URL=$FASTAPI_URL" > .env.local
echo "NODE_ENV=development" >> .env.local

# Start Next.js in development mode
echo "🚀 Starting Next.js (dev mode)..."
pnpm dev > /tmp/nextjs.log 2>&1 &
sleep 8

# Start ngrok for Next.js
echo "🌍 Starting Web App tunnel..."
ngrok http 3000 --web-addr=localhost:4041 --log=stdout > /tmp/webapp.log 2>&1 &
sleep 5

# Get Web App URL
WEBAPP_URL=""
for i in {1..10}; do
    WEBAPP_URL=$(curl -s http://localhost:4041/api/tunnels 2>/dev/null | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n1)
    if [ ! -z "$WEBAPP_URL" ]; then
        break
    fi
    sleep 1
done

echo ""
echo "🎉 SUCCESS! URLs Ready:"
echo "============================="
if [ ! -z "$WEBAPP_URL" ]; then
    echo "📱 Web App (for phone): $WEBAPP_URL"
else
    echo "📱 Web App (local only): http://localhost:3000"
    echo "   (use ngrok manually if needed)"
fi
echo "🔗 FastAPI Server: $FASTAPI_URL"
echo ""
echo "📲 Copy the Web App URL and open on your phone!"
echo ""
echo "🛑 Press Ctrl+C to stop"

# Keep running
trap 'echo ""; echo "🧹 Stopping..."; pkill -f ngrok; pkill -f next; exit 0' SIGINT SIGTERM
while true; do sleep 1; done