#!/bin/bash

echo "🚀 Starting License Plate Scanner"
echo "================================="

# Set ngrok auth
ngrok config add-authtoken 32bYAL7SXhImqo3AtsnG4RjnbLY_7VxTBCgST1q7LY7RFkUQH

# Kill existing ngrok and node processes
echo "🧹 Cleaning up existing processes..."
pkill -f ngrok || true
pkill -f next || true
sleep 2

# Check FastAPI
echo "🔍 Checking FastAPI server..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ FastAPI server is running"
else
    echo "❌ FastAPI server not running on port 8000"
    echo "   Please start it with: PYTHONPATH=/path/to/project uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    pnpm install
fi

# Build Next.js
echo "🔨 Building Next.js..."
pnpm build

# Start ngrok for FastAPI first
echo "🌍 Starting FastAPI tunnel..."
ngrok http 8000 --log=stdout > /dev/null 2>&1 &
sleep 5

# Get FastAPI URL
FASTAPI_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url' 2>/dev/null)
if [ "$FASTAPI_URL" = "null" ] || [ -z "$FASTAPI_URL" ]; then
    FASTAPI_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n1)
fi

if [ ! -z "$FASTAPI_URL" ] && [ "$FASTAPI_URL" != "null" ]; then
    echo "✅ FastAPI URL: $FASTAPI_URL"
    
    # Update environment
    echo "LPR_API_URL=$FASTAPI_URL" > .env.local
    echo "NODE_ENV=production" >> .env.local
    
    # Start Next.js
    echo "🚀 Starting Next.js server..."
    pnpm start &
    sleep 5
    
    # Start second ngrok for Next.js
    echo "🌍 Starting Web App tunnel..."
    ngrok http 3000 --web-addr=localhost:4041 --log=stdout > /dev/null 2>&1 &
    sleep 5
    
    # Get Web App URL
    WEBAPP_URL=$(curl -s http://localhost:4041/api/tunnels | jq -r '.tunnels[0].public_url' 2>/dev/null)
    if [ "$WEBAPP_URL" = "null" ] || [ -z "$WEBAPP_URL" ]; then
        WEBAPP_URL=$(curl -s http://localhost:4041/api/tunnels | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n1)
    fi
    
    echo ""
    echo "🎉 SUCCESS! URLs Ready:"
    echo "================================="
    echo "📱 Web App (open on phone): $WEBAPP_URL"
    echo "🔗 FastAPI Server: $FASTAPI_URL"
    echo ""
    echo "📲 Copy the Web App URL and open it on your phone!"
    echo "   Grant camera permission and start scanning license plates."
    echo ""
    echo "🛑 Press Ctrl+C to stop all services"
    echo ""
    
    # Keep running
    trap 'echo ""; echo "🧹 Stopping all services..."; pkill -f ngrok; pkill -f next; exit 0' SIGINT SIGTERM
    while true; do sleep 1; done
    
else
    echo "❌ Failed to get FastAPI ngrok URL"
    exit 1
fi