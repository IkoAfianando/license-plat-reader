#!/bin/bash

# Script to start the License Plate Scanner web app with ngrok

echo "🚀 Starting License Plate Scanner with ngrok..."

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed. Please install ngrok first:"
    echo "   - Visit https://ngrok.com/download"
    echo "   - Or use: npm install -g @ngrok/ngrok"
    exit 1
fi

# Set auth token if provided
NGROK_AUTH_TOKEN="32bYAL7SXhImqo3AtsnG4RjnbLY_7VxTBCgST1q7LY7RFkUQH"
if [ ! -z "$NGROK_AUTH_TOKEN" ]; then
    echo "🔑 Setting ngrok auth token..."
    ngrok config add-authtoken $NGROK_AUTH_TOKEN
fi

# Check if FastAPI server is running
echo "🔍 Checking if FastAPI server is running on port 8000..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "⚠️  FastAPI server not detected on port 8000"
    echo "   Please start your FastAPI server first:"
    echo "   cd ../.. && python src/api/main.py"
    echo ""
    echo "   Or run it in another terminal:"
    echo "   cd ../../ && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    pnpm install
fi

# Start ngrok for FastAPI in background
echo "🌍 Starting ngrok tunnel for FastAPI (port 8000)..."
ngrok http 8000 --log=stdout > /tmp/ngrok-fastapi.log 2>&1 &
NGROK_FASTAPI_PID=$!

# Wait for ngrok to start
echo "⏳ Waiting for ngrok tunnel to be ready..."
sleep 5

# Function to get ngrok URL
get_ngrok_url() {
    local port=$1
    local retries=0
    local max_retries=10
    
    while [ $retries -lt $max_retries ]; do
        local url=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//g' | head -n1)
        if [ ! -z "$url" ]; then
            echo $url
            return 0
        fi
        retries=$((retries + 1))
        sleep 2
    done
    return 1
}

# Get FastAPI ngrok URL
FASTAPI_NGROK_URL=$(get_ngrok_url 8000)
if [ -z "$FASTAPI_NGROK_URL" ]; then
    echo "❌ Failed to get FastAPI ngrok URL after waiting"
    echo "📋 Check ngrok status manually:"
    echo "   curl -s http://localhost:4040/api/tunnels | jq"
    kill $NGROK_FASTAPI_PID 2>/dev/null
    exit 1
fi

echo "✅ FastAPI available at: $FASTAPI_NGROK_URL"

# Update environment variables
export LPR_API_URL=$FASTAPI_NGROK_URL
echo "LPR_API_URL=$FASTAPI_NGROK_URL" > .env.local
echo "NODE_ENV=development" >> .env.local

# Build the Next.js app
echo "🔨 Building Next.js app..."
pnpm build

# Start Next.js app
echo "🚀 Starting Next.js app..."
pnpm start &
NEXTJS_PID=$!

# Wait for Next.js to start
sleep 5

# Start ngrok for Next.js on a different port (4041)
echo "🌍 Starting ngrok tunnel for Next.js (port 3000)..."
ngrok http 3000 --web-addr=localhost:4041 --log=stdout > /tmp/ngrok-nextjs.log 2>&1 &
NGROK_NEXTJS_PID=$!

# Wait for second ngrok to start
echo "⏳ Waiting for Next.js ngrok tunnel..."
sleep 5

# Get Next.js ngrok URL
get_nextjs_url() {
    local retries=0
    local max_retries=10
    
    while [ $retries -lt $max_retries ]; do
        local url=$(curl -s http://localhost:4041/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//g' | head -n1)
        if [ ! -z "$url" ]; then
            echo $url
            return 0
        fi
        retries=$((retries + 1))
        sleep 2
    done
    return 1
}

NEXTJS_NGROK_URL=$(get_nextjs_url)

if [ -z "$NEXTJS_NGROK_URL" ]; then
    echo "❌ Failed to get Next.js ngrok URL"
    echo "📱 You can still access the app locally at: http://localhost:3000"
    echo "   Make sure to update the API URL in the app settings to: $FASTAPI_NGROK_URL"
    echo ""
    echo "🔗 FastAPI: $FASTAPI_NGROK_URL"
else
    echo ""
    echo "🎉 License Plate Scanner is ready!"
    echo "📱 Web App: $NEXTJS_NGROK_URL"
    echo "🔗 FastAPI: $FASTAPI_NGROK_URL"
    echo ""
    echo "📲 Open the Web App URL on your mobile device to start scanning license plates!"
fi

echo ""
echo "🛑 To stop all services, press Ctrl+C"

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    kill $NGROK_FASTAPI_PID 2>/dev/null
    kill $NGROK_NEXTJS_PID 2>/dev/null
    kill $NEXTJS_PID 2>/dev/null
    rm -f /tmp/ngrok-*.log
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Keep script running
wait