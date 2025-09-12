#!/bin/bash

# Quick start script - Get URLs immediately

echo "🚀 Quick Start License Plate Scanner"
echo "=================================="

# Set auth token
NGROK_AUTH_TOKEN="32bYAL7SXhImqo3AtsnG4RjnbLY_7VxTBCgST1q7LY7RFkUQH"
ngrok config add-authtoken $NGROK_AUTH_TOKEN

# Check if FastAPI is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "⚠️  FastAPI server not running on port 8000"
    echo "   Start it first: cd ../.. && python src/api/main.py"
    echo ""
fi

# Install deps if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    pnpm install
fi

echo ""
echo "🌍 Starting ngrok tunnels..."
echo ""

# Kill any existing ngrok processes
pkill -f ngrok 2>/dev/null || true
sleep 2

# Start ngrok for FastAPI
echo "🔗 FastAPI tunnel (port 8000):"
ngrok http 8000 > /dev/null 2>&1 &
sleep 3
FASTAPI_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//g' | head -n1)
echo "   $FASTAPI_URL"

# Update env file
echo "LPR_API_URL=$FASTAPI_URL" > .env.local
echo "NODE_ENV=development" >> .env.local

# Build and start Next.js
echo ""
echo "🔨 Building Next.js..."
pnpm build > /dev/null 2>&1

echo ""
echo "🚀 Starting Next.js..."
pnpm start > /dev/null 2>&1 &
sleep 5

# Start ngrok for Next.js
echo ""
echo "📱 Web App tunnel (port 3000):"
ngrok http 3000 --web-addr=localhost:4041 > /dev/null 2>&1 &
sleep 3
WEBAPP_URL=$(curl -s http://localhost:4041/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//g' | head -n1)
echo "   $WEBAPP_URL"

echo ""
echo "🎉 Ready to use!"
echo "=================================="
echo "📱 Open this URL on your phone: $WEBAPP_URL"
echo "🔗 FastAPI URL: $FASTAPI_URL"
echo ""
echo "📲 Grant camera permission and start scanning!"
echo ""
echo "🛑 Press Ctrl+C to stop"

# Keep running
trap 'echo ""; echo "🧹 Stopping services..."; pkill -f ngrok; pkill -f next; exit 0' SIGINT SIGTERM
while true; do
    sleep 1
done