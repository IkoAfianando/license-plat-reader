#!/bin/bash

# Simple script to get ngrok URLs directly

echo "🔗 Getting ngrok URLs..."

# Set auth token (your token is already included)
ngrok config add-authtoken 32bYAL7SXhImqo3AtsnG4RjnbLY_7VxTBCgST1q7LY7RFkUQH

# Kill existing ngrok
pkill -f ngrok 2>/dev/null || true
sleep 1

# Start ngrok for FastAPI (port 8000)
ngrok http 8000 --log=stdout > /tmp/ngrok1.log 2>&1 &
echo "⏳ Starting FastAPI tunnel..."
sleep 4

# Get FastAPI URL
FASTAPI_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//g')

if [ ! -z "$FASTAPI_URL" ]; then
    echo "✅ FastAPI: $FASTAPI_URL"
    
    # Update env file
    echo "LPR_API_URL=$FASTAPI_URL" > .env.local
    echo "NODE_ENV=development" >> .env.local
    
    # Build and start Next.js if not running
    if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "🔨 Building Next.js..."
        pnpm build > /dev/null 2>&1
        echo "🚀 Starting Next.js..."
        pnpm start > /dev/null 2>&1 &
        sleep 8
    fi
    
    # Start second ngrok for Next.js (port 3000) with different web interface port
    ngrok http 3000 --web-addr=localhost:4041 --log=stdout > /tmp/ngrok2.log 2>&1 &
    echo "⏳ Starting Web App tunnel..."
    sleep 6
    
    # Get Web App URL with retry
    WEBAPP_URL=""
    for i in {1..5}; do
        WEBAPP_URL=$(curl -s http://localhost:4041/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//g' | head -n1)
        if [ ! -z "$WEBAPP_URL" ]; then
            break
        fi
        echo "   Retry $i/5..."
        sleep 2
    done
    
    echo ""
    echo "🎉 URLs Ready:"
    echo "📱 Web App (for phone): $WEBAPP_URL"
    echo "🔗 FastAPI Server: $FASTAPI_URL"
    echo ""
    echo "📲 Open Web App URL on your phone to start scanning!"
    
else
    echo "❌ Failed to get ngrok URL"
    echo "Make sure ngrok is installed and try again"
fi