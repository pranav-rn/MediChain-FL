#!/bin/bash

echo "🚀 Starting MediChain-FL Complete System"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down all services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "${YELLOW}⚠️  Virtual environment not found!${NC}"
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo "${BLUE}📦 Installing backend dependencies...${NC}"
pip install -q fastapi uvicorn python-socketio websockets

# 1. Start WebSocket/FL Server
echo ""
echo "${GREEN}1️⃣  Starting Integrated FL Server with Dashboard${NC}"
echo "   - Flower Server: 0.0.0.0:8080"
echo "   - WebSocket: ws://localhost:5000/ws"
echo ""
python backend/fl_server_with_dashboard.py &
SERVER_PID=$!
sleep 5

# 2. Start Frontend (in separate terminal window)
echo ""
echo "${GREEN}2️⃣  Starting Frontend Dashboard${NC}"
echo "   - Dashboard: http://localhost:3000"
echo ""
cd frontend && npm start &
FRONTEND_PID=$!
sleep 3

# 3. Start Hospital Clients
echo ""
echo "${GREEN}3️⃣  Starting Hospital Clients${NC}"
echo ""

sleep 5

echo "${BLUE}   Starting Hospital 1 Client...${NC}"
python backend/fl_client/client.py hospital_1 &
CLIENT1_PID=$!

sleep 2

echo "${BLUE}   Starting Hospital 2 Client...${NC}"
python backend/fl_client/client.py hospital_2 &
CLIENT2_PID=$!

echo ""
echo "========================================="
echo "${GREEN}✅ All services started!${NC}"
echo "========================================="
echo ""
echo "📊 Access Points:"
echo "   - Dashboard: http://localhost:3000"
echo "   - FL Server: localhost:8080"
echo "   - WebSocket: ws://localhost:5000/ws"
echo "   - Health Check: http://localhost:5000/health"
echo ""
echo "🔐 Login Credentials:"
echo "   Admin:    username=admin, password=any"
echo "   Hospital: username=hospital_1, password=any"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all processes
wait