#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      MediChain-FL Complete Setup Script                     ║"
echo "║      Setting up Backend + Frontend + WebSocket               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Error handling
set -e
trap 'echo -e "${RED}❌ Setup failed. Check errors above.${NC}"; exit 1' ERR

echo -e "${BLUE}📋 Step 1: Checking Prerequisites${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} found"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} Node.js ${NODE_VERSION} found"
else
    echo -e "${RED}✗${NC} Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} npm ${NPM_VERSION} found"
else
    echo -e "${RED}✗${NC} npm not found. Please install npm"
    exit 1
fi

echo ""
echo -e "${BLUE}📦 Step 2: Setting Up Python Environment${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "Upgrading pip..."
pip install --quiet --upgrade pip

# Install existing requirements
if [ -f "requirements.txt" ]; then
    echo "Installing existing requirements..."
    pip install --quiet -r requirements.txt
    echo -e "${GREEN}✓${NC} Base requirements installed"
fi

# Install WebSocket dependencies
echo "Installing WebSocket server dependencies..."
pip install --quiet fastapi==0.110.0 uvicorn[standard]==0.27.0 python-socketio==5.11.0 websockets==12.0
echo -e "${GREEN}✓${NC} WebSocket dependencies installed"

echo ""
echo -e "${BLUE}🎨 Step 3: Setting Up Frontend${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create frontend directory if it doesn't exist
if [ ! -d "frontend" ]; then
    echo "Creating frontend directory..."
    mkdir -p frontend
fi

cd frontend

# Create directory structure
echo "Creating frontend directory structure..."
mkdir -p src/components
mkdir -p src/hooks
mkdir -p public

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo -e "${YELLOW}⚠️  package.json not found${NC}"
    echo "   Please copy package.json from the artifacts"
    echo "   Then run: npm install"
else
    echo "Installing npm dependencies..."
    npm install
    echo -e "${GREEN}✓${NC} npm dependencies installed"
fi

# Check for Tailwind config
if [ ! -f "tailwind.config.js" ]; then
    echo "Initializing Tailwind CSS..."
    npx tailwindcss init -p
    echo -e "${GREEN}✓${NC} Tailwind initialized"
fi

cd ..

echo ""
echo -e "${BLUE}📁 Step 4: Verifying File Structure${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check backend files
echo "Checking backend files..."
BACKEND_FILES=(
    "backend/websocket_server.py"
    "backend/fl_server_with_dashboard.py"
    "backend/fl_client/client.py"
    "backend/model.py"
)

MISSING_BACKEND=0
for file in "${BACKEND_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file ${YELLOW}(MISSING - Copy from artifacts)${NC}"
        MISSING_BACKEND=1
    fi
done

# Check frontend files
echo ""
echo "Checking frontend files..."
FRONTEND_FILES=(
    "frontend/src/App.js"
    "frontend/src/index.js"
    "frontend/src/hooks/useWebSocket.js"
    "frontend/src/components/LoginScreen.js"
    "frontend/src/components/AggregatorDashboard.js"
    "frontend/src/components/HospitalDashboard.js"
    "frontend/.env"
)

MISSING_FRONTEND=0
for file in "${FRONTEND_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file ${YELLOW}(MISSING - Copy from artifacts)${NC}"
        MISSING_FRONTEND=1
    fi
done

echo ""
echo -e "${BLUE}🧪 Step 5: Running Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test WebSocket server
echo "Testing WebSocket server import..."
if python3 -c "import backend.websocket_server" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} WebSocket server can be imported"
else
    echo -e "${YELLOW}⚠️${NC}  WebSocket server import failed (may need file creation)"
fi

# Test model
echo "Testing model import..."
if python3 -c "from backend.model import load_model" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Model can be imported"
else
    echo -e "${YELLOW}⚠️${NC}  Model import failed"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Summary                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ $MISSING_BACKEND -eq 0 ] && [ $MISSING_FRONTEND -eq 0 ]; then
    echo -e "${GREEN}✅ Setup Complete! All files present.${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start the system:"
    echo "   ${BLUE}./scripts/run_complete_system.sh${NC}"
    echo ""
    echo "2. Or manually:"
    echo "   Terminal 1: ${BLUE}python backend/fl_server_with_dashboard.py${NC}"
    echo "   Terminal 2: ${BLUE}cd frontend && npm start${NC}"
    echo "   Terminal 3: ${BLUE}python backend/fl_client/client.py hospital_1${NC}"
    echo "   Terminal 4: ${BLUE}python backend/fl_client/client.py hospital_2${NC}"
    echo ""
    echo "3. Access dashboard: ${BLUE}http://localhost:3000${NC}"
else
    echo -e "${YELLOW}⚠️  Setup Incomplete${NC}"
    echo ""
    echo "Missing files detected. Please:"
    echo ""
    if [ $MISSING_BACKEND -eq 1 ]; then
        echo "1. Copy backend files from artifacts:"
        echo "   - backend/websocket_server.py"
        echo "   - backend/fl_server_with_dashboard.py"
    fi
    if [ $MISSING_FRONTEND -eq 1 ]; then
        echo "2. Copy frontend files from artifacts:"
        echo "   - All files in frontend/src/"
        echo "   - frontend/.env"
    fi
    echo ""
    echo "3. Then run this script again:"
    echo "   ${BLUE}./scripts/complete_setup.sh${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "For detailed instructions, see: ${BLUE}FRONTEND_INTEGRATION_GUIDE.md${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"