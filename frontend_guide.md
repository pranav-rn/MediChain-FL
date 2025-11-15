# MediChain-FL Frontend Integration Guide

Complete setup guide for integrating the React dashboard with your existing Federated Learning backend.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Backend Setup](#backend-setup)
4. [Frontend Setup](#frontend-setup)
5. [Running the System](#running-the-system)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python 3.10+** 
- **Node.js 18+ and npm 9+**
- **Git**

### Check Versions

```bash
python --version  # Should be 3.10+
node --version    # Should be v18+
npm --version     # Should be 9+
```

---

## Project Structure

Your final structure should look like this:

```
MediChain-FL/
├── backend/
│   ├── fl_client/
│   │   ├── client.py           # Existing
│   │   └── server.py           # Existing
│   ├── websocket_server.py     # NEW - WebSocket server
│   ├── fl_server_with_dashboard.py  # NEW - Integrated server
│   ├── model.py                # Existing
│   └── requirements.txt        # Updated
├── frontend/                    # NEW - Complete frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LoginScreen.js
│   │   │   ├── AggregatorDashboard.js
│   │   │   └── HospitalDashboard.js
│   │   ├── hooks/
│   │   │   └── useWebSocket.js
│   │   ├── App.js
│   │   ├── index.js
│   │   ├── index.css
│   │   └── App.css
│   ├── package.json
│   ├── tailwind.config.js
│   └── .env
├── data/
│   ├── hospital_1/
│   └── hospital_2/
└── scripts/
    ├── setup_frontend.sh
    └── run_complete_system.sh
```

---

## Backend Setup

### Step 1: Install Additional Python Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install WebSocket server dependencies
pip install fastapi==0.110.0 uvicorn[standard]==0.27.0 python-socketio==5.11.0 websockets==12.0
```

### Step 2: Create New Backend Files

Create these NEW files in your `backend/` directory:

1. **`backend/websocket_server.py`** - Copy from artifact "websocket-server"
2. **`backend/fl_server_with_dashboard.py`** - Copy from artifact "integrated-fl-server"

### Step 3: Test WebSocket Server

```bash
# Test standalone WebSocket server
python backend/websocket_server.py
```

You should see:
```
🚀 Starting MediChain-FL WebSocket Server
======================================================================
   WebSocket URL: ws://0.0.0.0:5000/ws
   Health Check: http://0.0.0.0:5000/health
```

Test health endpoint:
```bash
curl http://localhost:5000/health
```

---

## Frontend Setup

### Step 1: Create Frontend Directory Structure

```bash
# From project root
mkdir -p frontend/src/components
mkdir -p frontend/src/hooks
mkdir -p frontend/public
```

### Step 2: Initialize React App

```bash
cd frontend

# Create package.json
# Copy content from artifact "frontend-package-json"

# Install dependencies
npm install
```

### Step 3: Configure Tailwind CSS

```bash
# Initialize Tailwind
npx tailwindcss init -p
```

Then create `frontend/tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#4A90E2',
        secondary: '#50E3C2',
      }
    },
  },
  plugins: [],
}
```

### Step 4: Create All Component Files

Create these files and copy content from the artifacts:

**Core Files:**
1. **`src/index.js`** - Copy from "frontend-index-js"
2. **`src/App.js`** - Copy from "frontend-app-js"
3. **`src/index.css`** - Add Tailwind directives
4. **`src/App.css`** - Copy from "frontend-app-css"

**Hooks:**
5. **`src/hooks/useWebSocket.js`** - Copy from "use-websocket-hook"

**Components:**
6. **`src/components/LoginScreen.js`** - Copy from "login-screen-component"
7. **`src/components/AggregatorDashboard.js`** - Copy from "aggregator-dashboard"
8. **`src/components/HospitalDashboard.js`** - Copy from "hospital-dashboard"

**Configuration:**
9. **`frontend/.env`** - Copy from "frontend-env-file"

### Step 5: Update `index.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
```

### Step 6: Test Frontend

```bash
# From frontend directory
npm start
```

Browser should open to `http://localhost:3000` with the login screen.

---

## Running the System

### Option 1: Manual Start (Recommended for First Time)

**Terminal 1: WebSocket + FL Server**
```bash
cd backend
python fl_server_with_dashboard.py --rounds 5 --min-clients 2
```

**Terminal 2: Frontend Dashboard**
```bash
cd frontend
npm start
```

**Terminal 3: Hospital 1 Client**
```bash
cd backend
python fl_client/client.py hospital_1
```

**Terminal 4: Hospital 2 Client**
```bash
cd backend
python fl_client/client.py hospital_2
```

### Option 2: Automated Script

```bash
# Make script executable
chmod +x scripts/run_complete_system.sh

# Run everything
./scripts/run_complete_system.sh
```

---

## Usage Flow

### 1. Access Dashboard

Open browser to `http://localhost:3000`

### 2. Login

**As Admin (Aggregator View):**
- Username: `admin`
- Password: `anything`

**As Hospital (Client View):**
- Username: `hospital_1`
- Password: `anything`

### 3. Start Training (Admin View)

1. Click "Start Server" button
2. Wait for clients to connect
3. Watch real-time updates:
   - Training progress
   - Client connections
   - Round-by-round aggregation
   - Blockchain logs
   - Anomaly detection

### 4. Push Weights (Hospital View)

1. Login as `hospital_1`
2. Click "Push Model Weights"
3. Watch contribution status
4. See reputation score increase
5. View audit trail on blockchain

---

## Architecture Overview

```
┌─────────────────────┐
│  React Dashboard    │ ← User Interface
│  (localhost:3000)   │
└──────────┬──────────┘
           │ WebSocket
           ↓
┌─────────────────────┐
│  WebSocket Server   │ ← Real-time Updates
│  (localhost:5000)   │
└──────────┬──────────┘
           │ Event Broadcasting
           ↓
┌─────────────────────┐
│  Flower FL Server   │ ← Federated Learning
│  (localhost:8080)   │
└──────────┬──────────┘
           │ gRPC
           ↓
┌─────────────────────┐
│  Hospital Clients   │ ← Local Training
│  (hospital_1, etc)  │
└─────────────────────┘
```

---

## WebSocket Events Reference

### Server → Dashboard

| Event | Data | Purpose |
|-------|------|---------|
| `server_state` | `{state: 'RUNNING'}` | Server status |
| `log_message` | `{level: 'INFO', message: '...'}` | Log entries |
| `client_update` | `{clientId, samples, reputation}` | Client info |
| `round_start` | `{round, totalRounds}` | Round begins |
| `round_update` | `{step, details}` | Progress update |
| `round_complete` | `{round, accuracy}` | Round finished |
| `blockchain_logged` | `{txHash, message}` | Blockchain entry |
| `anomaly_detected` | `{clientId, message}` | Security alert |

### Dashboard → Server

| Event | Data | Purpose |
|-------|------|---------|
| `start_server` | `{}` | Start FL training |
| `push_weights` | `{hospitalId}` | Client contribution |

---

## Troubleshooting

### Frontend Won't Start

**Error: `npm: command not found`**
```bash
# Install Node.js from https://nodejs.org/
# Then retry: npm install
```

**Error: `Module not found`**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### WebSocket Connection Failed

**Error: `Failed to connect to ws://localhost:5000`**

1. Check if WebSocket server is running:
```bash
curl http://localhost:5000/health
```

2. Check firewall settings
3. Verify `.env` file exists with correct URL

### Backend Import Errors

**Error: `ModuleNotFoundError: No module named 'fastapi'`**
```bash
pip install fastapi uvicorn python-socketio websockets
```

### Dashboard Shows "Disconnected"

1. Ensure WebSocket server is running
2. Check browser console for errors (F12)
3. Verify CORS settings in `websocket_server.py`

### Clients Not Showing in Dashboard

1. Verify clients started AFTER server
2. Check FL server logs for client connections
3. Ensure clients are using correct server address

---

## Development Tips

### Hot Reload

- **Frontend**: Automatically reloads on file save
- **Backend**: Restart manually after changes

### Debug Mode

**Frontend:**
```bash
# Browser console (F12)
# WebSocket messages logged automatically
```

**Backend:**
```python
# Add to websocket_server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Custom Events

To add new events:

1. **Add event in backend** (`websocket_server.py`):
```python
await manager.broadcast('my_event', {'data': value})
```

2. **Listen in frontend** (`useWebSocket.js`):
```javascript
socket.on('my_event', (data) => {
  console.log('My event:', data);
  setState(prev => ({ ...prev, myData: data }));
});
```

3. **Use in component**:
```javascript
const { myData } = useWebSocket();
```

---

## File Checklist

Before running, ensure you have:

### Backend
- [x] `backend/websocket_server.py`
- [x] `backend/fl_server_with_dashboard.py`
- [x] `backend/fl_client/client.py` (existing)
- [x] `backend/fl_client/server.py` (existing)
- [x] `backend/model.py` (existing)

### Frontend
- [x] `frontend/package.json`
- [x] `frontend/.env`
- [x] `frontend/tailwind.config.js`
- [x] `frontend/src/index.js`
- [x] `frontend/src/index.css`
- [x] `frontend/src/App.js`
- [x] `frontend/src/App.css`
- [x] `frontend/src/hooks/useWebSocket.js`
- [x] `frontend/src/components/LoginScreen.js`
- [x] `frontend/src/components/AggregatorDashboard.js`
- [x] `frontend/src/components/HospitalDashboard.js`

---

## Next Steps

1. ✅ Complete setup using this guide
2. ✅ Test with 2 hospital clients
3. ✅ Verify dashboard updates in real-time
4. ✅ Test blockchain logging (if configured)
5. ✅ Test anomaly detection
6. 🚀 **Demo to judges!**

---

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Verify all files are in correct locations
3. Check terminal/console logs for errors
4. Ensure all dependencies are installed

---

**Good luck with your demo! 🎉**