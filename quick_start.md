# 🚀 MediChain-FL Quick Start Guide

Get the complete system running in **5 minutes**.

## ⚡ Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## 📦 Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd MediChain-FL

# Run complete setup
chmod +x scripts/complete_setup.sh
./scripts/complete_setup.sh
```

### 2. Copy Component Files

You need to copy these files from the artifacts I provided:

#### Backend Files (NEW):
```bash
# Create these files in backend/
backend/websocket_server.py              # From artifact "websocket-server"
backend/fl_server_with_dashboard.py      # From artifact "integrated-fl-server"
```

#### Frontend Files (ALL NEW):
```bash
# Copy all these files
frontend/package.json                              # From artifact "frontend-package-json"
frontend/.env                                      # From artifact "frontend-env-file"
frontend/src/index.js                              # From artifact "frontend-index-js"
frontend/src/App.js                                # From artifact "frontend-app-js"
frontend/src/App.css                               # From artifact "frontend-app-css"
frontend/src/hooks/useWebSocket.js                 # From artifact "use-websocket-hook"
frontend/src/components/LoginScreen.js             # From artifact "login-screen-component"
frontend/src/components/AggregatorDashboard.js     # From artifact "aggregator-dashboard"
frontend/src/components/HospitalDashboard.js       # From artifact "hospital-dashboard"
```

#### Update index.css:
```css
/* frontend/src/index.css */
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
```

### 3. Install Dependencies

```bash
# Backend
source venv/bin/activate
pip install fastapi uvicorn python-socketio websockets

# Frontend
cd frontend
npm install
cd ..
```

## 🎯 Running the System

### Option 1: Automated (Recommended)

```bash
chmod +x scripts/run_complete_system.sh
./scripts/run_complete_system.sh
```

### Option 2: Manual (4 terminals)

**Terminal 1: Integrated Server**
```bash
source venv/bin/activate
python backend/fl_server_with_dashboard.py --rounds 5 --min-clients 2
```

**Terminal 2: Frontend**
```bash
cd frontend
npm start
```

**Terminal 3: Hospital 1**
```bash
source venv/bin/activate
python backend/fl_client/client.py hospital_1
```

**Terminal 4: Hospital 2**
```bash
source venv/bin/activate
python backend/fl_client/client.py hospital_2
```

## 🌐 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:3000 | Main UI |
| **WebSocket** | ws://localhost:5000/ws | Real-time updates |
| **FL Server** | localhost:8080 | Federated learning |
| **Health Check** | http://localhost:5000/health | Server status |

## 🔐 Login Credentials

### Admin (Aggregator View)
- **Username:** `admin`
- **Password:** `anything`

### Hospital (Client View)
- **Username:** `hospital_1` or `hospital_2`
- **Password:** `anything`

## 🎬 Demo Flow

### 1. Login as Admin
```
http://localhost:3000
→ Username: admin
→ Password: test
→ Click Login
```

### 2. Start Server
```
Dashboard → Click "Start Server"
→ Watch logs populate
→ See clients connect
```

### 3. Watch Training
```
→ Round progress updates
→ Client table fills
→ Accuracy graph grows
→ Blockchain logs appear
```

### 4. Login as Hospital
```
New tab → http://localhost:3000
→ Username: hospital_1
→ Password: test
```

### 5. Push Weights
```
Click "Push Model Weights"
→ Watch reputation increase
→ See audit trail update
```

## 🐛 Troubleshooting

### Dashboard won't connect
```bash
# Check WebSocket server
curl http://localhost:5000/health

# Should return: {"status":"healthy",...}
```

### Frontend errors
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

### Backend import errors
```bash
source venv/bin/activate
pip install fastapi uvicorn python-socketio websockets
```

### Ports already in use
```bash
# Kill existing processes
lsof -ti:3000 | xargs kill -9  # Frontend
lsof -ti:5000 | xargs kill -9  # WebSocket
lsof -ti:8080 | xargs kill -9  # FL Server
```

## 📊 What You'll See

### Admin Dashboard
- ✅ Real-time server status
- ✅ Training progress bar
- ✅ Live round updates (4 steps)
- ✅ Connected hospitals table
- ✅ Terminal-style audit log
- ✅ Accuracy line chart

### Hospital Dashboard
- ✅ Push weights button
- ✅ Reputation score (MCT tokens)
- ✅ Reputation history chart
- ✅ Immutable audit trail
- ✅ Contribution status checklist
- ✅ Recent activity log

## 🎯 Success Criteria

You should see:
1. ✅ Dashboard loads at localhost:3000
2. ✅ WebSocket shows "Connected"
3. ✅ Server state changes to "RUNNING"
4. ✅ 2 hospitals appear in table
5. ✅ Training progresses through 5 rounds
6. ✅ Accuracy increases each round
7. ✅ Logs show blockchain entries
8. ✅ No anomalies detected

## 📁 File Checklist

Before running, verify these files exist:

### Backend (2 NEW files)
- [x] `backend/websocket_server.py`
- [x] `backend/fl_server_with_dashboard.py`

### Frontend (9 NEW files)
- [x] `frontend/package.json`
- [x] `frontend/.env`
- [x] `frontend/src/index.js`
- [x] `frontend/src/App.js`
- [x] `frontend/src/index.css`
- [x] `frontend/src/App.css`
- [x] `frontend/src/hooks/useWebSocket.js`
- [x] `frontend/src/components/LoginScreen.js`
- [x] `frontend/src/components/AggregatorDashboard.js`
- [x] `frontend/src/components/HospitalDashboard.js`

## 🆘 Need Help?

1. **Check logs** in all terminals
2. **Verify ports** are not in use
3. **Check file paths** match structure
4. **See full guide**: `FRONTEND_INTEGRATION_GUIDE.md`

## 🎉 You're Ready!

Once everything is running:
- **Demo to judges** from the dashboard
- **Show real-time updates** as training happens
- **Highlight blockchain logging** in the audit log
- **Demonstrate anomaly detection** (if triggered)

**Good luck! 🚀**