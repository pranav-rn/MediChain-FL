"""
WebSocket Server for MediChain-FL Dashboard
Broadcasts federated learning events to connected clients
"""

import asyncio
import json
from typing import Dict, Set
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

app = FastAPI(title="MediChain-FL WebSocket Server")

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: Set[WebSocket] = set()

# Global state
server_state = {
    'state': 'OFFLINE',
    'current_round': 0,
    'total_rounds': 5,
    'step': None,
    'clients': [],
    'logs': []
}


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"✅ Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"❌ Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, event_type: str, data: Dict):
        """Broadcast message to all connected clients"""
        message = json.dumps({
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            'type': 'initial_state',
            'data': server_state
        }))
        
        # Listen for client messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client commands
            if message.get('type') == 'start_server':
                await handle_start_server()
            elif message.get('type') == 'push_weights':
                await handle_push_weights(message.get('data', {}))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def handle_start_server():
    """Handle server start command"""
    global server_state
    
    # Update state
    server_state['state'] = 'STARTING'
    await manager.broadcast('server_state', {'state': 'STARTING'})
    
    # Emit log
    await emit_log('INFO', 'Initializing Flower server...')
    
    # Simulate startup sequence
    await asyncio.sleep(1)
    await emit_log('INFO', 'Loading model architecture...')
    
    await asyncio.sleep(1)
    await emit_log('INFO', 'Connected to blockchain at 0x2279B7A0...')
    
    await asyncio.sleep(1)
    server_state['state'] = 'RUNNING'
    await manager.broadcast('server_state', {'state': 'RUNNING'})
    await emit_log('INFO', 'Server is now running and waiting for clients')


async def handle_push_weights(data: Dict):
    """Handle weight push from hospital client"""
    hospital_id = data.get('hospitalId', 'unknown')
    
    await emit_log('INFO', f'{hospital_id} is pushing encrypted weights...')
    await asyncio.sleep(2)
    await emit_log('INFO', f'{hospital_id} weights received and verified')


async def emit_log(level: str, message: str):
    """Emit a log message"""
    await manager.broadcast('log_message', {
        'level': level,
        'message': message
    })


# Public API for FL server to call
class DashboardBroadcaster:
    """
    Public API for the Flower server to broadcast events
    Should be imported and used in fl_client/server.py
    """
    
    @staticmethod
    async def emit_server_state(state: str):
        """Broadcast server state change"""
        global server_state
        server_state['state'] = state
        await manager.broadcast('server_state', {'state': state})
    
    @staticmethod
    async def emit_log(level: str, message: str):
        """Broadcast log message"""
        await manager.broadcast('log_message', {
            'level': level,
            'message': message
        })
    
    @staticmethod
    async def emit_client_update(client_id: str, samples: int, reputation: int, status: str = 'Connected'):
        """Broadcast client update"""
        await manager.broadcast('client_update', {
            'clientId': client_id,
            'samples': samples,
            'reputation': reputation,
            'status': status
        })
    
    @staticmethod
    async def emit_round_start(round_num: int, total_rounds: int):
        """Broadcast round start"""
        global server_state
        server_state['current_round'] = round_num
        server_state['total_rounds'] = total_rounds
        await manager.broadcast('round_start', {
            'round': round_num,
            'totalRounds': total_rounds
        })
    
    @staticmethod
    async def emit_round_update(step: str, details: str):
        """Broadcast round progress update"""
        global server_state
        server_state['step'] = step
        await manager.broadcast('round_update', {
            'step': step,
            'details': details
        })
    
    @staticmethod
    async def emit_round_complete(round_num: int, accuracy: float, loss: float = None):
        """Broadcast round completion"""
        global server_state
        server_state['step'] = None
        await manager.broadcast('round_complete', {
            'round': round_num,
            'accuracy': accuracy,
            'loss': loss
        })
    
    @staticmethod
    async def emit_blockchain_logged(tx_hash: str, message: str):
        """Broadcast blockchain logging event"""
        await manager.broadcast('blockchain_logged', {
            'txHash': tx_hash,
            'message': message
        })
    
    @staticmethod
    async def emit_anomaly_detected(client_id: str, message: str):
        """Broadcast anomaly detection"""
        await manager.broadcast('anomaly_detected', {
            'clientId': client_id,
            'message': message
        })


# Singleton instance for import
broadcaster = DashboardBroadcaster()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "connections": len(manager.active_connections),
        "server_state": server_state['state']
    }


@app.get("/api/state")
async def get_state():
    """Get current server state"""
    return server_state


def run_server(host: str = "0.0.0.0", port: int = 5000):
    """Run the WebSocket server"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting MediChain-FL WebSocket Server")
    print(f"{'='*70}")
    print(f"   WebSocket URL: ws://{host}:{port}/ws")
    print(f"   Health Check: http://{host}:{port}/health")
    print(f"   Waiting for frontend connections...")
    print(f"{'='*70}\n")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_server_in_background(host: str = "0.0.0.0", port: int = 5000):
    """Run server in background thread"""
    thread = threading.Thread(target=run_server, args=(host, port), daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    run_server()