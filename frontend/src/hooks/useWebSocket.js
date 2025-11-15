import { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'http://localhost:5000';

export const useWebSocket = () => {
  const [state, setState] = useState({
    serverState: 'OFFLINE',
    currentRound: 0,
    totalRounds: 5,
    step: null,
    logs: [],
    clients: [],
    roundHistory: [],
    connected: false
  });

  const socketRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket server
    socketRef.current = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    const socket = socketRef.current;

    // Connection events
    socket.on('connect', () => {
      console.log('✅ Connected to WebSocket server');
      setState(prev => ({ ...prev, connected: true }));
    });

    socket.on('disconnect', () => {
      console.log('❌ Disconnected from WebSocket server');
      setState(prev => ({ ...prev, connected: false }));
    });

    socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
    });

    // Server state events
    socket.on('server_state', (data) => {
      console.log('📡 Server state:', data);
      setState(prev => ({ 
        ...prev, 
        serverState: data.state 
      }));
    });

    // Log events
    socket.on('log_message', (data) => {
      console.log('📝 Log:', data);
      setState(prev => ({
        ...prev,
        logs: [
          ...prev.logs,
          {
            ...data,
            timestamp: new Date().toLocaleTimeString()
          }
        ].slice(-100) // Keep last 100 logs
      }));
    });

    // Client events
    socket.on('client_update', (data) => {
      console.log('👥 Client update:', data);
      setState(prev => {
        const existingClientIndex = prev.clients.findIndex(
          c => c.clientId === data.clientId
        );

        let newClients;
        if (existingClientIndex >= 0) {
          // Update existing client
          newClients = [...prev.clients];
          newClients[existingClientIndex] = {
            ...newClients[existingClientIndex],
            ...data
          };
        } else {
          // Add new client
          newClients = [...prev.clients, { ...data, status: 'Connected' }];
        }

        return { ...prev, clients: newClients };
      });
    });

    // Round events
    socket.on('round_start', (data) => {
      console.log('🔄 Round start:', data);
      setState(prev => ({
        ...prev,
        currentRound: data.round,
        totalRounds: data.totalRounds,
        step: 'RECEIVING'
      }));
    });

    socket.on('round_update', (data) => {
      console.log('🔄 Round update:', data);
      setState(prev => ({
        ...prev,
        step: data.step,
        logs: [
          ...prev.logs,
          {
            level: 'INFO',
            message: data.details,
            timestamp: new Date().toLocaleTimeString()
          }
        ].slice(-100)
      }));
    });

    socket.on('round_complete', (data) => {
      console.log('✅ Round complete:', data);
      setState(prev => ({
        ...prev,
        roundHistory: [
          ...prev.roundHistory,
          {
            round: data.round,
            accuracy: data.accuracy,
            loss: data.loss
          }
        ],
        step: null
      }));
    });

    // Blockchain events
    socket.on('blockchain_logged', (data) => {
      console.log('🔗 Blockchain logged:', data);
      setState(prev => ({
        ...prev,
        logs: [
          ...prev.logs,
          {
            level: 'INFO',
            message: `Blockchain: ${data.message}`,
            timestamp: new Date().toLocaleTimeString()
          }
        ].slice(-100)
      }));
    });

    // Anomaly events
    socket.on('anomaly_detected', (data) => {
      console.log('⚠️ Anomaly detected:', data);
      setState(prev => ({
        ...prev,
        logs: [
          ...prev.logs,
          {
            level: 'WARNING',
            message: `Anomaly: ${data.message}`,
            timestamp: new Date().toLocaleTimeString()
          }
        ].slice(-100)
      }));
    });

    // Cleanup on unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  // Function to send start server command
  const startServer = () => {
    if (socketRef.current) {
      socketRef.current.emit('start_server');
    }
  };

  // Function to push weights (for hospital clients)
  const pushWeights = (hospitalId) => {
    if (socketRef.current) {
      socketRef.current.emit('push_weights', { hospitalId });
    }
  };

  return {
    ...state,
    startServer,
    pushWeights
  };
};