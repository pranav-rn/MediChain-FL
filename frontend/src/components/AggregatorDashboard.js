import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Shield, Users, Activity, LogOut, CheckCircle, Clock } from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';

const StepIndicator = ({ step, title, active, completed }) => (
  <div className={`flex items-center space-x-3 p-3 rounded-lg transition-all ${
    active ? 'bg-blue-50 border-2 border-blue-500' : completed ? 'bg-green-50' : 'bg-gray-50'
  }`}>
    {completed ? (
      <CheckCircle className="w-5 h-5 text-green-600" />
    ) : active ? (
      <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
    ) : (
      <Clock className="w-5 h-5 text-gray-400" />
    )}
    <span className={`font-medium ${active ? 'text-blue-700' : completed ? 'text-green-700' : 'text-gray-500'}`}>
      {title}
    </span>
  </div>
);

const AggregatorDashboard = ({ onLogout }) => {
  const wsState = useWebSocket();
  const [serverStarting, setServerStarting] = useState(false);
  const logEndRef = useRef(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [wsState.logs]);

  useEffect(() => {
    if (wsState.serverState === 'RUNNING') {
      setServerStarting(false);
    }
  }, [wsState.serverState]);

  const handleStartServer = () => {
    setServerStarting(true);
    wsState.startServer();
  };

  const isRunning = wsState.serverState === 'RUNNING';
  const progress = wsState.totalRounds > 0 ? (wsState.currentRound / wsState.totalRounds) * 100 : 0;

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Shield className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Aggregator Dashboard</h1>
              <div className="flex items-center space-x-2 mt-1">
                <div className={`w-2 h-2 rounded-full ${
                  wsState.connected && isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {wsState.connected ? (isRunning ? 'Server Running' : 'Server Offline') : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
          <button
            onClick={onLogout}
            className="flex items-center space-x-2 px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <LogOut className="w-4 h-4" />
            <span>Logout</span>
          </button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-4 space-y-4">
        {/* Training Progress */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Training Progress</h2>
          <div className="space-y-3">
            <div className="flex justify-between text-sm text-gray-600">
              <span>Current Round: {wsState.currentRound} of {wsState.totalRounds}</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Control Panel</h2>
          <button
            onClick={handleStartServer}
            disabled={isRunning || serverStarting || !wsState.connected}
            className={`w-full py-3 rounded-lg font-semibold transition-colors ${
              isRunning || serverStarting || !wsState.connected
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {!wsState.connected ? (
              'Connecting to Server...'
            ) : serverStarting ? (
              <span className="flex items-center justify-center space-x-2">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Starting...</span>
              </span>
            ) : isRunning ? (
              'Server Running'
            ) : (
              'Start Server'
            )}
          </button>
        </div>

        {/* Live Round Status */}
        {isRunning && wsState.currentRound > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              Round {wsState.currentRound} Status
            </h2>
            <div className="space-y-2">
              <StepIndicator
                step="RECEIVING"
                title="Receiving Weights"
                active={wsState.step === 'RECEIVING'}
                completed={wsState.step !== 'RECEIVING' && wsState.step !== null}
              />
              <StepIndicator
                step="AGGREGATING"
                title="Homomorphic Aggregation"
                active={wsState.step === 'AGGREGATING'}
                completed={['BLOCKCHAIN', 'ANOMALY'].includes(wsState.step)}
              />
              <StepIndicator
                step="BLOCKCHAIN"
                title="Blockchain Audit"
                active={wsState.step === 'BLOCKCHAIN'}
                completed={wsState.step === 'ANOMALY'}
              />
              <StepIndicator
                step="ANOMALY"
                title="Anomaly Detection"
                active={wsState.step === 'ANOMALY'}
                completed={false}
              />
            </div>
          </div>
        )}

        {/* Connected Hospitals */}
        {isRunning && wsState.clients.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center space-x-2">
              <Users className="w-5 h-5" />
              <span>Connected Hospitals</span>
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 px-3 text-sm font-semibold text-gray-700">Hospital ID</th>
                    <th className="text-left py-2 px-3 text-sm font-semibold text-gray-700">Status</th>
                    <th className="text-left py-2 px-3 text-sm font-semibold text-gray-700">Samples</th>
                    <th className="text-left py-2 px-3 text-sm font-semibold text-gray-700">Reputation</th>
                  </tr>
                </thead>
                <tbody>
                  {wsState.clients.map((client, idx) => (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="py-3 px-3 font-mono text-sm">{client.clientId}</td>
                      <td className="py-3 px-3">
                        <span className="inline-flex items-center space-x-1 px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                          <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                          <span>{client.status}</span>
                        </span>
                      </td>
                      <td className="py-3 px-3 text-sm">{client.samples}</td>
                      <td className="py-3 px-3">
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-[100px]">
                            <div
                              className="bg-green-500 h-2 rounded-full"
                              style={{ width: `${Math.min((client.reputation / 300) * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium">{client.reputation}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* System Audit Log */}
        {isRunning && wsState.logs.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center space-x-2">
              <Activity className="w-5 h-5" />
              <span>System Audit Log</span>
            </h2>
            <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
              {wsState.logs.map((log, idx) => (
                <div key={idx} className="mb-1">
                  <span className="text-gray-500">[{log.timestamp}]</span>
                  <span className={`ml-2 ${
                    log.level === 'ERROR' ? 'text-red-400' :
                    log.level === 'WARNING' ? 'text-yellow-400' :
                    'text-green-400'
                  }`}>
                    {log.level}
                  </span>
                  <span className="text-gray-300 ml-2">{log.message}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* Training History Chart */}
        {wsState.roundHistory.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Training Accuracy</h2>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={wsState.roundHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="accuracy" stroke="#4A90E2" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default AggregatorDashboard;