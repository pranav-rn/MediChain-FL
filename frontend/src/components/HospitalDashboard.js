import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, LogOut, CheckCircle, Clock } from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';

const HospitalDashboard = ({ username, onLogout }) => {
  const wsState = useWebSocket();
  const [reputation, setReputation] = useState(220);
  const [pushing, setPushing] = useState(false);
  const [auditTrail, setAuditTrail] = useState([
    { block: 3, action: 'Model Update', reward: '+10 MCT', time: '2 mins ago' },
    { block: 2, action: 'Model Update', reward: '+10 MCT', time: '5 mins ago' }
  ]);

  // Update reputation from WebSocket
  useEffect(() => {
    const myClient = wsState.clients.find(c => c.clientId === username);
    if (myClient && myClient.reputation) {
      setReputation(myClient.reputation);
    }
  }, [wsState.clients, username]);

  const handlePushWeights = () => {
    setPushing(true);
    wsState.pushWeights(username);
    
    // Simulate pushing process
    setTimeout(() => {
      setPushing(false);
      setAuditTrail(prev => [{
        block: prev[0].block + 1,
        action: 'Model Update',
        reward: '+10 MCT',
        time: 'Just now'
      }, ...prev]);
    }, 3000);
  };

  const reputationHistory = [
    { round: 1, score: 200 },
    { round: 2, score: 210 },
    { round: 3, score: reputation }
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Activity className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-800">{username} Dashboard</h1>
              <div className="flex items-center space-x-2 mt-1">
                <div className={`w-2 h-2 rounded-full ${
                  wsState.connected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {wsState.connected ? 'Connected to Aggregator' : 'Disconnected'}
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
        {/* Control Panel */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <button
            onClick={handlePushWeights}
            disabled={pushing || !wsState.connected}
            className={`w-full py-4 rounded-lg font-semibold text-lg transition-colors ${
              pushing || !wsState.connected
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {!wsState.connected ? (
              'Connecting...'
            ) : pushing ? (
              <span className="flex items-center justify-center space-x-2">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Pushing Encrypted Weights...</span>
              </span>
            ) : (
              'Push Model Weights'
            )}
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {/* Reputation Score */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Reputation Score</h2>
            <div className="text-center mb-4">
              <div className="text-5xl font-bold text-blue-600 mb-2">{reputation}</div>
              <div className="text-sm text-gray-600">MCT Tokens Earned</div>
            </div>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={reputationHistory}>
                <XAxis dataKey="round" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="score" stroke="#50E3C2" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Audit Trail */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Immutable Audit Trail</h2>
            <div className="space-y-3 max-h-[250px] overflow-y-auto">
              {auditTrail.map((entry, idx) => (
                <div key={idx} className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg border border-green-200">
                  <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                  <div className="flex-1">
                    <div className="font-semibold text-gray-800">Block #{entry.block}</div>
                    <div className="text-sm text-gray-600">{entry.action}</div>
                    <div className="text-xs text-gray-500 mt-1">{entry.time}</div>
                  </div>
                  <div className="text-green-600 font-semibold">{entry.reward}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Contribution Status */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">My Contribution Status</h2>
          <div className="space-y-2">
            <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="font-medium text-green-700">Local training complete</span>
            </div>
            <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="font-medium text-green-700">Gradients encrypted with CKKS</span>
            </div>
            <div className={`flex items-center space-x-3 p-3 rounded-lg ${
              pushing ? 'bg-blue-50 border-2 border-blue-500' : 'bg-gray-50'
            }`}>
              {pushing ? (
                <>
                  <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  <span className="font-medium text-blue-700">Pushing to aggregator...</span>
                </>
              ) : (
                <>
                  <Clock className="w-5 h-5 text-gray-400" />
                  <span className="font-medium text-gray-500">Ready to push</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Recent Logs */}
        {wsState.logs.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Recent Activity</h2>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {wsState.logs.slice(-5).reverse().map((log, idx) => (
                <div key={idx} className="flex items-start space-x-2 p-2 bg-gray-50 rounded text-sm">
                  <span className="text-gray-500 text-xs">{log.timestamp}</span>
                  <span className="text-gray-700">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HospitalDashboard;