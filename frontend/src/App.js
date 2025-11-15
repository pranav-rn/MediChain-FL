import React, { useState } from 'react';
import LoginScreen from './components/LoginScreen';
import AggregatorDashboard from './components/AggregatorDashboard';
import HospitalDashboard from './components/HospitalDashboard';
import './App.css';

function App() {
  const [user, setUser] = useState(null);

  const handleLogin = (role, username) => {
    setUser({ role, username });
  };

  const handleLogout = () => {
    setUser(null);
  };

  if (!user) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  if (user.role === 'admin') {
    return <AggregatorDashboard onLogout={handleLogout} />;
  }

  return <HospitalDashboard username={user.username} onLogout={handleLogout} />;
}

export default App;