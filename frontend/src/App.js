import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import Welcome from './pages/Welcome'; // Import the new Welcome page
//To Run: 1.cd frontend 2.npm start
// Helper component to handle routes that should only be visible when LOGGED OUT
const PublicRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  if (token) {
    // If user is logged in, send them to dashboard (don't let them see login/signup/welcome)
    return <Navigate to="/dashboard" />;
  }
  return children;
};

// Helper component to handle routes that should only be visible when LOGGED IN
const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  if (!token) {
    // If user is NOT logged in, send them to Welcome page
    return <Navigate to="/" />;
  }
  return children;
};

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Root Route - Welcome Page */}
          <Route 
            path="/" 
            element={
              <PublicRoute>
                <Welcome />
              </PublicRoute>
            } 
          />

          {/* Auth Routes */}
          <Route 
            path="/login" 
            element={
              <PublicRoute>
                <Login />
              </PublicRoute>
            } 
          />
          <Route 
            path="/signup" 
            element={
              <PublicRoute>
                <Signup />
              </PublicRoute>
            } 
          />

          {/* Protected Dashboard */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } 
          />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;