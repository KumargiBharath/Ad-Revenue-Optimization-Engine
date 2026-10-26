import React, { createContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if token exists in local storage on startup
    const storedToken = localStorage.getItem('token');
    if (storedToken) {
      setToken(storedToken);
      // Optionally: Verify token with backend here
      // For simplicity, we assume the token is valid if it exists
      setUser({ username: localStorage.getItem('username') }); 
    }
    setLoading(false);
  }, []);

  const login = async (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const res = await axios.post('http://localhost:8000/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    
    localStorage.setItem('token', res.data.access_token);
    localStorage.setItem('username', username);
    setToken(res.data.access_token);
    setUser({ username });
    return res.data;
  };

  const signup = async (username, password) => {
    await axios.post('http://localhost:8000/auth/signup', { username, password });
    // Automatically log in after signup
    return login(username, password);
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, signup, logout, loading }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export default AuthContext;