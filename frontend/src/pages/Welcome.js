import React from 'react';
import { useNavigate } from 'react-router-dom';

const Welcome = () => {
  const navigate = useNavigate();

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={{ marginBottom: '10px' }}>🚀 AdAuction AI</h1>
        <p style={{ color: '#666', marginBottom: '40px' }}>
          Please choose an option to continue
        </p>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', width: '100%' }}>
          <button 
            style={styles.primaryBtn} 
            onClick={() => navigate('/login')}
          >
            Sign In
          </button>
          <button 
            style={styles.secondaryBtn} 
            onClick={() => navigate('/signup')}
          >
            Sign Up
          </button>
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: { 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '100vh', 
    background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' 
  },
  card: { 
    background: 'white', 
    padding: '60px', 
    borderRadius: '16px', 
    boxShadow: '0 10px 25px rgba(0,0,0,0.1)', 
    textAlign: 'center', 
    width: '350px' 
  },
  primaryBtn: { 
    width: '100%', 
    padding: '12px', 
    background: '#4CAF50', 
    color: 'white', 
    border: 'none', 
    borderRadius: '5px', 
    fontSize: '16px', 
    cursor: 'pointer', 
    fontWeight: 'bold',
    transition: 'background 0.2s'
  },
  secondaryBtn: { 
    width: '100%', 
    padding: '12px', 
    background: 'white', 
    color: '#4CAF50', 
    border: '2px solid #4CAF50', 
    borderRadius: '5px', 
    fontSize: '16px', 
    cursor: 'pointer', 
    fontWeight: 'bold',
    transition: 'background 0.2s'
  }
};

export default Welcome;