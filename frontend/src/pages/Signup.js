import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthContext from '../context/AuthContext';

const Signup = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { signup } = useContext(AuthContext);
  const navigate = useNavigate();

    const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await signup(username, password);
      navigate('/dashboard'); // <--- CHANGE THIS
    } catch (err) {
      setError('Error creating account');
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.formBox}>
        <h2>Sign Up</h2>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <form onSubmit={handleSubmit}>
          <input 
            style={styles.input}
            placeholder="Username" 
            value={username} 
            onChange={(e) => setUsername(e.target.value)} 
          />
          <input 
            style={styles.input}
            type="password" 
            placeholder="Password" 
            value={password} 
            onChange={(e) => setPassword(e.target.value)} 
          />
          <button style={styles.button} type="submit">Create Account</button>
        </form>
        <p style={{ marginTop: '10px' }}>
          Already have an account? <a href="/login">Login</a>
        </p>
      </div>
    </div>
  );
};

// Reuse styles from Login
const styles = {
  container: { display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#f4f6f8' },
  formBox: { background: 'white', padding: '40px', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', width: '300px' },
  input: { width: '100%', padding: '10px', marginBottom: '10px', boxSizing: 'border-box' },
  button: { width: '100%', padding: '10px', background: '#2196F3', color: 'white', border: 'none', cursor: 'pointer' }
};

export default Signup;