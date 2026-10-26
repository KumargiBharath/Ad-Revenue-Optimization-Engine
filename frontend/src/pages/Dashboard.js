import React, { useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'; // <--- IMPORT useNavigate
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  LineChart, Line
} from 'recharts';
import AuthContext from '../context/AuthContext';

// --- CONFIG ---
const API_URL = "http://localhost:8000";

// --- STATIC DATA FOR ROI ---
const BASELINE_CLICKS = 7674;
const VOLUME_CLICKS = 22957;
const MARGIN_CLICKS = 22040;
const BASELINE_SPEND = 4355.75;
const VOLUME_SPEND = 60817.16;
const MARGIN_SPEND = 51700.12;

function Dashboard() {
  // --- AUTH HOOKS ---
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate(); // <--- INITIALIZE NAVIGATE

  // --- STATE ---
  const [prediction, setPrediction] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [sliderValue, setSliderValue] = useState(3.86);
  
  const [formData, setFormData] = useState({
    age_group: "25-34", device_type: "mobile", location: "US", user_interests: "tech",
    ad_category: "tech", bid: 2.5, quality_score: 0.8, position: "top", hour_of_day: 18
  });

  // --- LOGOUT HANDLER ---
  const handleLogout = () => {
    logout();      // Clears token from local storage
    navigate('/'); // Redirects to Welcome page
  };

  // --- EFFECT: Fetch Analysis Data whenever formData changes ---
  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const response = await axios.post(`${API_URL}/analyze`, formData);
        setAnalysisData(response.data);
      } catch (error) {
        console.error("Analysis Error:", error);
      }
    };

    const timer = setTimeout(() => { fetchAnalysis(); }, 300);
    return () => clearTimeout(timer);
  }, [formData]);

  // --- HANDLERS ---
  const handleChange = (e) => {
    const { name, value } = e.target;
    const parsedValue = name === 'bid' || name === 'quality_score' ? parseFloat(value) : 
                        name === 'hour_of_day' ? parseInt(value) : value;
                        
    setFormData({ ...formData, [name]: parsedValue });
  };

  const handlePredict = async () => {
    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setPrediction(response.data);
    } catch (error) {
      console.error("Predict Error:", error);
      alert("Error connecting to backend.");
    }
  };

  const calculateROI = () => {
    const value = sliderValue;
    return [
      { name: 'Baseline', Spend: BASELINE_SPEND, Profit: (BASELINE_CLICKS * value) - BASELINE_SPEND },
      { name: 'ML Volume', Spend: VOLUME_SPEND, Profit: (VOLUME_CLICKS * value) - VOLUME_SPEND },
      { name: 'ML Margin', Spend: MARGIN_SPEND, Profit: (MARGIN_CLICKS * value) - MARGIN_SPEND },
    ];
  };

  return (
    <div style={{ fontFamily: 'Segoe UI, sans-serif', padding: '20px', backgroundColor: '#f4f6f8', minHeight: '100vh' }}>
      
      {/* HEADER WITH LOGOUT */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', color: '#333' }}>
        <h1>🚀 AdAuction AI Dashboard</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <span style={{ fontWeight: '500' }}>Welcome, <span style={{color: '#4CAF50'}}>{user?.username}</span></span>
            <button 
                onClick={handleLogout} // <--- UPDATED HANDLER
                style={{padding:'8px 20px', cursor:'pointer', background: '#ff7043', color: 'white', border: 'none', borderRadius: '5px', fontWeight: 'bold'}}>
                Logout
            </button>
        </div>
      </header>

      {/* TOP ROW */}
      <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
        
        {/* LEFT: ROI SLIDER & CHART */}
        <div style={{ flex: 1, backgroundColor: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
          <h3>💰 Profit Sensitivity (Value per Click)</h3>
          <label style={{ fontWeight: 'bold' }}>Value Per Click: <span style={{color: '#4CAF50'}}>${sliderValue.toFixed(2)}</span></label>
          <input 
            type="range" min="1" max="10" step="0.1" value={sliderValue} 
            onChange={(e) => setSliderValue(parseFloat(e.target.value))}
            style={{ width: '100%', marginTop: '10px' }}
          />
          <div style={{ height: '200px', marginTop: '20px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={calculateROI()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(v) => `$${v.toFixed(2)}`} />
                <Bar dataKey="Spend" fill="#ff7043" name="Spend" />
                <Bar dataKey="Profit" fill="#66bb6a" name="Profit" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* RIGHT: INPUTS */}
        <div style={{ flex: 1, backgroundColor: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
          <h3>🎛️ Model Inputs</h3>
          <div style={{ display: 'grid', gap: '12px' }}>
            <label>Device: <select name="device_type" value={formData.device_type} onChange={handleChange} style={{width:'100%', padding:'5px'}}>
                <option value="mobile">Mobile</option><option value="desktop">Desktop</option><option value="tablet">Tablet</option>
            </select></label>
            <label>Category: <select name="ad_category" value={formData.ad_category} onChange={handleChange} style={{width:'100%', padding:'5px'}}>
                <option value="sports">Sports</option><option value="tech">Tech</option><option value="fashion">Fashion</option>
                <option value="finance">Finance</option><option value="health">Health</option><option value="gaming">Gaming</option>
            </select></label>
            <label>Bid (${formData.bid.toFixed(1)}): <input type="range" min="0.5" max="10" step="0.1" name="bid" value={formData.bid} onChange={handleChange} /></label>
            <label>Quality ({formData.quality_score.toFixed(1)}): <input type="range" min="0.1" max="1.0" step="0.1" name="quality_score" value={formData.quality_score} onChange={handleChange} /></label>
            <label>Hour ({formData.hour_of_day}): <input type="range" min="0" max="23" step="1" name="hour_of_day" value={formData.hour_of_day} onChange={handleChange} /></label>
          </div>
          <button onClick={handlePredict} style={{width:'100%', marginTop:'15px', padding:'10px', background:'#4CAF50', color:'white', border:'none', borderRadius:'5px', cursor:'pointer', fontWeight:'bold'}}>
            Get Single Prediction
          </button>
          {prediction && (
            <div style={{ marginTop:'10px', padding:'10px', background: prediction.decision === 'BID' ? '#e8f5e9' : '#fff3e0', borderRadius:'5px', borderLeft: `5px solid ${prediction.decision === 'BID' ? '#4CAF50' : '#ff9800'}` }}>
              <strong>Result: {prediction.decision}</strong> <br/>
              CTR: {(prediction.predicted_ctr * 100).toFixed(2)}%
            </div>
          )}
        </div>
      </div>

      {/* BOTTOM ROW: DYNAMIC CHARTS */}
      {analysisData && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px' }}>
          
          <div style={{ background: 'white', padding: '20px', borderRadius: '12px' }}>
            <h4>📈 Bid Impact</h4>
            <div style={{ height: '180px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analysisData.bid_sensitivity}>
                  <XAxis dataKey="bid" unit="$" /><YAxis tickFormatter={(t) => `${(t*100).toFixed(0)}%`} />
                  <Tooltip formatter={(v) => `${(v*100).toFixed(2)}%`} />
                  <Line type="monotone" dataKey="ctr" stroke="#8884d8" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ background: 'white', padding: '20px', borderRadius: '12px' }}>
            <h4>⭐ Quality Impact</h4>
            <div style={{ height: '180px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analysisData.quality_sensitivity}>
                  <XAxis dataKey="quality" /><YAxis tickFormatter={(t) => `${(t*100).toFixed(0)}%`} />
                  <Tooltip formatter={(v) => `${(v*100).toFixed(2)}%`} />
                  <Line type="monotone" dataKey="ctr" stroke="#82ca9d" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ background: 'white', padding: '20px', borderRadius: '12px' }}>
            <h4>⏰ Hourly Trend</h4>
            <div style={{ height: '180px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analysisData.hourly_trend}>
                  <XAxis dataKey="hour" /><YAxis tickFormatter={(t) => `${(t*100).toFixed(0)}%`} />
                  <Tooltip formatter={(v) => `${(v*100).toFixed(2)}%`} />
                  <Line type="monotone" dataKey="ctr" stroke="#ffc658" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ background: 'white', padding: '20px', borderRadius: '12px' }}>
            <h4>🎯 Category Performance</h4>
            <div style={{ height: '180px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analysisData.category_breakdown} layout="vertical">
                  <XAxis type="number" tickFormatter={(t) => `${(t*100).toFixed(0)}%`} />
                  <YAxis type="category" dataKey="category" width={60} />
                  <Tooltip formatter={(v) => `${(v*100).toFixed(2)}%`} />
                  <Bar dataKey="ctr" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
        </div>
      )}
    </div>
  );
}

export default Dashboard;