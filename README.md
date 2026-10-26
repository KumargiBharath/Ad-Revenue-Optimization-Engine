# 🚀 Ad Revenue Optimization Engine (AROE)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![RL](https://img.shields.io/badge/RL-Stable--Baselines3-red.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A full-stack, intelligent real-time system designed to automate the pricing, allocation, and optimization of digital ad inventory. AROE combines advanced Machine Learning, Reinforcement Learning, and an interactive React dashboard to maximize total revenue yield (eCPM/RPM) for publishers while preserving a premium user experience

---

## 📖 Executive Summary

Digital publishers face the constant challenge of maximizing revenue from ad inventory while preserving user experience. Decision-making regarding which ads to show, where to show them, and at what price is often static or fragmented.

**AROE** solves this by providing a unified, predictive mechanism that determines the true market value of an impression before it is sold, dynamically routing every ad impression to the highest-paying demand source in real-time.

---

## 🎯 The Problem & Solution

### The Problem ("As-Is" State)

- **Inventory Leakage**: Unsold ad slots due to poor demand source connections.
- **Sub-optimal Floor Prices**: Static price floors lead to either unsold inventory or lost revenue.
- **Latency**: Sequential "waterfall" ad loading degrades page load times and user experience.
- **Fragmented Demand**: Inability to evaluate competing bids simultaneously.

### The Solution ("To-Be" State)

- **Maximize Yield**: Real-time routing to the highest-paying demand source.
- **Optimize Fill Rate**: Intelligent dynamic floor pricing to reduce unsold inventory to near zero.
- **Minimize Latency**: Server-side decisioning to replace sequential waterfall loading.
- **UX Preservation**: Built-in frequency capping, diversity management, and ad quality filters.

---

## ✨ Key Features

### 🧠 Backend (Machine Learning & Optimization)

- **High-Performance Auction Simulator**: Vectorized Pandas/NumPy simulation of users, advertisers, and second-price auctions with non-linear ground-truth click models.
- **Advanced CTR Prediction**: Stacked Ensemble models (XGBoost + LightGBM + Random Forest) with Isotonic Probability Calibration and business-threshold optimization for maximum profit.
- **Reinforcement Learning Agents**: Gymnasium environment with Deep Q-Network (DQN) agents for strategic bid shading and ROI-optimized bidding.
- **Dynamic Auction Optimization**: Features dynamic reserve pricing, budget pacing, frequency capping, and diversity penalties to prevent market monopolies.
- **Data Drift & Schema Validation**: Robust `DatasetSchema` enforcement and `DriftMonitor` to detect feature distribution shifts in real-time.

### 💻 Frontend (Interactive Dashboard)

- **Secure Authentication**: JWT-based login/signup flow with protected routing.
- **Real-Time Prediction Engine**: Interactive form to input ad parameters and get instant "BID" or "SKIP" decisions with predicted CTR.
- **Dynamic Data Visualization**: Beautiful, responsive charts (Recharts) showing Bid Impact, Quality Impact, Hourly Trends, and Category Performance.
- **Profit Sensitivity Analyzer**: Interactive slider to simulate Return on Investment (ROI) across Baseline, ML Volume, and ML Margin strategies.

---

## 🛠️ Tech Stack

| Category                    | Technologies                                      |
| :-------------------------- | :------------------------------------------------ |
| **Backend Core**            | Python 3.12, FastAPI, Uvicorn, Gunicorn, Pydantic |
| **Machine Learning**        | Scikit-Learn, XGBoost, LightGBM, SHAP, Joblib     |
| **Reinforcement Learning**  | Gymnasium, Stable-Baselines3 (DQN)                |
| **Data Processing**         | Pandas, NumPy, SciPy, PyYAML                      |
| **Frontend**                | React 19, React Router DOM, Axios, Recharts       |
| **Database**                | SQLAlchemy, SQLite                                |
| **Validation & Monitoring** | Great Expectations, Evidently, Loguru             |
| **Testing**                 | Pytest, React Testing Library, Jest               |

---

## 📂 Project Structure

```text
AROE/
├── backend/                      # Python ML & API Backend
│   ├── configs/                  # YAML configurations (simulation & model settings)
│   ├── data/                     # Generated simulated datasets
│   ├── ml/
│   │   ├── api/                  # FastAPI application & endpoints
│   │   ├── data_generation/      # Simulator, Entities, Click Model, Budget Manager
│   │   ├── models/               # CTRModel, CTRSystem, Drift Monitor, Metrics
│   │   ├── optimization/         # Auction Optimizer, RL Environment, Strategic Agents
│   │   ├── pipelines/            # Training, Business Optimization, RL Training scripts
│   │   └── validation/           # DatasetSchema & data validation logic
│   ├── artifacts/models/         # Saved .pkl models and .zip RL agents
│   └── requirements.txt          # Python dependencies
│
└── frontend/                     # React Dashboard
    ├── public/
    ├── src/
    │   ├── context/              # AuthContext for JWT management
    │   ├── pages/                # Dashboard, Login, Signup, Welcome
    │   ├── App.js                # React Router configuration
    │   └── index.js              # React entry point
    └── package.json              # Node.js dependencies
```


# 🚀 Getting Started

## 1. Prerequisites

Python 3.12+

Node.js 18+ & npm

## 2. Backend Installation

## Navigate to backend directory (or project root if structured flat)
cd backend

## Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt


## 3. Frontend Installation

## Open a new terminal and navigate to the frontend directory
cd frontend

## Install Node dependencies
npm install

## ⚡ Usage & Running the Application

## Step 1: Generate Synthetic Data & Train Models (Backend)
Run the ML pipelines in the following order to generate data, train the stacked ensemble, and save the artifacts:

## 1. Generate synthetic auction data
python -m ml.data_generation.simulator

## 2. Train the CTR Model (includes hyperparameter tuning & calibration)
python -m ml.pipelines.training_pipeline

## 3. (Optional) Train the RL Bidding Agent
python -m ml.pipelines.train_rl_agent

## 4. Run Business Optimization Simulation to evaluate strategies
python -m ml.pipelines.business_optimization_pipeline

## Step 2: Start the Backend API Server


## Ensure your virtual environment is activated
uvicorn ml.api:app --reload --port 8000

The API will be available at http://localhost:8000. Interactive docs at http://localhost:8000/docs.

## Step 3: Start the Frontend Dashboard

## In the frontend directory
npm start

The React application will open automatically at http://localhost:3000.Method,Endpoint,Description
POST,/auth/signup,Register a new user
POST,/auth/login,Obtain JWT access token
POST,/predict,Get real-time CTR prediction and BID/SKIP decision
POST,/analyze,"Run multi-dimensional sensitivity analysis (bid, quality, time, device)"

## 🔌 API Endpoints

| Method     |	Endpoint	    | Description                                                               |
| :-----     | :------------- | :------------------------------------------------------------------------ |
| **POST**	 |  /auth/signup	| Register a new user                                                       |
| **POST**	 |  /auth/login	  | Obtain JWT access token                                                   |
| **POST**	 |  /predict	    | Get real-time CTR prediction and BID/SKIP decision                        |
| **POST**	 |  /analyze	    | Run multi-dimensional sensitivity analysis (bid, quality, time, device)   |

## 📊 Frontend Dashboard Features

Profit Sensitivity Analyzer: Adjust the "Value Per Click" slider to instantly visualize how different ML strategies (Baseline vs. ML Volume vs. ML Margin) impact overall profit.
Model Input Simulator: Tweak parameters like Device, Category, Bid, Quality Score, and Hour of Day to see how they affect the model's output.
Dynamic Charts:
📈 Bid Impact: Line chart showing CTR response to bid changes.
⭐ Quality Impact: Line chart showing CTR response to quality score changes.
⏰ Hourly Trend: Line chart tracking predicted CTR across 24 hours.
🎯 Category Performance: Horizontal bar chart comparing CTR across different ad categories.

## 📈 Success Metrics (KPIs)

The engine is designed to optimize and report on the following key performance indicators:
eCPM / RPM: Target a 15-20% increase in revenue per 1,000 impressions.
Fill Rate: Maintain >95% of ad requests successfully filled.
Viewability Rate: Ensure >70% of served ads are actually viewable.
Decision Latency: Achieve model inference and routing in <80ms.

## ⚠️ Key Challenges Addressed

Real-Time Latency: Optimized with vectorized NumPy operations and pre-computed feature lookups.
Cold Start Problem: Mitigated via synthetic data generation and fallback global mean encodings.
Sparse Data: Handled through advanced stacking ensembles and monotonic constraints to prevent overfitting.
Privacy Compliance: Designed to operate on contextual and aggregated features, aligning with cookie-less tracking trends.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request

## 📄 License
Distributed under the MIT License. See LICENSE for more information.
Built with ❤️ for the future of programmatic advertising.

