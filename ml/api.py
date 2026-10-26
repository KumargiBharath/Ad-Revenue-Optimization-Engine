import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
#To Run First activate the environment
#To Run:"uvicorn ml.api:app --reload"
# --- ROBUST PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent 
sys.path.append(str(project_root))
os.chdir(project_root)

from ml.models.ctr_system import CTRSystem
from ml.data_generation.config import load_config

# --- AUTH CONFIG ---
SECRET_KEY = "your_super_secret_key_change_this_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# --- DATABASE SETUP (SQLite) ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DB MODELS ---
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

# --- PYDANTIC MODELS ---
class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    username: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class AuctionInput(BaseModel):
    age_group: str = "25-34"
    device_type: str = "mobile"
    location: str = "US"
    user_interests: str = "tech"
    ad_category: str = "tech"
    bid: float = 2.5
    quality_score: float = 0.8
    position: str = "top"
    hour_of_day: int = 18

# --- AUTH UTILS ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# --- INITIALIZATION ---
app = FastAPI(title="AdAuction AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ctr_system = None

@app.on_event("startup")
def startup_event():
    global ctr_system
    try:
        config = load_config(str(project_root))
        model_path = project_root / "artifacts" / "models" / "ctr_system_v1.pkl"
        ctr_system = CTRSystem.load(str(model_path))
        print("✅ Model loaded successfully for API.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading model: {e}")

# --- AUTH ENDPOINTS ---
@app.post("/auth/signup", response_model=User)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pw = get_password_hash(user.password)
    new_user = UserDB(username=user.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/auth/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- ML ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "API is running", "model_loaded": ctr_system is not None}

@app.post("/predict")
def predict_ctr(input_data: AuctionInput):
    if not ctr_system: raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        data = {
            "auction_id": "api_call", "user_id": "api_user", "ad_id": "api_ad", "advertiser_id": "api_adv",
            "age_group": input_data.age_group, "device_type": input_data.device_type, "location": input_data.location,
            "user_historical_ctr": 0.05, "user_interests": input_data.user_interests, "ad_category": input_data.ad_category,
            "creative_type": "banner", "position": input_data.position, "bid": input_data.bid,
            "quality_score": input_data.quality_score, "ad_historical_ctr": 0.05,
            "hour_of_day": input_data.hour_of_day, "day_of_week": 4, "publisher_category": input_data.ad_category
        }
        df = pd.DataFrame([data])
        prob = ctr_system.predict(df)['click_probability'].values[0]
        return {
            "predicted_ctr": float(prob),
            "threshold": float(ctr_system.model.best_threshold),
            "decision": "BID" if prob >= ctr_system.model.best_threshold else "SKIP"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_sensitivity(input_data: AuctionInput):
    if not ctr_system: raise HTTPException(status_code=500, detail="Model not loaded.")

    def get_pred(overrides={}):
        base_data = {
            "auction_id": "api_call", "user_id": "api_user", "ad_id": "api_ad", "advertiser_id": "api_adv",
            "age_group": input_data.age_group, "device_type": input_data.device_type, "location": input_data.location,
            "user_historical_ctr": 0.05, "user_interests": input_data.user_interests, "ad_category": input_data.ad_category,
            "creative_type": "banner", "position": input_data.position, "bid": float(input_data.bid),
            "quality_score": float(input_data.quality_score), "ad_historical_ctr": 0.05,
            "hour_of_day": int(input_data.hour_of_day), "day_of_week": 4, "publisher_category": input_data.ad_category
        }
        base_data.update(overrides)
        df = pd.DataFrame([base_data])
        return ctr_system.predict(df)['click_probability'].values[0]

    try:
        bid_range = np.linspace(0.5, 10.0, 20)
        bid_data = [{"bid": round(b, 2), "ctr": get_pred({"bid": b})} for b in bid_range]

        quality_range = np.linspace(0.1, 1.0, 10)
        quality_data = [{"quality": round(q, 2), "ctr": get_pred({"quality_score": q})} for q in quality_range]

        hours = range(0, 24)
        hourly_data = [{"hour": h, "ctr": get_pred({"hour_of_day": h})} for h in hours]

        categories = ["sports", "tech", "fashion", "finance", "health", "gaming", "travel"]
        category_data = [{"category": c, "ctr": get_pred({"ad_category": c})} for c in categories]

        devices = ["mobile", "desktop", "tablet"]
        device_data = [{"device": d, "ctr": get_pred({"device_type": d})} for d in devices]

        return {
            "bid_sensitivity": bid_data, "quality_sensitivity": quality_data,
            "hourly_trend": hourly_data, "category_breakdown": category_data,
            "device_comparison": device_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))