import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
from ..data_generation.config import GlobalConfig

class CTRModel:
    def __init__(self, config: GlobalConfig, feature_names: List[str]):
        self.config = config
        self.feature_names = feature_names
        self.model_type = config.model.type
        self.model = self._init_model()
        self.metadata = {}
        self.best_threshold = 0.1
        self.calibrator = None

    def _init_model(self):
        hp = self.config.active_hyperparameters
        
        # --- MOVE IMPORTS HERE TO AVOID SCOPE ERRORS ---
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        # ---------------------------------------------

        if self.model_type == "xgboost":
            return xgb.XGBClassifier(**hp, use_label_encoder=False, eval_metric='logloss')
        
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(**hp)
            
                # === STACKING ENSEMBLE UPGRADE ===
        elif self.model_type == "stacked_ensemble":
            estimators = [
                ('xgb', xgb.XGBClassifier(
                    n_estimators=800, max_depth=8, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, 
                    use_label_encoder=False, eval_metric='logloss', n_jobs=-1
                )),
                ('lgb', lgb.LGBMClassifier(
                    n_estimators=800, max_depth=8, learning_rate=0.05,
                    num_leaves=127, verbose=-1,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1
                )),
                # OLD: Keep RF as requested, but limit depth to prevent overfitting
                ('rf', RandomForestClassifier(
                    n_estimators=400, max_depth=8, n_jobs=-1
                ))
            ]
            
            # NEW: Use XGBoost as Meta-Learner (Non-Linear Stacking)
            # This allows the model to learn: "If XGB predicts high but LGB predicts low, trust XGB"
            final_estimator = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1, 
                use_label_encoder=False, eval_metric='logloss'
            )
            
            return StackingClassifier(
                estimators=estimators, 
                final_estimator=final_estimator, 
                cv=3, 
                n_jobs=-1,
                passthrough=True # CRITICAL: Feed original features to meta-learner too
            )
        elif self.model_type == "sklearn_rf":
            return RandomForestClassifier(**hp)
            
        else: 
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[List[Tuple]] = None):
        X = X[self.feature_names]
        
        # Handle Imbalance
        neg, pos = (y == 0).sum(), (y == 1).sum()
        scale = neg / pos
        if self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=scale)
            print(f"⚖️ Applying Scale Pos Weight: {scale:.2f}")
        
        # Train Base Model
        if self.model_type in ["xgboost", "lightgbm"] and eval_set:
            print("🚀 Training with Early Stopping...")
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)
            
        self.metadata["train_samples"] = len(X)
        print(f"✅ Model Trained: {self.model_type} on {len(X)} samples.")

        # --- UPGRADED CALIBRATION STEP ---
        if eval_set:
            print("🔧 Calibrating Probabilities (Cross-Validated Isotonic)...")
            from sklearn.calibration import CalibratedClassifierCV
            
            # We use the validation set we already split, but we can do better.
            # Let's use the CalibratedClassifierCV on the trained model using the validation data.
            # This is safer than IsotonicRegression directly as it handles edge cases.
            
            X_val, y_val = eval_set[0]
            
            # Use 'prefit' because our model is already trained.
            # Use 'isotonic' for maximum flexibility (better than sigmoid for complex patterns).
            calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
            calibrated_model.fit(X_val, y_val)
            
            self.calibrator = calibrated_model # Store the whole wrapper
            
            # Report Log Loss
            raw_probs = self.model.predict_proba(X_val)[:, 1]
            cal_probs = calibrated_model.predict_proba(X_val)[:, 1]
            print(f"   Log Loss (Raw): {log_loss(y_val, raw_probs):.4f} -> (Calibrated): {log_loss(y_val, cal_probs):.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        
        # If we have a calibrator, use it (it handles the base model internally)
        if self.calibrator:
            return self.calibrator.predict_proba(X)[:, 1]
        
        # Fallback to raw model
        return self.model.predict_proba(X)[:, 1]

        # REPLACE tune_ensemble_hyperparameters in ctr_model.py
    def tune_ensemble_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        print(f"--- Tuning Base Estimators for Stacked Ensemble (Speed Optimized) ---")
        
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV

        scoring = 'roc_auc'

        # --- MONOTONIC CONSTRAINTS ---
        feature_names = X.columns.tolist()
        monotone_constraints = {}
        target_features = ['bid', 'quality_score', 'log_bid', 'bid_quality_score', 'bid_per_quality']
        for col in feature_names:
            if any(t in col for t in target_features):
                monotone_constraints[col] = 1
        print(f"🔒 Applying Monotonic Constraints to {len(monotone_constraints)} features.")

        # 1. TUNE XGBoost (Standard Booster + Reg)
        # We revert to 'gbtree' for speed, but keep regularization for performance.
        print("Tuning XGBoost...")
        xgb_model = xgb.XGBClassifier(
            booster='gbtree',            # FAST & ROBUST
            use_label_encoder=False, eval_metric='logloss', 
            n_jobs=1,                    # FIX: Set to 1 to let SearchCV handle parallelism
            objective='binary:logistic',
            monotone_constraints=monotone_constraints if monotone_constraints else None,
            random_state=42
        )
        xgb_params = {
            'n_estimators': [500, 800],
            'max_depth': [6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0, 0.1],          # Regularization
            'reg_lambda': [1, 5]        # L2 Regularization
        }
        xgb_search = RandomizedSearchCV(
            xgb_model, xgb_params, n_iter=10, scoring=scoring, cv=3, 
            verbose=1, n_jobs=-1, random_state=42 # Parallelize the FOLDS
        )
        xgb_search.fit(X, y)
        best_xgb = xgb_search.best_estimator_
        print(f"Best XGBoost AUC: {xgb_search.best_score_:.4f}")

        # 2. TUNE LIGHTGBM
        print("Tuning LightGBM...")
        lgb_model = lgb.LGBMClassifier(verbose=-1, n_jobs=1, random_state=42)
        lgb_params = {
            'n_estimators': [500, 800],
            'max_depth': [6, 8],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [63, 127],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        lgb_search = RandomizedSearchCV(
            lgb_model, lgb_params, n_iter=10, scoring=scoring, cv=3, 
            verbose=1, n_jobs=-1, random_state=42
        )
        lgb_search.fit(X, y)
        best_lgb = lgb_search.best_estimator_
        print(f"Best LightGBM AUC: {lgb_search.best_score_:.4f}")

        # 3. PREPARE RANDOM FOREST
        best_rf = RandomForestClassifier(
            n_estimators=400, max_depth=10, n_jobs=-1, random_state=42
        )

        # 4. BUILD ENSEMBLE
        print("Building Robust Ensemble...")
        
        estimators = [
            ('xgb', best_xgb),
            ('lgb', best_lgb),
            ('rf', best_rf)
        ]
        
        meta_learner = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, 
            use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42
        )
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,
            passthrough=True, 
            n_jobs=-1
        )
        
        return {}
        # REPLACE tune_hyperparameters in ctr_model.py
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        print(f"--- Starting Hyperparameter Optimization ---")
        
        # Logic for Stacked Ensemble
        if self.model_type == "stacked_ensemble":
            return self.tune_ensemble_hyperparameters(X, y)
        
        # Logic for Single XGBoost
        if self.model_type == "xgboost":
            param_dist = {
                'n_estimators': [1000, 1500, 2000],
                'max_depth': [6, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1.0, 5.0],
                'min_child_weight': [1, 3, 5]
            }
            from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            search = RandomizedSearchCV(
                self.model, param_dist, 
                n_iter=30,      
                scoring='roc_auc', 
                cv=cv, 
                verbose=1, 
                n_jobs=-1, 
                random_state=42
            )
            search.fit(X, y)
            best_params = search.best_params_
            
            print(f"🏆 Best Params: {best_params}")
            self.model = search.best_estimator_
            return best_params
            
        return {}

    def optimize_threshold_for_profit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, revenue_per_click: float = 1.0, cost_per_impression: float = 0.15) -> float:
        print("--- Optimizing Threshold for MAX PROFIT (Mathematical) ---")
        
        from scipy.optimize import minimize_scalar

        # Define negative profit function (to minimize)
        def neg_profit(threshold):
            preds = (y_pred_proba >= threshold).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            # Profit = Revenue - Cost
            # Revenue = TP * Rev_per_click
            # Cost = (TP + FP) * Cost_per_imp
            profit = (tp * revenue_per_click) - ((tp + fp) * cost_per_impression)
            return -profit # Minimize negative profit

        # Optimize
        res = minimize_scalar(neg_profit, bounds=(0.05, 0.95), method='bounded')
        
        best_threshold = res.x
        best_profit = -res.fun
        
        self.best_threshold = best_threshold
        print(f"✅ Optimal Threshold: {best_threshold:.4f} (Est. Profit: ${best_profit:.2f})")
        return best_threshold

    def save(self, path: str):
        joblib.dump({'model': self, 'threshold': self.best_threshold}, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str) -> 'CTRModel':
        artifact = joblib.load(path)
        return artifact['model']