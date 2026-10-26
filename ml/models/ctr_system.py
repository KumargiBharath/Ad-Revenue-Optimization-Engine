import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from ..validation.schema import DatasetSchema
from ..data_generation.config import GlobalConfig
from .ctr_model import CTRModel
from ..monitoring.drift_detection import DriftMonitor
from ..monitoring.performance_monitor import MetricsSuite

class CTRSystem:
    # ... (Keep __init__ and _feature_engineering exactly as provided in your context) ...
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.schema_version = DatasetSchema.VERSION
        self.schema_fingerprint = DatasetSchema.get_schema_fingerprint()
        
        # Define raw features - EXCLUDING user_historical_ctr (Pure Noise)
        self.raw_schema_features = [
            f for f in DatasetSchema.get_feature_names() 
            if DatasetSchema.FEATURES[f].category not in ['id', 'target'] 
            and f != 'revenue' 
            and f != 'user_interests'
            and f != 'user_historical_ctr' # CRITICAL: Removed distractor
        ]
        
        self.categorical_features = [f for f in self.raw_schema_features if DatasetSchema.FEATURES[f].dtype == 'category']
        self.numerical_features = [f for f in self.raw_schema_features if DatasetSchema.FEATURES[f].dtype != 'category']
        
        # Explicit list of engineered features
        self.numerical_features.extend([
            # Core Features
            'interest_match', 'bid_quality_score', 'user_ad_fit', 
            'bid_per_quality', 'is_mobile_top',
            'is_prime_time', 'publisher_match', 
            
            # Ground Truth Math
            'log_bid', 'golden_interaction', 
            
            # Temporal
            'hour_sin', 'hour_cos', 'mobile_evening_boost',
            
            # Optimization
            'log_bid_x_quality', 'quality_gt_07', 'bid_deviation', 
            'hour_dist_to_peak', 'is_exact_sim_window',
            
            # NEW: Extreme Optimization Features
            'bid_to_quality_ratio', 'quality_x_interest', 'bid_x_interest',
            'prime_time_intensity', 'is_high_value_user', 'log_bid_squared',
        ])
        
        self.model = CTRModel(config, feature_names=[])
        self.drift_monitor = DriftMonitor(threshold=0.15)
        self.encoder = None
        self.is_fitted = False
        self.target_encodings = {}
        self.freq_encodings = {} 
        self.global_mean = 0.05

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # --- 1. BASE FEATURES ---
        def check_match(row):
            if pd.isna(row['ad_category']) or pd.isna(row['user_interests']): return 0
            return 1 if str(row['ad_category']).lower() in str(row['user_interests']).lower().split('|') else 0
        df['interest_match'] = df.apply(check_match, axis=1)
        
        df['bid_quality_score'] = df['bid'] * df['quality_score']
        df['user_ad_fit'] = df['user_historical_ctr'] * df['ad_historical_ctr']
        df['bid_per_quality'] = df['bid'] / (df['quality_score'] + 1e-5)
        df['is_mobile_top'] = ((df['device_type'] == 'mobile') & (df['position'] == 'top')).astype(int)
        
        if 'hour_of_day' in df.columns:
            df['is_prime_time'] = ((df['hour_of_day'] >= 18) & (df['hour_of_day'] <= 23)).astype(int)
        else: df['is_prime_time'] = 0

        if 'publisher_category' in df.columns:
            df['publisher_match'] = (df['publisher_category'].astype(str) == df['ad_category'].astype(str)).astype(int)
        else: df['publisher_match'] = 0

        # --- 2. GROUND TRUTH FEATURES ---
        df['log_bid'] = np.log1p(df['bid'])
        df['golden_interaction'] = (df['interest_match'] & (df['quality_score'] > 0.7)).astype(int)

        # --- 3. TEMPORAL FEATURES ---
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['mobile_evening_boost'] = ((df['device_type'] == 'mobile') & (df['hour_of_day'] >= 15) & (df['hour_of_day'] <= 21)).astype(int)
        else:
            df['hour_sin'] = 0.0; df['hour_cos'] = 0.0; df['mobile_evening_boost'] = 0

        # --- 4. EXTREME OPTIMIZATION FEATURES ---
        df['bid_to_quality_ratio'] = df['bid'] / (df['quality_score'] + 1e-5)
        df['quality_x_interest'] = df['quality_score'] * df['interest_match']
        df['bid_x_interest'] = df['log_bid'] * df['interest_match']
        df['prime_time_intensity'] = df['is_prime_time'] * df['mobile_evening_boost']
        df['is_high_value_user'] = (df['user_historical_ctr'] > 0.15).astype(int)
        df['log_bid_squared'] = np.log1p(df['bid']) ** 2
        
        # --- 5. LEGACY OPTIMIZATION ---
        df['log_bid_x_quality'] = np.log1p(df['bid']) * df['quality_score']
        df['quality_gt_07'] = (df['quality_score'] > 0.7).astype(int)
        df['bid_deviation'] = df['bid'] - 2.75 
        
        if 'hour_of_day' in df.columns:
            df['hour_dist_to_peak'] = np.minimum(np.abs(df['hour_of_day'] - 18), 24 - np.abs(df['hour_of_day'] - 18))
            df['is_exact_sim_window'] = ((df['hour_of_day'] >= 15) & (df['hour_of_day'] <= 21)).astype(int)
        else:
            df['hour_dist_to_peak'] = 6.0; df['is_exact_sim_window'] = 0

        return df

    def _preprocess_for_tuning(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        df = DatasetSchema.enforce_schema(raw_df)
        df = self._feature_engineering(df)
        
        self.global_mean = df[DatasetSchema.TARGET].mean()
        
        from sklearn.model_selection import KFold
        cols_to_encode = ['ad_category', 'location', 'creative_type', 'publisher_category']
        smoothing = 10
        
        for col in cols_to_encode:
            col_name_enc = f'{col}_target_enc'
            
            df[col_name_enc] = np.nan 
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(df):
                fold_stats = df.iloc[train_idx].groupby(col, observed=False)[DatasetSchema.TARGET].agg(['mean', 'count'])
                fold_stats['smoothed'] = (fold_stats['count'] * fold_stats['mean'] + smoothing * self.global_mean) / (fold_stats['count'] + smoothing)
                mapping = fold_stats['smoothed'].to_dict()
                df.loc[df.index[val_idx], col_name_enc] = df.iloc[val_idx][col].astype(str).map(mapping)
            
            df[col_name_enc] = df[col_name_enc].astype(float).fillna(self.global_mean)
            if col_name_enc not in self.numerical_features: self.numerical_features.append(col_name_enc)
            
            freq_map = df[col].value_counts().to_dict()
            self.freq_encodings[col] = freq_map
            col_name_freq = f'{col}_freq'
            df[col_name_freq] = df[col].map(freq_map).astype(float).fillna(0)
            if col_name_freq not in self.numerical_features: self.numerical_features.append(col_name_freq)
            
            global_stats = df.groupby(col, observed=False)[DatasetSchema.TARGET].agg(['mean', 'count'])
            global_stats['smoothed'] = (global_stats['count'] * global_stats['mean'] + smoothing * self.global_mean) / (global_stats['count'] + smoothing)
            self.target_encodings[col] = global_stats['smoothed'].to_dict()

        final_feature_cols = self.categorical_features + self.numerical_features
        
        if not self.is_fitted:
            self.encoder = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features), ('num', 'passthrough', self.numerical_features)], remainder='drop')
            X_encoded = self.encoder.fit_transform(df[final_feature_cols])
            self.model.feature_names = list(self.encoder.get_feature_names_out())
            self.is_fitted = True
        else:
            X_encoded = self.encoder.transform(df[final_feature_cols])
            
        return {"X": pd.DataFrame(X_encoded, columns=self.model.feature_names), "y": df[DatasetSchema.TARGET]}

    def train(self, raw_df: pd.DataFrame, eval_set: List = None):
        df = DatasetSchema.enforce_schema(raw_df)
        df = self._feature_engineering(df)
        
        self.global_mean = df[DatasetSchema.TARGET].mean()
        smoothing = 10
        
        cols_to_encode = ['ad_category', 'location', 'creative_type', 'publisher_category']
        
        for col in cols_to_encode:
            stats = df.groupby(col, observed=False)[DatasetSchema.TARGET].agg(['mean', 'count'])
            stats['smoothed_mean'] = (stats['count'] * stats['mean'] + smoothing * self.global_mean) / (stats['count'] + smoothing)
            self.target_encodings[col] = stats['smoothed_mean'].to_dict()
            
            col_name_enc = f'{col}_target_enc'
            df[col_name_enc] = df[col].astype(str).map(self.target_encodings[col]).fillna(self.global_mean)
            
            freq_map = df[col].value_counts().to_dict()
            self.freq_encodings[col] = freq_map
            col_name_freq = f'{col}_freq'
            df[col_name_freq] = df[col].map(freq_map).astype(float).fillna(0)
        
        final_feature_cols = self.categorical_features + self.numerical_features
        
        if not self.is_fitted:
             self.encoder = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features), ('num', 'passthrough', self.numerical_features)], remainder='drop')
             X_encoded = self.encoder.fit_transform(df[final_feature_cols])
             self.model.feature_names = list(self.encoder.get_feature_names_out())
             self.is_fitted = True
        else:
            X_encoded = self.encoder.transform(df[final_feature_cols])
        
        self.drift_monitor.fit(df[self.numerical_features], self.numerical_features)
        self.model.train(pd.DataFrame(X_encoded, columns=self.model.feature_names), df[DatasetSchema.TARGET], eval_set=eval_set)

    def _preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = DatasetSchema.enforce_schema(raw_df)
        df = self._feature_engineering(df)
        
        cols_to_encode = ['ad_category', 'location', 'creative_type', 'publisher_category']
        
        for col in cols_to_encode:
             col_name_enc = f'{col}_target_enc'
             df[col_name_enc] = df[col].astype(str).map(self.target_encodings[col]).fillna(self.global_mean)
             
             col_name_freq = f'{col}_freq'
             df[col_name_freq] = df[col].map(self.freq_encodings.get(col, {})).astype(float).fillna(0)
             
        return pd.DataFrame(self.encoder.transform(df[self.categorical_features + self.numerical_features]), columns=self.model.feature_names)

    # ==========================================
    # RECTIFIED PREDICT METHOD
    # ==========================================
    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = DatasetSchema.enforce_schema(raw_df)
        df = self._feature_engineering(df)
        
        # --- FIX: Generate Encoded Features BEFORE Drift Check ---
        cols_to_encode = ['ad_category', 'location', 'creative_type', 'publisher_category']
        for col in cols_to_encode:
             # Target Encoding
             col_name_enc = f'{col}_target_enc'
             # Use stored mappings. Fallback to global mean if category is unseen.
             df[col_name_enc] = df[col].astype(str).map(self.target_encodings.get(col, {})).fillna(self.global_mean)
             
             # Frequency Encoding
             col_name_freq = f'{col}_freq'
             df[col_name_freq] = df[col].map(self.freq_encodings.get(col, {})).astype(float).fillna(0)
        
        # Now check_drift will find the columns it expects
        self.drift_monitor.check_drift(df[self.numerical_features])
        
        X = self._preprocess(raw_df)
        output = raw_df[['auction_id']].copy()
        output['click_probability'] = self.model.predict_proba(X)
        return output

    def evaluate(self, raw_df: pd.DataFrame, threshold: float = None):
        df = DatasetSchema.enforce_schema(raw_df)
        X = self._preprocess(df)
        y = df[DatasetSchema.TARGET]
        y_pred = self.model.predict_proba(X)
        return MetricsSuite.evaluate(y, y_pred, threshold if threshold else self.model.best_threshold)

    def save(self, path: str = "ctr_system_v1.pkl"): joblib.dump(self, path)
    
    @staticmethod
    def load(path: str) -> 'CTRSystem':
        system = joblib.load(path)
        if system.schema_fingerprint != DatasetSchema.get_schema_fingerprint(): raise ValueError("Schema Mismatch!")
        return system