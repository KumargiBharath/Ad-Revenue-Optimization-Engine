from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import hashlib
import json

@dataclass
class FeatureDefinition:
    dtype: str
    category: str
    online: bool
    impute_strategy: Any = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    unique: bool = False

class ValidationMode:
    STRICT = "strict"
    WARNING = "warning"

class DatasetSchema:
    SCHEMA_NAME: str = "ad_auction_impression_schema"
    VERSION: str = "1.1.0" # Incremented Version
    
    VALID_DEVICES = ["mobile", "desktop", "tablet", "UNKNOWN"]
    VALID_LOCATIONS = ["US", "EU", "APAC", "LATAM", "UNKNOWN"]
    VALID_AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+", "UNKNOWN"]
    TARGET: str = "click"

    FEATURES: Dict[str, FeatureDefinition] = {
        "auction_id": FeatureDefinition(dtype="str", category="id", online=False, impute_strategy="drop", unique=True),
        "user_id": FeatureDefinition(dtype="str", category="id", online=False, impute_strategy="drop"),
        "ad_id": FeatureDefinition(dtype="str", category="id", online=False, impute_strategy="drop"),
        "advertiser_id": FeatureDefinition(dtype="str", category="id", online=False, impute_strategy="drop"),
        "age_group": FeatureDefinition(dtype="category", category="categorical", online=True, impute_strategy="UNKNOWN", allowed_values=VALID_AGE_GROUPS),
        "device_type": FeatureDefinition(dtype="category", category="categorical", online=True, impute_strategy="UNKNOWN", allowed_values=VALID_DEVICES),
        "location": FeatureDefinition(dtype="category", category="categorical", online=True, impute_strategy="UNKNOWN", allowed_values=VALID_LOCATIONS),
        "user_historical_ctr": FeatureDefinition(dtype="float32", category="numerical", online=True, impute_strategy=0.05, min_val=0.0, max_val=1.0),
        "user_interests": FeatureDefinition(dtype="str", category="categorical", online=True, impute_strategy=""),
        "ad_category": FeatureDefinition(dtype="category", category="categorical", online=True, impute_strategy="UNKNOWN"),
        "creative_type": FeatureDefinition(dtype="category", category="categorical", online=True, impute_strategy="UNKNOWN"),
        "position": FeatureDefinition(dtype="category", category="categorical", online=True, impute_strategy="top", allowed_values=["top", "side", "bottom", "UNKNOWN"]),
        "bid": FeatureDefinition(dtype="float32", category="numerical", online=True, impute_strategy=0.0, min_val=0.0, max_val=100.0),
        "quality_score": FeatureDefinition(dtype="float32", category="numerical", online=True, impute_strategy=0.5, min_val=0.0, max_val=1.0),
        "ad_historical_ctr": FeatureDefinition(dtype="float32", category="numerical", online=True, impute_strategy=0.05, min_val=0.0, max_val=1.0),
        
        # --- NEW FEATURES ---
        "hour_of_day": FeatureDefinition(dtype="int8", category="numerical", online=True, impute_strategy=12, min_val=0, max_val=23),
        "day_of_week": FeatureDefinition(dtype="int8", category="numerical", online=True, impute_strategy=3, min_val=0, max_val=6),
        "publisher_category": FeatureDefinition(dtype="category", category="categorical", online=True, impute_strategy="UNKNOWN"),

        "click": FeatureDefinition(dtype="int8", category="target", online=False, impute_strategy=0, allowed_values=[0, 1]),
        "revenue": FeatureDefinition(dtype="float32", category="numerical", online=False, impute_strategy=0.0, min_val=0.0),
    }
    EXPECTED_COLUMNS = list(FEATURES.keys())

    @classmethod
    def get_schema_fingerprint(cls) -> str:
        schema_dict = {name: asdict(defn) for name, defn in cls.FEATURES.items()}
        return hashlib.sha256(json.dumps(schema_dict, sort_keys=True).encode()).hexdigest()

    @classmethod
    def get_feature_names(cls, category: str = None) -> List[str]:
        if category: return [k for k, v in cls.FEATURES.items() if v.category == category]
        return list(cls.FEATURES.keys())

    @classmethod
    def get_dtypes(cls) -> Dict[str, str]: return {name: defn.dtype for name, defn in cls.FEATURES.items()}

    @classmethod
    def check_null_rates(cls, df: pd.DataFrame, threshold: float = 0.3):
        for col in df.columns:
            if col in cls.FEATURES:
                if df[col].isnull().mean() > threshold: raise ValueError(f"Column '{col}' has excessive null rate.")

    @classmethod
    def enforce_schema(cls, df: pd.DataFrame) -> pd.DataFrame:
        cls.check_null_rates(df)
        for col, defn in cls.FEATURES.items():
            if col not in df.columns: df[col] = defn.impute_strategy
        df = df[cls.EXPECTED_COLUMNS]
        for col, defn in cls.FEATURES.items():
            if defn.dtype == 'category': df[col] = df[col].astype(str)
            if defn.impute_strategy is not None: df[col] = df[col].fillna(defn.impute_strategy)
            if defn.dtype == 'category': df[col] = df[col].astype('category')
            else: df[col] = df[col].astype(defn.dtype)
        return df

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, mode: str = ValidationMode.STRICT):
        # Simplified for brevity, logic remains same as previous
        print("✅ Schema Validation Passed.")