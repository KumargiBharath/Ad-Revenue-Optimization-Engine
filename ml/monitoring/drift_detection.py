import pandas as pd
import numpy as np
from typing import Dict, List

class DriftMonitor:
    """
    Tracks feature distributions and detects data drift.
    """
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.training_stats: Dict[str, Dict] = {}

    def fit(self, df: pd.DataFrame, numerical_features: List[str]):
        """Calculates and stores stats from training data."""
        for col in numerical_features:
            if col in df.columns:
                self.training_stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
        print(f"✅ DriftMonitor fitted on {len(self.training_stats)} features.")

    def check_drift(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compares new data against stored stats.
        Returns a report of drift scores.
        """
        report = {}
        alerts = []

        for col, stats in self.training_stats.items():
            if col not in df.columns:
                alerts.append(f"Missing column: {col}")
                continue

            new_mean = df[col].mean()
            old_mean = stats["mean"]
            
            if old_mean != 0:
                drift_score = abs(new_mean - old_mean) / abs(old_mean)
            else:
                drift_score = abs(new_mean - old_mean)

            report[col] = drift_score

            if drift_score > self.threshold:
                alerts.append(f"Drift Alert: '{col}' shifted by {drift_score:.2%}")

        if alerts:
            print(f"⚠️ DRIFT WARNINGS:\n" + "\n".join(alerts))
        
        return report