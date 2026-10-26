from sklearn.metrics import (
    roc_auc_score, log_loss, precision_score, 
    recall_score, f1_score, brier_score_loss
)
import numpy as np

class MetricsSuite:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.1) -> dict:
        """
        Computes comprehensive classification and business metrics.
        UPGRADE: Threshold is now a parameter.
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba),
            "brier_score": brier_score_loss(y_true, y_pred_proba),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics