import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shap 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 2
#To Run:- "python -m ml.pipelines.training_pipeline"
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ml.data_generation.simulator import AdAuctionSimulator
from ml.data_generation.config import load_config
from ml.models.ctr_system import CTRSystem
from ml.validation.schema import DatasetSchema

def main():
    # Define Paths
    DATA_PATH = project_root / "data" / "raw" / "simulated_ad_auction_dataset.csv"
    MODEL_PATH = project_root / "artifacts" / "models" / "ctr_system_v1.pkl"
    
    # Load Config
    config = load_config(str(project_root))

    # ==========================================
    # STEP 1: DATA GENERATION (FORCE FRESH)
    # ==========================================
    # We force regeneration to ensure the CSV has the latest features 
    # (mobile_evening_boost, high_quality_interest, etc.)
    
    # Ensure directory exists
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("--- Step 1: Generating Fresh Dataset ---")
    print("🚀 Running Simulation to create new dataset with updated features...")
    sim = AdAuctionSimulator(str(project_root))
    df = sim.run()
    sim.save(df, str(DATA_PATH))
    print(f"✅ Fresh Dataset saved to {DATA_PATH}")

    # ==========================================
    # STEP 2: STRATIFIED SPLITTING
    # ==========================================
    print("\n--- Step 2: Stratified Splitting ---")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[DatasetSchema.TARGET])
    print(f"✅ Stratified Split Complete. Train CTR: {train_df[DatasetSchema.TARGET].mean():.4f}")
    
    # ==========================================
    # STEP 3: INITIALIZE SYSTEM
    # ==========================================
    system = CTRSystem(config)
    
    # ==========================================
    # STEP 4: PREPROCESSING FOR TUNING
    # ==========================================
    print("\n--- Step 3: Preprocessing for Optimization ---")
    train_processed = system._preprocess_for_tuning(train_df)
    X_train_full = train_processed['X']
    y_train_full = train_processed['y']

    # ==========================================
    # STEP 5: HYPERPARAMETER OPTIMIZATION
    # ==========================================
    print("\n--- Step 4: Hyperparameter Optimization ---")
    # This now actually works for Stacked Ensemble
    system.model.tune_hyperparameters(X_train_full, y_train_full)
    
    # Split for Early Stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
    # ==========================================
    # STEP 6: TRAINING (WITH CALIBRATION)
    # ==========================================
    print("\n--- Step 5: Training with Early Stopping & Calibration ---")
    system.model.train(X_tr, y_tr, eval_set=[(X_val, y_val)])

    # ==========================================
    # STEP 7: BUSINESS THRESHOLD OPTIMIZATION
    # ==========================================
    print("\n--- Step 6: Business Threshold Optimization ---")
    val_probs = system.model.predict_proba(X_val)
    optimal_threshold = system.model.optimize_threshold_for_profit(
        y_val.to_numpy(), val_probs, revenue_per_click=1.0, cost_per_impression=0.15
    )

    # ==========================================
    # STEP 8: SAVE SYSTEM
    # ==========================================
    (project_root / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    system.save(str(MODEL_PATH))

    # ==========================================
    # STEP 9: EVALUATION
    # ==========================================
    print("\n--- Step 7: Model Evaluation (Test Set) ---")
    loaded_system = CTRSystem.load(str(MODEL_PATH))
    metrics = loaded_system.evaluate(test_df, threshold=optimal_threshold)
    
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"Log Loss:  {metrics['log_loss']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")

    # ==========================================
    # STEP 10: SHAP EXPLAINABILITY
    # ==========================================
    print("\n--- Step 8: SHAP Explainability ---")
    try:
        xgb_model = None
        
        # logic to extract the model
        if loaded_system.model.model_type == "stacked_ensemble":
            # Access the XGBoost model inside the stack
            xgb_model = loaded_system.model.model.named_estimators_['xgb']
        else:
            xgb_model = loaded_system.model.model
            
        # Prepare test sample
        test_processed = loaded_system._preprocess_for_tuning(test_df)
        X_test_sample = test_processed['X'].iloc[:100] 

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_sample)
        
        plt.title("Feature Importance (SHAP Values)")
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar")
        plt.show()
        print("✅ SHAP Analysis Complete.")
        
    except Exception as e:
        print(f"⚠️ SHAP Analysis Failed: {e}")


if __name__ == "__main__":
    main()