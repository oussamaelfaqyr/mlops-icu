import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from model import get_model
from pathlib import Path
import joblib
import os

def train_pipeline(data_path):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Temporal Split: Use earlier data for training, later for testing
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    cols_to_drop = ['stay_id', 'subject_id', 'target', 'timestamp']
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df['target']
    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df['target']
    
    mlflow.set_experiment("ICU_Deterioration_v2")
    
    with mlflow.start_run() as run:
        params = {
            'n_estimators': 150,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'eval_metric': 'auc',
            'random_state': 42  # For reproducibility
        }
        
        mlflow.log_params(params)
        
        model = get_model(params)
        model.fit(X_train, y_train)
        
        # Eval
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        metrics = {
            "auc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0)
        }
        
        print(f"Test Metrics: {metrics}")
        mlflow.log_metrics(metrics)
        
        # Log and Register Model
        mlflow.xgboost.log_model(
            model, 
            artifact_path="model",
            registered_model_name="ICU_Deterioration_XGB"
        )
        
        # Save locally for deployment
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/icu_model.joblib")
        joblib.dump(X_train.columns.tolist(), "models/feature_columns.joblib")
        
        # Save training distribution for monitoring
        train_df.to_csv("data/training_reference.csv", index=False)

if __name__ == "__main__":
    DATA_PATH = Path("data/features_final.csv")
    if DATA_PATH.exists():
        train_pipeline(DATA_PATH)
    else:
        print("Run features.py first.")
