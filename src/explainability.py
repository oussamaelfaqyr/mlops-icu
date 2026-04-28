import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import os


def generate_explanations(model_path, data_path):
    print("Generating SHAP explanations...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    cols_to_drop = ["stay_id", "subject_id", "target", "timestamp"]
    X = df.drop(columns=cols_to_drop)

    # Use SHAP to explain predictions
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Save a summary plot
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("reports/shap_summary.png")
    print("SHAP Summary plot saved to reports/shap_summary.png")

    # Feature importance relative comparison
    # (Optional: can also save individual patient force plots)


if __name__ == "__main__":
    MODEL = "models/icu_model.joblib"
    DATA = "data/features_final.csv"
    if Path(MODEL).exists() and Path(DATA).exists():
        generate_explanations(MODEL, DATA)
