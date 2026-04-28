import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path


def detect_drift(reference_path, current_path):
    """
    Compares the distribution of features between training (reference)
    and recent inference (current) data.
    """
    if not Path(reference_path).exists() or not Path(current_path).exists():
        print("Reference or Current data missing for drift analysis.")
        return

    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    # We only check vital mean features for simplicity
    vitals = [
        "heart_rate_mean",
        "spo2_mean",
        "resp_rate_mean",
        "temp_mean",
        "mean_bp_mean",
    ]

    drift_report = {}

    print("--- Drift Detection Report ---")
    for v in vitals:
        if v in ref_df.columns and v in cur_df.columns:
            stat, p_value = ks_2samp(ref_df[v].dropna(), cur_df[v].dropna())
            is_drift = p_value < 0.05
            drift_report[v] = {"p_value": p_value, "drift_detected": is_drift}
            status = "[DRIFT]" if is_drift else "[OK]"
            print(f"{v:20} | P-Value: {p_value:.4f} | {status}")

    return drift_report


if __name__ == "__main__":
    # For demo, we compare the training data with a slightly perturbed version of itself
    REF = "data/training_reference.csv"
    CUR = "data/features_final.csv"  # In production, this would be logged API requests

    detect_drift(REF, CUR)
