import pandas as pd
import numpy as np
import sys
from pathlib import Path

class DataValidator:
    def __init__(self):
        self.expected_vitals = ['heart_rate', 'spo2', 'resp_rate', 'temp', 'mean_bp']
        self.stats = ['mean', 'std', 'trend']
        
        # Define ranges for "mean" features
        self.ranges = {
            'heart_rate_mean': (20, 250),
            'spo2_mean': (50, 100),
            'resp_rate_mean': (0, 60),
            'temp_mean': (30, 45),
            'mean_bp_mean': (20, 200)
        }

    def validate_schema(self, df):
        """Checks if all required features are present."""
        expected_cols = ['gender', 'anchor_age']
        for v in self.expected_vitals:
            for s in self.stats:
                expected_cols.append(f"{v}_{s}")
        
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"[ERROR] Missing columns: {missing}")
            return False
        return True

    def validate_ranges(self, df):
        """Checks if vital sign means are within physiological ranges.
        Treats 0 as 'missing/placeholder' rather than a physical value for most vitals.
        """
        for feat, (min_val, max_val) in self.ranges.items():
            if feat in df.columns:
                # Actual physical values (excluding our 0.0 placeholder)
                real_values = df[df[feat] != 0][feat]
                out_of_range = real_values[(real_values < min_val) | (real_values > max_val)]
                
                missing_count = len(df[df[feat] == 0])
                if missing_count > 0:
                    print(f"[INFO] {feat}: {missing_count} samples have missing data (0.0)")
                
                if not out_of_range.empty:
                    print(f"[WARNING] {feat}: {len(out_of_range)} genuine outliers detected out of range {min_val}-{max_val}")
        return True

    def check_nulls(self, df, threshold=0.3):
        """Fails if a column has more than 'threshold' null values."""
        null_counts = df.isnull().mean()
        high_nulls = null_counts[null_counts > threshold]
        if not high_nulls.empty:
            print(f"[ERROR] Columns with too many nulls (> {threshold*100}%):\n{high_nulls}")
            return False
        return True

def main():
    DATA_PATH = Path("data/features_final.csv")
    if not DATA_PATH.exists():
        print("[ERROR] No feature file found to validate.")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    validator = DataValidator()
    
    print("--- Starting Data Validation ---")
    
    success = True
    if not validator.validate_schema(df): success = False
    if not validator.validate_ranges(df): success = False
    if not validator.check_nulls(df): success = False
    
    if success:
        print("[SUCCESS] Data Validation Passed!")
    else:
        print("[FAIL] Data Validation Failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
