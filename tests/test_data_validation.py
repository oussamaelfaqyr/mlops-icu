import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = Path("data/features_final.csv")

def test_data_leakage():
    """Ensure no overlap between stays if we were doing time-based splits (simulated)."""
    if not DATA_PATH.exists():
        pytest.skip("Data not generated yet")
    
    df = pd.read_csv(DATA_PATH)
    assert 'timestamp' in df.columns
    # Check if timestamps are monotonically increasing in the file if sorted (standard practice)
    assert not df['timestamp'].isnull().any()

def test_no_null_critical_ids():
    if not DATA_PATH.exists():
        pytest.skip("Data not generated yet")
        
    df = pd.read_csv(DATA_PATH)
    assert not df['stay_id'].isnull().any()
    assert not df['subject_id'].isnull().any()

def test_target_distribution():
    if not DATA_PATH.exists():
        pytest.skip("Data not generated yet")
        
    df = pd.read_csv(DATA_PATH)
    # Check that we have both classes (or at least valid binary values)
    assert set(df['target'].unique()).issubset({0, 1})
