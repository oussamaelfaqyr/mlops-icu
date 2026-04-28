import joblib
from pathlib import Path
import numpy as np

MODEL_PATH = Path("models/icu_model.joblib")
COLS_PATH = Path("models/feature_columns.joblib")


def test_model_loading():
    if not MODEL_PATH.exists():
        return  # Skip if not trained
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_feature_consistency():
    if not COLS_PATH.exists():
        return
    cols = joblib.load(COLS_PATH)
    assert isinstance(cols, list)
    assert "gender" in cols
    assert "anchor_age" in cols


def test_model_prediction_shape():
    if not MODEL_PATH.exists() or not COLS_PATH.exists():
        return
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)

    # Create dummy input
    dummy_input = np.zeros((1, len(cols)))
    pred = model.predict(dummy_input)
    assert len(pred) == 1
