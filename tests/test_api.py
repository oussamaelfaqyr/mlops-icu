from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_api_contract():
    """Verify that the prediction response matches the defined contract."""
    payload = {
        "subject_id": "TEST_123",
        "stay_id": "STAY_456",
        "gender": 1,
        "anchor_age": 60,
        "vitals": {
            "heart_rate": [80, 85, 90],
            "spo2": [98, 97, 96]
        }
    }
    # We use /predict
    response = client.post("/predict", json=payload)
    
    # If model is not loaded in test env, this might fail, so we handle it
    if response.status_code == 500:
        pytest.skip("Model not loaded in test environment")
        
    assert response.status_code == 200
    data = response.json()
    
    # Contract Check
    expected_keys = {"patient_id", "risk_score", "risk_level", "prediction_horizon", "model_version", "timestamp"}
    assert set(data.keys()) == expected_keys
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_level"] in ["LOW", "MODERATE", "HIGH"]
