from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

app = FastAPI(title="Clinical ICU Deterioration API", version="1.0.0")

# CORS — allow the dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
MODEL_PATH = "models/icu_model.joblib"
COLS_PATH = "models/feature_columns.joblib"

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLS_PATH)
except:
    model = None
    feature_columns = None

# --- Serve Dashboard Static Files ---
DASHBOARD_DIR = Path("dashboard")
STATIC_DIR = DASHBOARD_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_dashboard():
    """Serve the ICU monitoring dashboard."""
    index_path = DASHBOARD_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Dashboard not found. API is running."}

@app.get("/about")
def serve_about():
    """Serve the project documentation page."""
    about_path = DASHBOARD_DIR / "about.html"
    if about_path.exists():
        return FileResponse(str(about_path))
    return {"message": "Documentation page not found."}

# --- API Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ready",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

class VitalInput(BaseModel):
    subject_id: str
    stay_id: str
    gender: int
    anchor_age: int
    vitals: Dict[str, List[float]]

class PredictionResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    prediction_horizon: str
    model_version: str
    timestamp: str

@app.post("/predict", response_model=PredictionResponse)
def predict(data: VitalInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not found")
    
    features = {'gender': data.gender, 'anchor_age': data.anchor_age}
    vital_names = ['heart_rate', 'spo2', 'resp_rate', 'temp', 'mean_bp']
    
    for v in vital_names:
        values = data.vitals.get(v, [])
        if values:
            series = pd.Series(values)
            features[f'{v}_mean'] = series.mean()
            features[f'{v}_std'] = series.std() if len(series) > 1 else 0
            features[f'{v}_trend'] = np.polyfit(np.arange(len(series)), series.values, 1)[0] if len(series) > 1 else 0
        else:
            features[f'{v}_mean'] = 0
            features[f'{v}_std'] = 0
            features[f'{v}_trend'] = 0

    df_input = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)
    
    risk_score = float(model.predict_proba(df_input)[0, 1])
    
    risk_level = "LOW"
    if risk_score > 0.7: risk_level = "HIGH"
    elif risk_score > 0.4: risk_level = "MODERATE"

    return {
        "patient_id": data.subject_id,
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "prediction_horizon": "6-12h",
        "model_version": "v1.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
