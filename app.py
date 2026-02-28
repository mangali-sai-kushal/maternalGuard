"""
app.py
------
Maternal Health Risk Prediction — FastAPI microservice.

Startup: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from utils import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    normalize_features,
    compute_shap_importance,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("maternal-guard-ml")

# ─── Global model state (loaded once at startup) ──────────────────────────────
_model   = None
_encoder = None
_scaler  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artefacts once at startup; clean up on shutdown."""
    global _model, _encoder, _scaler
    try:
        _model   = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)
        _scaler  = joblib.load(SCALER_PATH)
        logger.info(f"Model loaded from '{MODEL_PATH}'")
        logger.info(f"Classes: {list(_encoder.classes_)}")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}. Run model_training.py first.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Maternal-Guard ML Service",
    description="Maternal health risk prediction powered by Random Forest + SHAP",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schema ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:          float = Field(..., ge=10,  le=70,  description="Patient age in years")
    systolicBP:   float = Field(..., ge=60,  le=220, description="Systolic blood pressure (mmHg)")
    diastolicBP:  float = Field(..., ge=40,  le=140, description="Diastolic blood pressure (mmHg)")
    bloodGlucose: float = Field(..., ge=6.0, le=25.0,description="Blood sugar (mmol/L) — Kaggle dataset scale")
    bodyTemp:     float = Field(..., ge=95,  le=105, description="Body temperature (°F)")
    heartRate:    float = Field(..., ge=30,  le=200, description="Heart rate (bpm)")

    # Accept alternative field names from the Express backend
    @field_validator("*", mode="before")
    @classmethod
    def coerce_number(cls, v):
        if isinstance(v, str):
            return float(v)
        return v


class FeatureImportance(BaseModel):
    feature: str
    impact:  float


class PredictResponse(BaseModel):
    riskLevel:         str
    probabilityScores: dict[str, float]
    featureImportance: list[FeatureImportance]


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":   "ok",
        "modelLoaded": _model is not None,
        "classes":  list(_encoder.classes_) if _encoder else [],
    }


@app.post("/predict", response_model=PredictResponse, summary="Predict maternal health risk")
def predict(req: PredictRequest):
    """
    Accepts physiological measurements and returns:
    - **riskLevel**: "Low" | "Mid" | "High"
    - **probabilityScores**: probability per class
    - **featureImportance**: top-3 SHAP-based feature impacts
    """
    if _model is None or _encoder is None or _scaler is None:
        raise HTTPException(503, "Model not loaded. Run model_training.py first.")

    # Build feature vector in the exact column order used during training
    raw = np.array([[
        req.age,
        req.systolicBP,
        req.diastolicBP,
        req.bloodGlucose,
        req.bodyTemp,
        req.heartRate,
    ]])

    # Normalize using saved scaler
    X = normalize_features(raw, fit=False, scaler=_scaler)

    # Predict
    pred_class   = int(_model.predict(X)[0])
    pred_proba   = _model.predict_proba(X)[0]
    risk_label   = _encoder.inverse_transform([pred_class])[0]

    # Probability scores per class
    prob_scores = {
        str(_encoder.inverse_transform([i])[0]): round(float(p), 4)
        for i, p in enumerate(pred_proba)
    }

    # SHAP feature importance
    fi = compute_shap_importance(_model, X, FEATURE_COLUMNS)

    logger.info(f"Prediction: {risk_label} | proba={prob_scores}")

    return PredictResponse(
        riskLevel=risk_label,
        probabilityScores=prob_scores,
        featureImportance=[FeatureImportance(**f) for f in fi],
    )


# ─── Dev entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
