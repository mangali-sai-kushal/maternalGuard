# Maternal-Guard ML Microservice

FastAPI + scikit-learn risk prediction service.

## Setup

```bash
pip install -r requirements.txt

# Download dataset from Kaggle:
# https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data
# Save as: maternal_health_risk.csv

# Train model (saves model.pkl, label_encoder.pkl, scaler.pkl)
python model_training.py --data maternal_health_risk.csv

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoint

**POST /predict**
```json
{
  "age": 25,
  "systolicBP": 120,
  "diastolicBP": 80,
  "bloodGlucose": 7.5,
  "bodyTemp": 98.6,
  "heartRate": 72
}
```

**Response:**
```json
{
  "riskLevel": "low",
  "probabilityScores": { "high risk": 0.05, "low risk": 0.82, "mid risk": 0.13 },
  "featureImportance": [
    { "feature": "systolicBP",   "impact": 0.38 },
    { "feature": "bloodGlucose", "impact": 0.29 },
    { "feature": "age",          "impact": 0.18 }
  ]
}
```

## Model Details

- **Algorithm**: RandomForestClassifier (200 trees), compared with XGBoost
- **Best model** auto-selected by test accuracy
- **Explainability**: SHAP TreeExplainer (falls back to feature_importances_)
- **Preprocessing**: IQR outlier removal, StandardScaler normalization
- **CV**: 5-fold cross-validation reported during training
