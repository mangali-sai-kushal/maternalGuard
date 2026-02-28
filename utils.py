"""
utils.py
--------
Shared constants, preprocessing helpers, and SHAP feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET_COLUMN   = "RiskLevel"

MODEL_PATH   = "model.pkl"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH  = "scaler.pkl"

# Human-readable feature names for API responses
FEATURE_LABELS = {
    "Age":         "age",
    "SystolicBP":  "systolicBP",
    "DiastolicBP": "diastolicBP",
    "BS":          "bloodGlucose",
    "BodyTemp":    "bodyTemp",
    "HeartRate":   "heartRate",
}


def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 2.5) -> pd.DataFrame:
    """Remove rows where column value is beyond IQR * factor."""
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - factor * IQR) & (df[column] <= Q3 + factor * IQR)]


def normalize_features(X: np.ndarray, fit: bool = False, scaler: StandardScaler = None):
    """
    Standardize features.
    - fit=True: fit a new scaler and return (X_scaled, scaler)
    - fit=False: transform with provided scaler, return X_scaled
    """
    if fit:
        sc = StandardScaler()
        return sc.fit_transform(X), sc
    return scaler.transform(X)


def compute_shap_importance(model, X_sample: np.ndarray, feature_names: list[str]) -> list[dict]:
    """
    Compute SHAP values for a single prediction row.
    Returns top-3 features sorted by |SHAP| descending.
    """
    try:
        import shap

        # TreeExplainer works for both RF and XGBoost
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For multi-class: shap_values is list[array] per class
        # Use the predicted class's SHAP values
        if isinstance(shap_values, list):
            pred_class  = int(model.predict(X_sample)[0])
            values      = np.abs(shap_values[pred_class][0])
        else:
            values      = np.abs(shap_values[0])

        # Normalize to [0, 1]
        total   = values.sum() or 1.0
        impacts = (values / total).tolist()

        features = [
            {
                "feature": FEATURE_LABELS.get(name, name),
                "impact":  round(imp, 4),
            }
            for name, imp in zip(feature_names, impacts)
        ]
        return sorted(features, key=lambda x: x["impact"], reverse=True)[:3]

    except Exception as e:
        # Fallback: use model's built-in feature_importances_
        importances = getattr(model, "feature_importances_", np.ones(len(feature_names)) / len(feature_names))
        total = importances.sum() or 1.0
        normed = importances / total
        features = [
            {"feature": FEATURE_LABELS.get(n, n), "impact": round(float(v), 4)}
            for n, v in zip(feature_names, normed)
        ]
        return sorted(features, key=lambda x: x["impact"], reverse=True)[:3]
