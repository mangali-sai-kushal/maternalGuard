"""
model_training.py
-----------------
Trains a RandomForestClassifier (+ optional XGBoost) on the
Maternal Health Risk dataset from Kaggle.

Dataset CSV columns expected:
  Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate, RiskLevel

Usage:
    python model_training.py --data maternal_health_risk.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from utils import (
    remove_outliers_iqr,
    normalize_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_and_preprocess(csv_path: str):
    """Load CSV, clean, encode, and split."""
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"[DATA] Columns: {list(df.columns)}")

    # ── 1. Handle missing values ─────────────────────────────────────────────
    before = len(df)
    df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True)
    print(f"[CLEAN] Dropped {before - len(df)} rows with nulls")

    # ── 2. Remove outliers (Age, HeartRate, SystolicBP) ──────────────────────
    for col in ["Age", "HeartRate", "SystolicBP"]:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)

    # Hard-range clamp for physiological plausibility
    df = df[df["Age"].between(10, 70)]
    df = df[df["HeartRate"].between(30, 200)]
    df = df[df["SystolicBP"].between(60, 220)]
    df = df[df["DiastolicBP"].between(40, 140)]
    print(f"[CLEAN] {len(df)} rows after outlier removal")

    # ── 3. Encode target ─────────────────────────────────────────────────────
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[TARGET_COLUMN])
    joblib.dump(le, ENCODER_PATH)
    print(f"[ENCODE] Classes: {list(le.classes_)}")

    # ── 4. Features / target split ───────────────────────────────────────────
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    # ── 5. Normalize ─────────────────────────────────────────────────────────
    X_scaled, scaler = normalize_features(X, fit=True)
    joblib.dump(scaler, SCALER_PATH)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y), le


def evaluate_model(name, model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  {name} — Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"  F1-Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy_score(y_test, y_pred)


def train(csv_path: str):
    (X_train, X_test, y_train, y_test), le = load_and_preprocess(csv_path)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_acc = evaluate_model("Random Forest", rf, X_test, y_test, le)

    best_model = rf

    # ── XGBoost (optional) ────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        xgb_acc = evaluate_model("XGBoost", xgb, X_test, y_test, le)

        if xgb_acc > rf_acc:
            best_model = xgb
            print(f"\n[BEST] XGBoost wins ({xgb_acc:.4f} > {rf_acc:.4f})")
        else:
            print(f"\n[BEST] Random Forest wins ({rf_acc:.4f} ≥ {xgb_acc:.4f})")

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"\n[CV] 5-fold F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[SAVE] Model saved → {MODEL_PATH}")
    print(f"[SAVE] Encoder saved → {ENCODER_PATH}")
    print(f"[SAVE] Scaler saved  → {SCALER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="maternal_health_risk.csv", help="Path to dataset CSV")
    args = parser.parse_args()
    train(args.data)
