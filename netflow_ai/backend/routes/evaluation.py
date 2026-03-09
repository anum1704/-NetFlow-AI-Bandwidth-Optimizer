"""
routes/evaluation.py — Sections 4 & 5: Model Testing + Metrics

POST /api/evaluation/run           Evaluate model (optional file_id)
GET  /api/evaluation/results       Get last evaluation results
POST /api/evaluation/upload-test   Upload test PCAP and evaluate immediately
GET  /api/evaluation/feature-importance  Feature importance (tree models)
"""

import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import joblib
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
)

from utils.state import uploaded_files, model_artifact, evaluation_results
from utils.pcap_parser import FEATURE_COLUMNS, _synthetic_fallback, parse_pcap

router = APIRouter()
UPLOAD_DIR = "uploads"


# ── Schemas ───────────────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    accuracy:         float
    precision:        float
    recall:           float
    f1_score:         float
    classes:          List[str]
    confusion_matrix: List[List[int]]
    report:           str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model():
    if model_artifact["is_trained"]:
        return (model_artifact["model"], model_artifact["scaler"],
                model_artifact["label_encoder"], model_artifact["feature_names"])
    path = os.path.join("saved_models", "netflow_model.joblib")
    if os.path.exists(path):
        art = joblib.load(path)
        return art["model"], art["scaler"], art["label_encoder"], art["feature_names"]
    raise HTTPException(400, "No trained model. Train first.")


def _evaluate(df, model, scaler, le, feature_names):
    import pandas as pd
    feature_cols = [c for c in feature_names if c in df.columns]
    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    if "label" not in df.columns:
        df = df.copy()
        df["label"] = np.select(
            [df["bytes_per_second"] > 5e7, df.get("flag_rst", 0) == 1],
            ["Peak Spike", "Anomaly"], default="Stable"
        )

    known = set(le.classes_)
    y_str = np.array([l if l in known else le.classes_[0] for l in df["label"].values])
    y_true = le.transform(y_str)
    y_pred = model.predict(X_scaled)

    classes = list(le.classes_)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=classes)

    return {
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":        round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score":         round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "classes":          classes,
        "confusion_matrix": cm,
        "report":           report,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=EvaluationResult)
def run_evaluation(file_id: Optional[str] = None):
    model, scaler, le, feature_names = _load_model()
    df = uploaded_files[file_id]["dataframe"].copy() if file_id and file_id in uploaded_files \
         else _synthetic_fallback(500)
    results = _evaluate(df, model, scaler, le, feature_names)
    evaluation_results.update(results)
    return EvaluationResult(**results)


@router.get("/results", response_model=EvaluationResult)
def get_results():
    if evaluation_results["precision"] is None:
        raise HTTPException(404, "No evaluation run yet")
    return EvaluationResult(**evaluation_results)


@router.post("/upload-test")
async def upload_test_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pcap", ".pcapng", ".cap"}:
        raise HTTPException(400, "Must be a .pcap/.pcapng file")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"test_{file.filename}")
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    df = parse_pcap(dest)
    file_id = f"test_{file.filename}"
    uploaded_files[file_id] = {
        "id": file_id, "filename": file.filename, "path": dest,
        "size_bytes": len(content), "size_mb": round(len(content) / (1024 * 1024), 2),
        "uploaded_at": datetime.utcnow().isoformat(),
        "flow_count": len(df), "feature_count": len(FEATURE_COLUMNS), "dataframe": df,
    }

    model, scaler, le, feature_names = _load_model()
    results = _evaluate(df, model, scaler, le, feature_names)
    evaluation_results.update(results)
    return {"file_id": file_id, "flow_count": len(df), "metrics": EvaluationResult(**results)}


@router.get("/feature-importance")
def feature_importance():
    """Return per-feature importance (tree-based models only)."""
    model, scaler, le, feature_names = _load_model()
    if not hasattr(model, "feature_importances_"):
        raise HTTPException(400, "Feature importance not available for this model type")
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances.tolist()),
                    key=lambda x: x[1], reverse=True)
    return {"features": [{"name": n, "importance": round(v, 5)} for n, v in ranked]}
