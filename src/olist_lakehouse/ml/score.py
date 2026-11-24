from __future__ import annotations

from datetime import datetime, timezone

import mlflow
import pandas as pd

from .features import FEATURES


def risk_band(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"


def score(df: pd.DataFrame, model_uri: str) -> pd.DataFrame:
    model = mlflow.sklearn.load_model(model_uri)
    proba = model.predict_proba(df[FEATURES])[:, 1]
    out = df[["order_id", "purchased_at", "customer_state", "seller_state", "promised_days"]].copy()
    out["late_probability"] = proba.round(4)
    out["risk_band"] = [risk_band(p) for p in proba]
    out["model_uri"] = model_uri
    out["scored_at"] = datetime.now(timezone.utc)
    return out.sort_values("late_probability", ascending=False).reset_index(drop=True)
