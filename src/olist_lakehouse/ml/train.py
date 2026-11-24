"""Training and evaluation, tracked with MLflow.

Split is by time: the model is trained on older orders and evaluated on the
most recent ones, which is how it will be used. A random split would mix
future orders into training and give optimistic scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import CATEGORICAL, FEATURES, LABEL, NUMERIC


@dataclass
class TrainResult:
    run_id: str
    model_name: str
    metrics: dict
    baseline_metrics: dict


def time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("purchased_at")
    cut = df["purchased_at"].quantile(1 - test_fraction)
    return df[df["purchased_at"] < cut], df[df["purchased_at"] >= cut]


def _preprocess(scale: bool) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
    ])


def baseline_model() -> Pipeline:
    return Pipeline([
        ("prep", _preprocess(scale=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def xgb_model(pos_weight: float, params: dict | None = None) -> Pipeline:
    p = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    } | (params or {})
    return Pipeline([
        ("prep", _preprocess(scale=False)),
        ("clf", XGBClassifier(scale_pos_weight=pos_weight, **p)),
    ])


def evaluate(y_true: np.ndarray, proba: np.ndarray, top_share: float = 0.1) -> dict:
    """Besides AUC, measures what the business would do with the score:
    look at the 10% riskiest orders and see how many late ones are caught."""
    n_top = max(1, int(len(proba) * top_share))
    top = np.argsort(-proba)[:n_top]
    late_total = max(1, int(y_true.sum()))
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "base_rate": round(float(y_true.mean()), 4),
        "precision_top10": round(float(y_true[top].mean()), 4),
        "recall_top10": round(float(y_true[top].sum() / late_total), 4),
    }


def feature_importance(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["prep"].get_feature_names_out()
    imp = model.named_steps["clf"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train(
    df: pd.DataFrame,
    experiment: str,
    registered_model: str | None = None,
    params: dict | None = None,
) -> TrainResult:
    df = df.dropna(subset=[LABEL])
    train_df, test_df = time_split(df)
    X_train, y_train = train_df[FEATURES], train_df[LABEL].astype(int).to_numpy()
    X_test, y_test = test_df[FEATURES], test_df[LABEL].astype(int).to_numpy()
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="late-delivery") as run:
        mlflow.set_tags({"task": "late_delivery", "split": "time"})
        mlflow.log_params({
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_until": str(train_df["purchased_at"].max()),
            "test_from": str(test_df["purchased_at"].min()),
            "scale_pos_weight": round(pos_weight, 3),
            "features": ",".join(FEATURES),
        })

        base = baseline_model().fit(X_train, y_train)
        base_metrics = evaluate(y_test, base.predict_proba(X_test)[:, 1])
        mlflow.log_metrics({f"baseline_{k}": v for k, v in base_metrics.items()})

        model = xgb_model(pos_weight, params).fit(X_train, y_train)
        mlflow.log_params({f"xgb_{k}": v for k, v in model.named_steps["clf"].get_params().items()
                           if k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight")})
        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)
        mlflow.log_metrics(metrics)

        imp = feature_importance(model)
        mlflow.log_table(imp.head(30), "feature_importance.json")
        test_scores = test_df[["order_id"]].assign(late_probability=proba, is_late=y_test)
        mlflow.log_table(test_scores.head(5000), "test_predictions.json")

        signature = infer_signature(X_test.head(200), proba[:200])
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=registered_model,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        return TrainResult(run.info.run_id, registered_model or "", metrics, base_metrics)


def promote_if_better(registered_model: str, run_id: str, metric: str = "pr_auc", alias: str = "champion") -> bool:
    """Moves the alias to the new version only if it beats the current champion."""
    client = mlflow.MlflowClient()
    new = next(v for v in client.search_model_versions(f"name='{registered_model}'") if v.run_id == run_id)
    new_score = client.get_run(run_id).data.metrics[metric]
    try:
        current = client.get_model_version_by_alias(registered_model, alias)
        current_score = client.get_run(current.run_id).data.metrics.get(metric, -1)
    except Exception:
        current_score = -1
    if new_score > current_score:
        client.set_registered_model_alias(registered_model, alias, new.version)
        return True
    return False
