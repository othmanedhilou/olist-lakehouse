"""Trains the late delivery model on a laptop.

Same silver code and feature code as on Databricks; MLflow tracking and model
registry go to a local SQLite file.

    python jobs/train_local.py --input data/sample
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import json
import logging
from pathlib import Path

import mlflow
from pyspark.sql import SparkSession

from olist_lakehouse import bronze, silver
from olist_lakehouse.config import SOURCES
from olist_lakehouse.ml.features import build_features, open_orders, training_set
from olist_lakehouse.ml.score import score
from olist_lakehouse.ml.train import promote_if_better, train

MODEL_NAME = "olist_late_delivery"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/sample")
    p.add_argument("--output", default="data/gold")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    spark = (
        SparkSession.builder.master("local[*]").appName("olist-ml")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    raw = {
        name: bronze.add_ingestion_metadata(bronze.read_landing_csv(spark, str((Path(a.input) / f).resolve())), name, "local")
        for name, f in SOURCES.items()
    }
    s = {name: fn(raw[name]) for name, fn in silver.CLEANERS.items()}
    s["products"] = silver.clean_products(raw["products"], raw["category_translation"])

    feats = build_features(s["orders"], s["order_items"], s["order_payments"], s["products"],
                           s["customers"], s["sellers"], s["geolocation"]).cache()
    train_pdf = training_set(feats).toPandas()
    open_pdf = open_orders(feats).toPandas()
    spark.stop()

    mlflow.set_tracking_uri(a.tracking_uri)
    result = train(train_pdf, experiment="olist-late-delivery", registered_model=MODEL_NAME)
    promoted = promote_if_better(MODEL_NAME, result.run_id)
    print(json.dumps({"run_id": result.run_id, "promoted": promoted,
                      "xgboost": result.metrics, "baseline_logreg": result.baseline_metrics}, indent=2))

    scores = score(open_pdf, f"models:/{MODEL_NAME}@champion")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / "late_delivery_scores.parquet", index=False)
    print(f"{len(scores)} open orders scored -> {out / 'late_delivery_scores.parquet'}")
    print(scores.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
