"""Databricks task: build features from silver, train, register in Unity Catalog.

The new version gets the `champion` alias only if its PR AUC on the most
recent orders beats the current champion.
"""
import json
import logging

import mlflow
from pyspark.sql import SparkSession

from olist_lakehouse.config import LakeConfig
from olist_lakehouse.ml.features import build_features, training_set
from olist_lakehouse.ml.train import promote_if_better, train


def main():
    logging.basicConfig(level=logging.INFO)
    spark = SparkSession.builder.getOrCreate()
    cfg = LakeConfig.from_env()

    def t(name):
        return spark.table(cfg.table("silver", name))

    feats = build_features(t("orders"), t("order_items"), t("order_payments"), t("products"),
                           t("customers"), t("sellers"), t("geolocation"))
    feats.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        cfg.table("gold", "ml_order_features")
    )

    mlflow.set_registry_uri("databricks-uc")
    model_name = f"{cfg.catalog}.ml.late_delivery"
    spark.sql(f"create schema if not exists {cfg.catalog}.ml")
    result = train(
        training_set(spark.table(cfg.table("gold", "ml_order_features"))).toPandas(),
        experiment="/Shared/olist-late-delivery",
        registered_model=model_name,
    )
    promoted = promote_if_better(model_name, result.run_id)
    print(json.dumps({"run_id": result.run_id, "promoted": promoted, "metrics": result.metrics}))


if __name__ == "__main__":
    main()
