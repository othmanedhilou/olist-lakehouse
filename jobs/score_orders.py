"""Databricks task: scores orders that are not delivered yet with the champion
model and writes gold.late_delivery_scores for the Power BI report."""
import mlflow
from pyspark.sql import SparkSession

from olist_lakehouse.config import LakeConfig
from olist_lakehouse.ml.features import open_orders
from olist_lakehouse.ml.score import score


def main():
    spark = SparkSession.builder.getOrCreate()
    cfg = LakeConfig.from_env()
    mlflow.set_registry_uri("databricks-uc")

    pending = open_orders(spark.table(cfg.table("gold", "ml_order_features"))).toPandas()
    scores = score(pending, f"models:/{cfg.catalog}.ml.late_delivery@champion")
    (
        spark.createDataFrame(scores)
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(cfg.table("gold", "late_delivery_scores"))
    )


if __name__ == "__main__":
    main()
