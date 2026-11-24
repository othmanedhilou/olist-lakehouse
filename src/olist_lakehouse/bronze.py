from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import SOURCES, LakeConfig


def read_landing_csv(spark: SparkSession, path: str) -> DataFrame:
    # everything is read as string in bronze, typing happens in silver
    return (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .option("inferSchema", False)
        .csv(path)
    )


def add_ingestion_metadata(df: DataFrame, source_name: str, batch_id: str) -> DataFrame:
    return (
        df.withColumn("_source", F.lit(source_name))
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def ingest_source(spark: SparkSession, cfg: LakeConfig, name: str, batch_id: str) -> int:
    path = f"{cfg.landing_path}/{SOURCES[name]}"
    df = add_ingestion_metadata(read_landing_csv(spark, path), name, batch_id)
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(cfg.table("bronze", name))
    )
    return df.count()


def ingest_all(spark: SparkSession, cfg: LakeConfig, batch_id: str) -> dict:
    return {name: ingest_source(spark, cfg, name, batch_id) for name in SOURCES}
