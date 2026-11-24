from pyspark.sql import DataFrame, SparkSession


def merge_condition(keys: list[str]) -> str:
    return " and ".join(f"t.{k} = s.{k}" for k in keys)


def upsert(spark: SparkSession, df: DataFrame, table: str, keys: list[str]) -> None:
    """Incremental load: new keys are inserted, existing keys updated only when
    the incoming row is more recent."""
    from delta.tables import DeltaTable

    if not spark.catalog.tableExists(table):
        df.write.format("delta").saveAsTable(table)
        return

    target = DeltaTable.forName(spark, table)
    builder = target.alias("t").merge(df.alias("s"), merge_condition(keys))
    if "_ingested_at" in df.columns:
        builder = builder.whenMatchedUpdateAll(condition="s._ingested_at > t._ingested_at")
    else:
        builder = builder.whenMatchedUpdateAll()
    builder.whenNotMatchedInsertAll().execute()


def overwrite(df: DataFrame, table: str) -> None:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
