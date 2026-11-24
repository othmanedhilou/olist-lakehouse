"""Entry point used by the Databricks job (see databricks.yml).

    python jobs/run_pipeline.py --layer bronze --batch-id 2026-11-24
    python jobs/run_pipeline.py --layer silver
    python jobs/run_pipeline.py --layer gold
"""
import argparse
import json
import logging
from datetime import datetime, timezone

from pyspark.sql import SparkSession

from olist_lakehouse import bronze, gold, silver
from olist_lakehouse.config import KEYS, LakeConfig
from olist_lakehouse.merge import overwrite, upsert
from olist_lakehouse.quality import SILVER_EXPECTATIONS, enforce

log = logging.getLogger("olist")


def run_bronze(spark, cfg, batch_id):
    counts = bronze.ingest_all(spark, cfg, batch_id)
    log.info("bronze rows: %s", json.dumps(counts))


def run_silver(spark, cfg):
    def read(name):
        return spark.table(cfg.table("bronze", name))

    cleaned = {name: fn(read(name)) for name, fn in silver.CLEANERS.items()}
    cleaned["products"] = silver.clean_products(read("products"), read("category_translation"))

    results = []
    for name, df in cleaned.items():
        if name in SILVER_EXPECTATIONS:
            results += SILVER_EXPECTATIONS[name].run(df)
    warnings = enforce(results)
    for w in warnings:
        log.warning("quality warning %s.%s: %d rows", w.table, w.check, w.failed_rows)

    for name, df in cleaned.items():
        if name == "geolocation":
            overwrite(df, cfg.table("silver", name))
        else:
            upsert(spark, df, cfg.table("silver", name), KEYS[name])


def run_gold(spark, cfg):
    def t(name):
        return spark.table(cfg.table("silver", name))

    geo = t("geolocation")
    overwrite(gold.dim_date(spark), cfg.table("gold", "dim_date"))
    overwrite(gold.dim_customer(t("customers"), geo), cfg.table("gold", "dim_customer"))
    overwrite(gold.dim_seller(t("sellers"), geo), cfg.table("gold", "dim_seller"))
    overwrite(gold.dim_product(t("products")), cfg.table("gold", "dim_product"))
    fact = gold.fact_sales(t("orders"), t("order_items"), t("order_payments"), t("order_reviews"))
    overwrite(fact, cfg.table("gold", "fact_sales"))
    overwrite(gold.kpi_monthly(spark.table(cfg.table("gold", "fact_sales"))), cfg.table("gold", "kpi_monthly"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=["bronze", "silver", "gold", "all"], default="all")
    p.add_argument("--batch-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spark = SparkSession.builder.getOrCreate()
    cfg = LakeConfig.from_env()
    for schema in (cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema):
        spark.sql(f"create schema if not exists {cfg.catalog}.{schema}")

    if args.layer in ("bronze", "all"):
        run_bronze(spark, cfg, args.batch_id)
    if args.layer in ("silver", "all"):
        run_silver(spark, cfg)
    if args.layer in ("gold", "all"):
        run_gold(spark, cfg)


if __name__ == "__main__":
    main()
