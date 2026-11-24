"""Runs the full pipeline on a laptop without Delta or a metastore.

Reads the raw Olist CSV files, applies the same silver / quality / gold code as
the Databricks job and writes the gold tables as Parquet so they can be opened
in Power BI Desktop.

    python jobs/run_local.py --input data/raw --output data/gold
"""
import argparse
import logging
from pathlib import Path

from pyspark.sql import SparkSession

from olist_lakehouse import bronze, gold, silver
from olist_lakehouse.config import SOURCES
from olist_lakehouse.quality import SILVER_EXPECTATIONS, enforce

log = logging.getLogger("olist.local")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw")
    p.add_argument("--output", default="data/gold")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("olist-local")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    raw = {}
    for name, file in SOURCES.items():
        path = Path(args.input) / file
        raw[name] = bronze.add_ingestion_metadata(
            bronze.read_landing_csv(spark, str(path.resolve())), name, "local"
        )

    s = {name: fn(raw[name]) for name, fn in silver.CLEANERS.items()}
    s["products"] = silver.clean_products(raw["products"], raw["category_translation"])
    for df in s.values():
        df.cache()

    results = []
    for name, df in s.items():
        if name in SILVER_EXPECTATIONS:
            results += SILVER_EXPECTATIONS[name].run(df)
    for r in results:
        log.info("%-16s %-28s failed=%d", r.table, r.check, r.failed_rows)
    enforce(results)

    fact = gold.fact_sales(s["orders"], s["order_items"], s["order_payments"], s["order_reviews"]).cache()
    tables = {
        "dim_date": gold.dim_date(spark),
        "dim_customer": gold.dim_customer(s["customers"], s["geolocation"]),
        "dim_seller": gold.dim_seller(s["sellers"], s["geolocation"]),
        "dim_product": gold.dim_product(s["products"]),
        "fact_sales": fact,
        "kpi_monthly": gold.kpi_monthly(fact),
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        pdf = df.toPandas()
        pdf.to_parquet(out / f"{name}.parquet", index=False)
        log.info("wrote %s (%d rows)", name, len(pdf))

    spark.stop()


if __name__ == "__main__":
    main()
