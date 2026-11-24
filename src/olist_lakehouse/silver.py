from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import KEYS

TS = "yyyy-MM-dd HH:mm:ss"


def _ts(col: str):
    return F.to_timestamp(F.col(col), TS)


def _latest_per_key(df: DataFrame, keys: list[str]) -> DataFrame:
    # bronze is append-only, so the same key can show up in several batches
    order_col = F.col("_ingested_at").desc() if "_ingested_at" in df.columns else F.lit(1)
    w = Window.partitionBy(*keys).orderBy(order_col)
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def _drop_meta(df: DataFrame) -> DataFrame:
    return df.drop(*[c for c in df.columns if c.startswith("_") and c != "_ingested_at"])


def clean_orders(df: DataFrame) -> DataFrame:
    out = df.select(
        F.trim("order_id").alias("order_id"),
        F.trim("customer_id").alias("customer_id"),
        F.lower(F.trim("order_status")).alias("order_status"),
        _ts("order_purchase_timestamp").alias("purchased_at"),
        _ts("order_approved_at").alias("approved_at"),
        _ts("order_delivered_carrier_date").alias("delivered_carrier_at"),
        _ts("order_delivered_customer_date").alias("delivered_customer_at"),
        _ts("order_estimated_delivery_date").alias("estimated_delivery_at"),
        *[c for c in df.columns if c == "_ingested_at"],
    )
    out = out.withColumn(
        "delivery_days",
        F.datediff("delivered_customer_at", "purchased_at"),
    ).withColumn(
        "is_late",
        F.when(F.col("delivered_customer_at").isNull(), None).otherwise(
            F.col("delivered_customer_at") > F.col("estimated_delivery_at")
        ),
    )
    return _latest_per_key(out, KEYS["orders"])


def clean_order_items(df: DataFrame) -> DataFrame:
    out = df.select(
        F.trim("order_id").alias("order_id"),
        F.col("order_item_id").cast("int").alias("order_item_id"),
        F.trim("product_id").alias("product_id"),
        F.trim("seller_id").alias("seller_id"),
        _ts("shipping_limit_date").alias("shipping_limit_at"),
        F.col("price").cast("decimal(12,2)").alias("price"),
        F.col("freight_value").cast("decimal(12,2)").alias("freight_value"),
        *[c for c in df.columns if c == "_ingested_at"],
    )
    return _latest_per_key(out, KEYS["order_items"])


def clean_order_payments(df: DataFrame) -> DataFrame:
    out = df.select(
        F.trim("order_id").alias("order_id"),
        F.col("payment_sequential").cast("int").alias("payment_sequential"),
        F.lower(F.trim("payment_type")).alias("payment_type"),
        F.col("payment_installments").cast("int").alias("payment_installments"),
        F.col("payment_value").cast("decimal(12,2)").alias("payment_value"),
        *[c for c in df.columns if c == "_ingested_at"],
    )
    return _latest_per_key(out, KEYS["order_payments"])


def clean_order_reviews(df: DataFrame) -> DataFrame:
    out = df.select(
        F.trim("review_id").alias("review_id"),
        F.trim("order_id").alias("order_id"),
        F.col("review_score").cast("int").alias("review_score"),
        F.col("review_comment_message").alias("comment"),
        _ts("review_creation_date").alias("created_at"),
        _ts("review_answer_timestamp").alias("answered_at"),
        *[c for c in df.columns if c == "_ingested_at"],
    )
    return _latest_per_key(out, KEYS["order_reviews"])


def clean_customers(df: DataFrame) -> DataFrame:
    out = df.select(
        F.trim("customer_id").alias("customer_id"),
        F.trim("customer_unique_id").alias("customer_unique_id"),
        F.lpad(F.trim("customer_zip_code_prefix"), 5, "0").alias("zip_code_prefix"),
        F.initcap(F.trim("customer_city")).alias("city"),
        F.upper(F.trim("customer_state")).alias("state"),
        *[c for c in df.columns if c == "_ingested_at"],
    )
    return _latest_per_key(out, KEYS["customers"])


def clean_sellers(df: DataFrame) -> DataFrame:
    out = df.select(
        F.trim("seller_id").alias("seller_id"),
        F.lpad(F.trim("seller_zip_code_prefix"), 5, "0").alias("zip_code_prefix"),
        F.initcap(F.trim("seller_city")).alias("city"),
        F.upper(F.trim("seller_state")).alias("state"),
        *[c for c in df.columns if c == "_ingested_at"],
    )
    return _latest_per_key(out, KEYS["sellers"])


def clean_products(df: DataFrame, translation: DataFrame) -> DataFrame:
    tr = translation.select(
        F.trim("product_category_name").alias("product_category_name"),
        F.trim("product_category_name_english").alias("category_en"),
    ).dropDuplicates(["product_category_name"])
    out = df.select(
        F.trim("product_id").alias("product_id"),
        F.trim("product_category_name").alias("product_category_name"),
        F.col("product_photos_qty").cast("int").alias("photos_qty"),
        F.col("product_weight_g").cast("int").alias("weight_g"),
        (
            F.col("product_length_cm").cast("int")
            * F.col("product_height_cm").cast("int")
            * F.col("product_width_cm").cast("int")
        ).alias("volume_cm3"),
        *[c for c in df.columns if c == "_ingested_at"],
    )
    out = (
        out.join(tr, "product_category_name", "left")
        .withColumn("category", F.coalesce("category_en", "product_category_name", F.lit("unknown")))
        .drop("category_en")
    )
    return _latest_per_key(out, KEYS["products"])


def clean_geolocation(df: DataFrame) -> DataFrame:
    # the raw file has ~1M rows with many points per zip prefix; keep one centroid per prefix
    return (
        df.select(
            F.lpad(F.trim("geolocation_zip_code_prefix"), 5, "0").alias("zip_code_prefix"),
            F.col("geolocation_lat").cast("double").alias("lat"),
            F.col("geolocation_lng").cast("double").alias("lng"),
            F.upper(F.trim("geolocation_state")).alias("state"),
        )
        .filter(F.col("lat").between(-34, 6) & F.col("lng").between(-74, -34))
        .groupBy("zip_code_prefix")
        .agg(F.avg("lat").alias("lat"), F.avg("lng").alias("lng"), F.first("state").alias("state"))
    )


CLEANERS = {
    "orders": clean_orders,
    "order_items": clean_order_items,
    "order_payments": clean_order_payments,
    "order_reviews": clean_order_reviews,
    "customers": clean_customers,
    "sellers": clean_sellers,
    "geolocation": clean_geolocation,
}
