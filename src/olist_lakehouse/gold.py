from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def dim_date(spark: SparkSession, start: str = "2016-01-01", end: str = "2019-12-31") -> DataFrame:
    return (
        spark.sql(f"select explode(sequence(to_date('{start}'), to_date('{end}'), interval 1 day)) as date")
        .select(
            F.date_format("date", "yyyyMMdd").cast("int").alias("date_key"),
            "date",
            F.year("date").alias("year"),
            F.quarter("date").alias("quarter"),
            F.month("date").alias("month"),
            F.date_format("date", "MMMM").alias("month_name"),
            F.weekofyear("date").alias("week"),
            F.dayofweek("date").alias("day_of_week"),
            F.dayofweek("date").isin(1, 7).alias("is_weekend"),
        )
    )


def dim_customer(customers: DataFrame, geo: DataFrame) -> DataFrame:
    return (
        customers.join(geo.select("zip_code_prefix", "lat", "lng"), "zip_code_prefix", "left")
        .select("customer_id", "customer_unique_id", "zip_code_prefix", "city", "state", "lat", "lng")
    )


def dim_seller(sellers: DataFrame, geo: DataFrame) -> DataFrame:
    return (
        sellers.join(geo.select("zip_code_prefix", "lat", "lng"), "zip_code_prefix", "left")
        .select("seller_id", "zip_code_prefix", "city", "state", "lat", "lng")
    )


def dim_product(products: DataFrame) -> DataFrame:
    return products.select("product_id", "category", "photos_qty", "weight_g", "volume_cm3")


def fact_sales(orders: DataFrame, items: DataFrame, payments: DataFrame, reviews: DataFrame) -> DataFrame:
    """One row per order line. Payments and reviews are per order, so they are
    spread across lines proportionally to the line price to keep sums correct."""
    pay = payments.groupBy("order_id").agg(
        F.sum("payment_value").alias("order_payment_value"),
        F.max("payment_installments").alias("installments"),
        F.first("payment_type").alias("main_payment_type"),
    )
    rev = reviews.groupBy("order_id").agg(F.avg("review_score").alias("review_score"))
    order_total = items.groupBy("order_id").agg(F.sum("price").alias("order_items_total"))

    return (
        items.join(orders, "order_id", "inner")
        .join(order_total, "order_id", "left")
        .join(pay, "order_id", "left")
        .join(rev, "order_id", "left")
        .select(
            "order_id",
            "order_item_id",
            "customer_id",
            "product_id",
            "seller_id",
            F.date_format("purchased_at", "yyyyMMdd").cast("int").alias("purchase_date_key"),
            "order_status",
            "price",
            "freight_value",
            (F.col("price") + F.col("freight_value")).alias("line_revenue"),
            F.round(
                F.col("order_payment_value") * F.col("price") / F.col("order_items_total"), 2
            ).alias("allocated_payment"),
            "installments",
            "main_payment_type",
            "delivery_days",
            "is_late",
            "review_score",
        )
    )


def kpi_monthly(fact: DataFrame) -> DataFrame:
    return (
        fact.filter(F.col("order_status") != "canceled")
        .withColumn("month", F.substring(F.col("purchase_date_key").cast("string"), 1, 6))
        .groupBy("month")
        .agg(
            F.countDistinct("order_id").alias("orders"),
            F.round(F.sum("line_revenue"), 2).alias("revenue"),
            F.round(F.sum("line_revenue") / F.countDistinct("order_id"), 2).alias("avg_basket"),
            F.round(F.avg(F.col("is_late").cast("int")) * 100, 2).alias("late_rate_pct"),
            F.round(F.avg("review_score"), 2).alias("avg_review"),
        )
        .orderBy("month")
    )
