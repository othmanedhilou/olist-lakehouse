"""Order-level features for late delivery prediction.

Only information known when the order is placed is used (basket, products,
customer and seller location, promised delivery date, payment). Anything that
happens after purchase (carrier date, actual delivery, review) is excluded,
otherwise the model would learn from the answer.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

NUMERIC = [
    "n_items",
    "n_sellers",
    "total_price",
    "total_freight",
    "freight_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "promised_days",
    "distance_km",
    "installments",
    "purchase_month",
    "purchase_dow",
    "purchase_hour",
]
CATEGORICAL = ["customer_state", "seller_state", "main_category", "payment_type", "same_state"]
FEATURES = NUMERIC + CATEGORICAL
LABEL = "is_late"


def _haversine_km(lat1, lon1, lat2, lon2):
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = F.sin(dlat / 2) ** 2 + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2
    return F.round(2 * 6371.0 * F.asin(F.sqrt(a)), 1)


def build_features(
    orders: DataFrame,
    items: DataFrame,
    payments: DataFrame,
    products: DataFrame,
    customers: DataFrame,
    sellers: DataFrame,
    geo: DataFrame,
) -> DataFrame:
    lines = items.join(products.select("product_id", "category", "weight_g", "volume_cm3"), "product_id", "left")

    # main seller and category = those of the most expensive line
    w = Window.partitionBy("order_id").orderBy(F.desc("price"), "order_item_id")
    main_line = (
        lines.withColumn("_rn", F.row_number().over(w))
        .filter("_rn = 1")
        .select("order_id", "seller_id", F.col("category").alias("main_category"))
    )
    basket = lines.groupBy("order_id").agg(
        F.count("*").alias("n_items"),
        F.countDistinct("seller_id").alias("n_sellers"),
        F.sum("price").cast("double").alias("total_price"),
        F.sum("freight_value").cast("double").alias("total_freight"),
        F.sum("weight_g").alias("total_weight_g"),
        F.sum("volume_cm3").alias("total_volume_cm3"),
    )
    pay = payments.groupBy("order_id").agg(
        F.max("payment_installments").alias("installments"),
        F.max_by("payment_type", "payment_value").alias("payment_type"),
    )
    g = geo.select("zip_code_prefix", "lat", "lng")
    cust = customers.join(g, "zip_code_prefix", "left").select(
        "customer_id", F.col("state").alias("customer_state"), F.col("lat").alias("c_lat"), F.col("lng").alias("c_lng")
    )
    sell = sellers.join(g, "zip_code_prefix", "left").select(
        "seller_id", F.col("state").alias("seller_state"), F.col("lat").alias("s_lat"), F.col("lng").alias("s_lng")
    )

    df = (
        orders.select("order_id", "customer_id", "order_status", "purchased_at", "estimated_delivery_at", "is_late")
        .join(basket, "order_id", "inner")
        .join(main_line, "order_id", "left")
        .join(pay, "order_id", "left")
        .join(cust, "customer_id", "left")
        .join(sell, "seller_id", "left")
    )
    return df.select(
        "order_id",
        "order_status",
        "purchased_at",
        "n_items",
        "n_sellers",
        "total_price",
        "total_freight",
        F.round(F.col("total_freight") / (F.col("total_price") + F.col("total_freight")), 4).alias("freight_ratio"),
        F.col("total_weight_g").cast("double").alias("total_weight_g"),
        F.col("total_volume_cm3").cast("double").alias("total_volume_cm3"),
        F.datediff("estimated_delivery_at", "purchased_at").alias("promised_days"),
        _haversine_km(F.col("c_lat"), F.col("c_lng"), F.col("s_lat"), F.col("s_lng")).alias("distance_km"),
        F.coalesce("installments", F.lit(1)).alias("installments"),
        F.month("purchased_at").alias("purchase_month"),
        F.dayofweek("purchased_at").alias("purchase_dow"),
        F.hour("purchased_at").alias("purchase_hour"),
        F.coalesce("customer_state", F.lit("NA")).alias("customer_state"),
        F.coalesce("seller_state", F.lit("NA")).alias("seller_state"),
        F.coalesce("main_category", F.lit("unknown")).alias("main_category"),
        F.coalesce("payment_type", F.lit("unknown")).alias("payment_type"),
        F.when(F.col("customer_state") == F.col("seller_state"), "yes").otherwise("no").alias("same_state"),
        F.col("is_late").cast("int").alias(LABEL),
    )


def training_set(features: DataFrame) -> DataFrame:
    return features.filter((F.col("order_status") == "delivered") & F.col(LABEL).isNotNull())


def open_orders(features: DataFrame) -> DataFrame:
    return features.filter(F.col("order_status").isin("created", "approved", "invoiced", "processing", "shipped"))
