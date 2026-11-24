import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from olist_lakehouse import silver
from olist_lakehouse.ml.features import FEATURES, build_features, open_orders, training_set
from olist_lakehouse.ml.train import evaluate, time_split


def _silver_inputs(spark, raw_orders, raw_items, raw_payments):
    products = spark.createDataFrame(
        [("p1", "beleza_saude", "1", "500", "10", "10", "10"), ("p2", "esporte_lazer", "2", "2000", "30", "20", "20")],
        "product_id string, product_category_name string, product_photos_qty string, product_weight_g string, "
        "product_length_cm string, product_height_cm string, product_width_cm string",
    )
    tr = spark.createDataFrame([("beleza_saude", "health_beauty")], "product_category_name string, product_category_name_english string")
    customers = spark.createDataFrame(
        [("c1", "u1", "01037", "sao paulo", "SP"), ("c2", "u2", "20000", "rio", "RJ"), ("c3", "u3", "01037", "sao paulo", "SP")],
        "customer_id string, customer_unique_id string, customer_zip_code_prefix string, customer_city string, customer_state string",
    )
    sellers = spark.createDataFrame(
        [("s1", "01037", "sao paulo", "SP"), ("s2", "20000", "rio", "RJ")],
        "seller_id string, seller_zip_code_prefix string, seller_city string, seller_state string",
    )
    geo = spark.createDataFrame(
        [("01037", "-23.55", "-46.63", "SP"), ("20000", "-22.90", "-43.20", "RJ")],
        "geolocation_zip_code_prefix string, geolocation_lat string, geolocation_lng string, geolocation_state string",
    )
    return (
        silver.clean_orders(raw_orders),
        silver.clean_order_items(raw_items),
        silver.clean_order_payments(raw_payments),
        silver.clean_products(products, tr),
        silver.clean_customers(customers),
        silver.clean_sellers(sellers),
        silver.clean_geolocation(geo),
    )


def test_features_one_row_per_order(spark, raw_orders, raw_items, raw_payments):
    feats = build_features(*_silver_inputs(spark, raw_orders, raw_items, raw_payments))
    rows = {r.order_id: r for r in feats.collect()}
    assert set(rows) == {"o1", "o2", "o3"}
    o2 = rows["o2"]
    assert (o2.n_items, o2.n_sellers) == (2, 2)
    assert o2.total_price == 150.0
    assert o2.main_category == "esporte_lazer"          # category of the most expensive line
    assert o2.seller_state == "RJ" and o2.customer_state == "RJ" and o2.same_state == "yes"
    assert o2.promised_days == 20
    assert rows["o1"].distance_km == 0.0
    assert set(FEATURES) <= set(feats.columns)


def test_no_post_purchase_information_in_features():
    leaking = {"delivered_customer_at", "delivered_carrier_at", "delivery_days", "review_score", "approved_at"}
    assert not leaking & set(FEATURES)


def test_training_and_open_order_sets(spark, raw_orders, raw_items, raw_payments):
    feats = build_features(*_silver_inputs(spark, raw_orders, raw_items, raw_payments))
    assert {r.order_id for r in training_set(feats).collect()} == {"o1", "o2"}
    assert open_orders(feats).count() == 0
    assert training_set(feats).filter(F.col("is_late").isNull()).count() == 0


def test_time_split_keeps_recent_orders_for_test():
    df = pd.DataFrame({"purchased_at": pd.date_range("2017-01-01", periods=100, freq="D"), "x": range(100)})
    train, test = time_split(df, 0.2)
    assert train["purchased_at"].max() < test["purchased_at"].min()
    assert 18 <= len(test) <= 22


def test_evaluate_top_decile():
    y = np.array([1] * 10 + [0] * 90)
    perfect = np.linspace(1, 0, 100)
    m = evaluate(y, perfect)
    assert m["roc_auc"] == 1.0
    assert m["precision_top10"] == 1.0
    assert m["recall_top10"] == 1.0
    assert m["base_rate"] == 0.1
