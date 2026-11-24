import pytest
from pyspark.sql import SparkSession


def _strings(cols):
    return ", ".join(f"{c} string" for c in cols)


@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .appName("olist-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture
def raw_orders(spark):
    cols = [
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date", "_ingested_at",
    ]
    rows = [
        ("o1", "c1", "delivered", "2017-10-02 10:56:33", "2017-10-02 11:07:15",
         "2017-10-04 19:55:00", "2017-10-10 21:25:13", "2017-10-18 00:00:00", "2026-01-01 00:00:00"),
        ("o2", "c2", "DELIVERED ", "2018-07-24 20:41:37", "2018-07-26 03:24:27",
         "2018-07-26 14:31:00", "2018-08-20 15:17:33", "2018-08-13 00:00:00", "2026-01-01 00:00:00"),
        ("o3", "c3", "canceled", "2018-08-08 08:38:49", None, None, None,
         "2018-09-04 00:00:00", "2026-01-01 00:00:00"),
        # same order re-sent in a later batch with a new status
        ("o3", "c3", "unavailable", "2018-08-08 08:38:49", None, None, None,
         "2018-09-04 00:00:00", "2026-01-02 00:00:00"),
    ]
    from pyspark.sql import functions as F
    return spark.createDataFrame(rows, _strings(cols)).withColumn("_ingested_at", F.to_timestamp("_ingested_at"))


@pytest.fixture
def raw_items(spark):
    cols = ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"]
    rows = [
        ("o1", "1", "p1", "s1", "2017-10-06 11:07:15", "29.99", "8.72"),
        ("o2", "1", "p2", "s2", "2018-07-30 03:24:27", "100.00", "10.00"),
        ("o2", "2", "p1", "s1", "2018-07-30 03:24:27", "50.00", "5.00"),
        ("o3", "1", "p2", "s2", "2018-08-10 03:24:27", "20.00", "4.00"),
    ]
    return spark.createDataFrame(rows, _strings(cols))


@pytest.fixture
def raw_payments(spark):
    cols = ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"]
    rows = [
        ("o1", "1", "credit_card", "1", "38.71"),
        ("o2", "1", "boleto", "1", "100.00"),
        ("o2", "2", "voucher", "1", "65.00"),
        ("o3", "1", "credit_card", "3", "24.00"),
    ]
    return spark.createDataFrame(rows, _strings(cols))


@pytest.fixture
def raw_reviews(spark):
    cols = ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
            "review_creation_date", "review_answer_timestamp"]
    rows = [
        ("r1", "o1", "4", None, "ok", "2017-10-11 00:00:00", "2017-10-12 03:43:48"),
        ("r2", "o2", "2", None, "atrasado", "2018-08-21 00:00:00", "2018-08-22 19:07:58"),
    ]
    return spark.createDataFrame(rows, _strings(cols))
