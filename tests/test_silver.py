from olist_lakehouse import silver


def test_orders_keep_latest_version(raw_orders):
    out = {r.order_id: r for r in silver.clean_orders(raw_orders).collect()}
    assert len(out) == 3
    assert out["o3"].order_status == "unavailable"


def test_orders_status_normalized_and_late_flag(raw_orders):
    out = {r.order_id: r for r in silver.clean_orders(raw_orders).collect()}
    assert out["o2"].order_status == "delivered"
    assert out["o2"].is_late is True
    assert out["o1"].is_late is False
    assert out["o1"].delivery_days == 8
    assert out["o3"].is_late is None


def test_items_types(raw_items):
    row = silver.clean_order_items(raw_items).filter("order_id = 'o1'").first()
    assert row.order_item_id == 1
    assert float(row.price) == 29.99


def test_products_translated(spark):
    products = spark.createDataFrame(
        [("p1", "beleza_saude", "1", "500", "10", "10", "10"),
         ("p2", None, "2", "200", "5", "5", "4")],
        ["product_id", "product_category_name", "product_photos_qty", "product_weight_g",
         "product_length_cm", "product_height_cm", "product_width_cm"],
    )
    tr = spark.createDataFrame([("beleza_saude", "health_beauty")],
                               ["product_category_name", "product_category_name_english"])
    out = {r.product_id: r for r in silver.clean_products(products, tr).collect()}
    assert out["p1"].category == "health_beauty"
    assert out["p1"].volume_cm3 == 1000
    assert out["p2"].category == "unknown"


def test_geolocation_one_row_per_zip_and_outliers_removed(spark):
    geo = spark.createDataFrame(
        [("1037", "-23.54", "-46.63", "SP", "sp"),
         ("01037", "-23.56", "-46.65", "sp", "sp"),
         ("2000", "45.0", "2.0", "XX", "x")],
        ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_state",
         "geolocation_city"],
    )
    out = silver.clean_geolocation(geo).collect()
    assert len(out) == 1
    assert out[0].zip_code_prefix == "01037"
    assert round(out[0].lat, 2) == -23.55
