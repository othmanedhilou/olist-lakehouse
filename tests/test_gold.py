from olist_lakehouse import gold, silver
from olist_lakehouse.merge import merge_condition


def _silver(raw_orders, raw_items, raw_payments, raw_reviews):
    return (
        silver.clean_orders(raw_orders),
        silver.clean_order_items(raw_items),
        silver.clean_order_payments(raw_payments),
        silver.clean_order_reviews(raw_reviews),
    )


def test_fact_one_row_per_line_and_payment_allocated(raw_orders, raw_items, raw_payments, raw_reviews):
    fact = gold.fact_sales(*_silver(raw_orders, raw_items, raw_payments, raw_reviews))
    rows = {(r.order_id, r.order_item_id): r for r in fact.collect()}
    assert len(rows) == 4
    # order o2: 165 paid, lines of 100 and 50 -> 110 and 55
    assert float(rows[("o2", 1)].allocated_payment) == 110.0
    assert float(rows[("o2", 2)].allocated_payment) == 55.0
    assert rows[("o2", 1)].purchase_date_key == 20180724


def test_kpi_monthly_excludes_canceled(spark, raw_orders, raw_items, raw_payments, raw_reviews):
    fact = gold.fact_sales(*_silver(raw_orders, raw_items, raw_payments, raw_reviews))
    kpi = {r.month: r for r in gold.kpi_monthly(fact).collect()}
    assert set(kpi) == {"201710", "201807", "201808"}
    assert kpi["201807"].orders == 1
    assert float(kpi["201807"].revenue) == 165.0
    assert kpi["201807"].late_rate_pct == 100.0


def test_dim_date_covers_range(spark):
    d = gold.dim_date(spark, "2017-01-01", "2017-01-31")
    assert d.count() == 31
    assert d.filter("is_weekend").count() == 9


def test_merge_condition():
    assert merge_condition(["order_id", "order_item_id"]) == "t.order_id = s.order_id and t.order_item_id = s.order_item_id"
