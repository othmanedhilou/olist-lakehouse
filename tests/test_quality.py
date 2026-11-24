import pytest

from olist_lakehouse.quality import Expectations, QualityError, enforce


def test_blocking_check_raises(spark):
    df = spark.createDataFrame([("a", 1), ("a", 2), (None, 3)], ["id", "v"])
    exp = Expectations("t", not_null=["id"], unique=[["id"]])
    results = exp.run(df)
    assert {r.check: r.failed_rows for r in results} == {"not_null:id": 1, "unique:id": 1}
    with pytest.raises(QualityError):
        enforce(results)


def test_range_and_accepted(spark):
    df = spark.createDataFrame([(1, "x"), (6, "y"), (None, "z")], ["score", "status"])
    exp = Expectations("t", ranges={"score": (1, 5)}, accepted={"status": ["x", "y"]})
    res = {r.check: r.failed_rows for r in exp.run(df)}
    assert res == {"range:score": 1, "accepted:status": 1}


def test_warn_only_does_not_raise(spark):
    df = spark.createDataFrame([(400,)], ["delivery_days"])
    exp = Expectations("t", ranges={"delivery_days": (0, 365)}, warn_only={"range:delivery_days"})
    warnings = enforce(exp.run(df))
    assert len(warnings) == 1
