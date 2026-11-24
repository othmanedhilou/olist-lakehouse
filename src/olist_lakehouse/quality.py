from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class QualityError(Exception):
    pass


@dataclass
class CheckResult:
    table: str
    check: str
    failed_rows: int
    blocking: bool

    @property
    def passed(self) -> bool:
        return self.failed_rows == 0


@dataclass
class Expectations:
    table: str
    not_null: list[str] = field(default_factory=list)
    unique: list[list[str]] = field(default_factory=list)
    ranges: dict[str, tuple] = field(default_factory=dict)
    accepted: dict[str, list] = field(default_factory=dict)
    # checks listed here only warn instead of stopping the pipeline
    warn_only: set[str] = field(default_factory=set)

    def run(self, df: DataFrame) -> list[CheckResult]:
        results = []
        for col in self.not_null:
            name = f"not_null:{col}"
            n = df.filter(F.col(col).isNull()).count()
            results.append(CheckResult(self.table, name, n, name not in self.warn_only))
        for cols in self.unique:
            name = f"unique:{','.join(cols)}"
            n = df.groupBy(*cols).count().filter("count > 1").count()
            results.append(CheckResult(self.table, name, n, name not in self.warn_only))
        for col, (lo, hi) in self.ranges.items():
            name = f"range:{col}"
            out_of_range = F.lit(False)
            if lo is not None:
                out_of_range = out_of_range | (F.col(col) < lo)
            if hi is not None:
                out_of_range = out_of_range | (F.col(col) > hi)
            n = df.filter(F.col(col).isNotNull() & out_of_range).count()
            results.append(CheckResult(self.table, name, n, name not in self.warn_only))
        for col, values in self.accepted.items():
            name = f"accepted:{col}"
            n = df.filter(F.col(col).isNotNull() & ~F.col(col).isin(values)).count()
            results.append(CheckResult(self.table, name, n, name not in self.warn_only))
        return results


ORDER_STATUSES = [
    "created", "approved", "invoiced", "processing", "shipped",
    "delivered", "canceled", "unavailable",
]

SILVER_EXPECTATIONS = {
    "orders": Expectations(
        "orders",
        not_null=["order_id", "customer_id", "purchased_at"],
        unique=[["order_id"]],
        accepted={"order_status": ORDER_STATUSES},
        ranges={"delivery_days": (0, 365)},
        warn_only={"range:delivery_days"},
    ),
    "order_items": Expectations(
        "order_items",
        not_null=["order_id", "product_id", "seller_id", "price"],
        unique=[["order_id", "order_item_id"]],
        ranges={"price": (0, None), "freight_value": (0, None)},
    ),
    "order_payments": Expectations(
        "order_payments",
        not_null=["order_id", "payment_value"],
        ranges={"payment_value": (0, None)},
    ),
    "order_reviews": Expectations(
        "order_reviews",
        not_null=["review_id", "order_id"],
        ranges={"review_score": (1, 5)},
    ),
    "customers": Expectations("customers", not_null=["customer_id"], unique=[["customer_id"]]),
    "sellers": Expectations("sellers", not_null=["seller_id"], unique=[["seller_id"]]),
    "products": Expectations("products", not_null=["product_id"], unique=[["product_id"]]),
}


def enforce(results: list[CheckResult]) -> list[CheckResult]:
    failures = [r for r in results if not r.passed and r.blocking]
    if failures:
        detail = ", ".join(f"{r.table}.{r.check}={r.failed_rows}" for r in failures)
        raise QualityError(f"blocking checks failed: {detail}")
    return [r for r in results if not r.passed]
