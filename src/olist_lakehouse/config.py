import os
from dataclasses import dataclass

SOURCES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# business keys used for deduplication in silver and for MERGE conditions
KEYS = {
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "order_payments": ["order_id", "payment_sequential"],
    "order_reviews": ["review_id", "order_id"],
    "customers": ["customer_id"],
    "sellers": ["seller_id"],
    "products": ["product_id"],
    "geolocation": ["geolocation_zip_code_prefix"],
    "category_translation": ["product_category_name"],
}


@dataclass(frozen=True)
class LakeConfig:
    landing_path: str
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"

    def table(self, layer: str, name: str) -> str:
        schema = {"bronze": self.bronze_schema, "silver": self.silver_schema, "gold": self.gold_schema}[layer]
        return f"{self.catalog}.{schema}.{name}"

    @classmethod
    def from_env(cls) -> "LakeConfig":
        return cls(
            landing_path=os.environ.get(
                "OLIST_LANDING_PATH",
                "abfss://landing@stolistlake.dfs.core.windows.net/olist",
            ),
            catalog=os.environ.get("OLIST_CATALOG", "olist"),
        )
