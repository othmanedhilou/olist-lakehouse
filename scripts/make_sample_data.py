"""Writes a small fake dataset with the exact Olist file layout.

Handy to try the pipeline before downloading the real files from Kaggle.

    python scripts/make_sample_data.py --out data/sample --orders 2000
"""
import argparse
import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

STATES = {"SP": (-23.55, -46.63), "RJ": (-22.90, -43.20), "MG": (-19.92, -43.94),
          "RS": (-30.03, -51.23), "PR": (-25.43, -49.27), "BA": (-12.97, -38.50)}
CATEGORIES = [("beleza_saude", "health_beauty"), ("informatica_acessorios", "computers_accessories"),
              ("moveis_decoracao", "furniture_decor"), ("esporte_lazer", "sports_leisure"),
              ("cama_mesa_banho", "bed_bath_table"), ("relogios_presentes", "watches_gifts")]
PAYMENT_TYPES = ["credit_card", "credit_card", "credit_card", "boleto", "voucher", "debit_card"]
FMT = "%Y-%m-%d %H:%M:%S"


def uid():
    return uuid.uuid4().hex


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument("--orders", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    zips = {s: [f"{random.randint(1000, 99999):05d}" for _ in range(15)] for s in STATES}
    geo = []
    for s, (lat, lng) in STATES.items():
        for z in zips[s]:
            for _ in range(3):
                geo.append([z, round(lat + random.uniform(-1, 1), 6), round(lng + random.uniform(-1, 1), 6), "cidade", s.lower()])

    customers = []
    for _ in range(int(a.orders * 0.95)):
        s = random.choice(list(STATES))
        customers.append([uid(), uid(), random.choice(zips[s]), "cidade", s])
    sellers = []
    for _ in range(80):
        s = random.choice(list(STATES))
        sellers.append([uid(), random.choice(zips[s]), "cidade", s])
    products = []
    for _ in range(300):
        cat = random.choice(CATEGORIES)[0]
        products.append([uid(), cat, random.randint(20, 60), random.randint(50, 900), random.randint(1, 6),
                         random.randint(100, 5000), random.randint(10, 60), random.randint(2, 40), random.randint(10, 50)])

    orders, items, payments, reviews = [], [], [], []
    start = datetime(2017, 1, 1)
    for _ in range(a.orders):
        oid = uid()
        cust = random.choice(customers)[0]
        bought = start + timedelta(minutes=random.randint(0, 60 * 24 * 600))
        estimated = bought + timedelta(days=random.randint(15, 30))
        status = random.choices(["delivered", "shipped", "canceled", "invoiced"], [92, 4, 3, 1])[0]
        delivered = bought + timedelta(days=random.randint(3, 35)) if status == "delivered" else None
        orders.append([oid, cust, status, bought.strftime(FMT), (bought + timedelta(hours=2)).strftime(FMT),
                       (bought + timedelta(days=2)).strftime(FMT) if delivered else "",
                       delivered.strftime(FMT) if delivered else "", estimated.strftime("%Y-%m-%d 00:00:00")])
        total = 0.0
        for i in range(1, random.choices([1, 2, 3], [85, 12, 3])[0] + 1):
            price = round(random.uniform(15, 400), 2)
            freight = round(random.uniform(5, 40), 2)
            total += price + freight
            items.append([oid, i, random.choice(products)[0], random.choice(sellers)[0],
                          (bought + timedelta(days=5)).strftime(FMT), price, freight])
        payments.append([oid, 1, random.choice(PAYMENT_TYPES), random.randint(1, 10), round(total, 2)])
        if delivered:
            late = delivered > estimated
            score = random.choices([1, 2, 3, 4, 5], [40, 15, 15, 15, 15] if late else [5, 3, 9, 20, 63])[0]
            reviews.append([uid(), oid, score, "", "", (delivered + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00"),
                            (delivered + timedelta(days=2)).strftime(FMT)])

    write(out / "olist_geolocation_dataset.csv",
          ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"], geo)
    write(out / "olist_customers_dataset.csv",
          ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"], customers)
    write(out / "olist_sellers_dataset.csv",
          ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"], sellers)
    write(out / "olist_products_dataset.csv",
          ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght",
           "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"], products)
    write(out / "product_category_name_translation.csv",
          ["product_category_name", "product_category_name_english"], CATEGORIES)
    write(out / "olist_orders_dataset.csv",
          ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
           "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"], orders)
    write(out / "olist_order_items_dataset.csv",
          ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"], items)
    write(out / "olist_order_payments_dataset.csv",
          ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"], payments)
    write(out / "olist_order_reviews_dataset.csv",
          ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
           "review_creation_date", "review_answer_timestamp"], reviews)
    print(f"{len(orders)} orders, {len(items)} items written to {out}")


if __name__ == "__main__":
    main()
