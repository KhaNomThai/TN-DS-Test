"""
=============================================================================
SME Retail Mock Data Generator
Goal: Maximize Revenue
=============================================================================
สร้าง Mock Data สำหรับ 8 ตาราง พร้อม Feature Engineering
ข้อมูลทั้งหมดมีความสัมพันธ์เชิงตรรกะ (Logical Consistency)

Tables:
  1. Product Master
  2. Promotion Master
  3. Store Master
  4. Customer Master
  5. Warehouse Master
  6. Purchasing Order
  7. Stock Movement Transaction
  8. Sales Transaction

Requirements:
  - pip install pandas faker numpy
=============================================================================
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("th_TH")
Faker.seed(SEED)

OUTPUT_DIR = "mock_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Volume settings – ปรับได้ตามต้องการ
NUM_PRODUCTS = 100
NUM_STORES = 10
NUM_WAREHOUSES = 3
NUM_CUSTOMERS = 3_000
NUM_PROMOTIONS = 150
NUM_PO = 2_500
NUM_STOCK_MOVEMENTS = 5_000
NUM_SALES = 300_000

# Date range for the dataset (2 years of data)
DATE_START = datetime(2024, 1, 1)
DATE_END = datetime(2025, 12, 31)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def random_date(start: datetime, end: datetime) -> datetime:
    """สุ่มวันที่ระหว่าง start กับ end"""
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))


def random_date_between(start: datetime, end: datetime) -> datetime:
    """Alias ที่ปลอดภัยกว่า – guarantee start <= result <= end"""
    if start >= end:
        return start
    return start + timedelta(days=random.randint(0, (end - start).days))


# ===================================================================
# 1. PRODUCT MASTER
# ===================================================================
print("[1/8] Generating Product Master ...")

CATEGORIES = {
    "อาหารสด": {
        "subcategories": ["เนื้อสัตว์", "อาหารทะเล", "ผักสด", "ผลไม้"],
        "price_range": (30, 500),
        "margin_range": (0.10, 0.35),
        "shelf_life_days": (3, 14),
    },
    "อาหารแห้ง": {
        "subcategories": ["บะหมี่กึ่งสำเร็จรูป", "เครื่องปรุงรส", "ข้าวสาร", "ขนมขบเคี้ยว"],
        "price_range": (10, 300),
        "margin_range": (0.20, 0.45),
        "shelf_life_days": (90, 730),
    },
    "เครื่องดื่ม": {
        "subcategories": ["น้ำอัดลม", "น้ำผลไม้", "นม", "กาแฟ", "น้ำดื่ม"],
        "price_range": (10, 200),
        "margin_range": (0.25, 0.50),
        "shelf_life_days": (30, 365),
    },
    "ของใช้ในบ้าน": {
        "subcategories": ["ผงซักฟอก", "น้ำยาล้างจาน", "กระดาษทิชชู่", "ถุงขยะ"],
        "price_range": (20, 400),
        "margin_range": (0.15, 0.40),
        "shelf_life_days": (365, 1095),
    },
    "ของใช้ส่วนตัว": {
        "subcategories": ["สบู่", "แชมพู", "ยาสีฟัน", "ผ้าอนามัย"],
        "price_range": (25, 500),
        "margin_range": (0.20, 0.50),
        "shelf_life_days": (365, 1095),
    },
    "เครื่องเขียน": {
        "subcategories": ["ปากกา", "สมุด", "ดินสอ", "กาว"],
        "price_range": (5, 150),
        "margin_range": (0.30, 0.60),
        "shelf_life_days": (730, 1825),
    },
}

BRANDS = [
    "CP", "Unilever", "P&G", "Nestlé", "Thai Union", "Betagro",
    "Oishi", "Singha", "Dutch Mill", "Mama", "Knorr", "Colgate",
    "Sunsilk", "Lux", "Downy", "Boon Rawd", "ThaiBev", "Ichitan",
    "Double A", "SCG",
]

products = []
for i in range(1, NUM_PRODUCTS + 1):
    cat = random.choice(list(CATEGORIES.keys()))
    info = CATEGORIES[cat]
    subcat = random.choice(info["subcategories"])
    brand = random.choice(BRANDS)

    price = round(random.uniform(*info["price_range"]), 2)
    margin = round(random.uniform(*info["margin_range"]), 4)
    cost_price = round(price * (1 - margin), 2)
    shelf_life = random.randint(*info["shelf_life_days"])

    # Feature Engineering columns
    is_perishable = 1 if shelf_life <= 30 else 0
    price_tier = (
        "budget" if price < 50
        else "mid" if price < 200
        else "premium"
    )

    products.append({
        "product_id": f"PRD-{i:04d}",
        "product_name": f"{brand} {subcat} #{i}",
        "price": price,
        "cost_price": cost_price,                 # Feature Engineering
        "profit_margin": margin,                   # Feature Engineering
        "category": cat,
        "subcategory": subcat,
        "brand": brand,
        "shelf_life_days": shelf_life,
        "is_perishable": is_perishable,            # Feature Engineering
        "price_tier": price_tier,                  # Feature Engineering
        "product_taxonomies": f"{cat} > {subcat} > {brand}",
    })

df_products = pd.DataFrame(products)

# ===================================================================
# 2. STORE MASTER
# ===================================================================
print("[2/8] Generating Store Master ...")

STORE_TYPES = ["convenience", "supermarket", "minimart", "hypermarket"]
REGIONS = ["กรุงเทพฯ", "ภาคกลาง", "ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ", "ภาคใต้"]

stores = []
for i in range(1, NUM_STORES + 1):
    store_type = random.choice(STORE_TYPES)
    region = random.choice(REGIONS)
    size_sqm = random.randint(50, 3000)
    open_date = random_date(datetime(2015, 1, 1), datetime(2023, 12, 31))

    stores.append({
        "store_id": f"STR-{i:03d}",
        "store_name": f"ร้าน {fake.city()} สาขา {i}",
        "store_type": store_type,
        "region": region,
        "province": fake.city(),
        "size_sqm": size_sqm,
        "open_date": open_date.strftime("%Y-%m-%d"),
        "store_taxonomies": f"{region} > {store_type} > {size_sqm}sqm",
    })

df_stores = pd.DataFrame(stores)

# ===================================================================
# 3. CUSTOMER MASTER
# ===================================================================
print("[3/8] Generating Customer Master ...")

GENDERS = ["M", "F", "Other"]
AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
MEMBERSHIP_TIERS = ["bronze", "silver", "gold", "platinum"]

customers = []
for i in range(1, NUM_CUSTOMERS + 1):
    gender = random.choices(GENDERS, weights=[0.45, 0.50, 0.05])[0]
    age_group = random.choice(AGE_GROUPS)
    membership = random.choices(
        MEMBERSHIP_TIERS, weights=[0.40, 0.30, 0.20, 0.10]
    )[0]
    registration_date = random_date(datetime(2020, 1, 1), DATE_END)

    # Feature Engineering – behavioural proxies (จะถูก update หลังสร้าง Sales)
    customers.append({
        "customer_id": f"CUS-{i:05d}",
        "customer_name": fake.name(),
        "gender": gender,
        "age_group": age_group,
        "membership_tier": membership,
        "registration_date": registration_date.strftime("%Y-%m-%d"),
        "preferred_store_id": random.choice(
            [f"STR-{s:03d}" for s in range(1, NUM_STORES + 1)]
        ),
        # Placeholder – จะ update ภายหลัง
        "total_purchases": 0,            # Feature Engineering
        "total_spend": 0.0,              # Feature Engineering
        "avg_basket_size": 0.0,          # Feature Engineering
        "purchase_frequency": 0.0,       # Feature Engineering
        "days_since_last_purchase": 0,   # Feature Engineering
        "customer_taxonomies": f"{gender} > {age_group} > {membership}",
    })

df_customers = pd.DataFrame(customers)

# ===================================================================
# 4. WAREHOUSE MASTER
# ===================================================================
print("[4/8] Generating Warehouse Master ...")

warehouses = []
for i in range(1, NUM_WAREHOUSES + 1):
    wh_type = random.choice(["central", "regional", "cold_storage"])
    region = random.choice(REGIONS)
    capacity = random.randint(5000, 50000)

    warehouses.append({
        "warehouse_id": f"WH-{i:02d}",
        "warehouse_name": f"คลัง {region} #{i}",
        "warehouse_type": wh_type,
        "region": region,
        "capacity_units": capacity,
        "warehouse_taxonomies": f"{region} > {wh_type} > cap:{capacity}",
    })

df_warehouses = pd.DataFrame(warehouses)

# ===================================================================
# 5. PROMOTION MASTER
# ===================================================================
print("[5/8] Generating Promotion Master ...")

PROMO_TYPES = ["percentage_off", "buy_x_get_y", "bundle", "clearance"]

promotions = []
product_ids = df_products["product_id"].tolist()

for i in range(1, NUM_PROMOTIONS + 1):
    promo_type = random.choice(PROMO_TYPES)

    if promo_type == "percentage_off":
        discount = round(random.choice([5, 10, 15, 20, 25, 30, 40, 50]), 2)
    elif promo_type == "clearance":
        discount = round(random.choice([30, 40, 50, 60, 70]), 2)
    elif promo_type == "buy_x_get_y":
        discount = round(random.choice([20, 25, 33, 50]), 2)
    else:  # bundle
        discount = round(random.choice([10, 15, 20, 25]), 2)

    # Promotion period: 7-60 days
    start_date = random_date(DATE_START, DATE_END - timedelta(days=60))
    duration = random.randint(7, 60)
    end_date = start_date + timedelta(days=duration)
    if end_date > DATE_END:
        end_date = DATE_END

    # Each promotion covers 1-5 products
    num_prods = random.randint(1, 5)
    promo_products = random.sample(product_ids, num_prods)

    for pid in promo_products:
        promotions.append({
            "promotion_id": f"PRM-{i:03d}",
            "promotion_type": promo_type,
            "discount": discount,
            "product_id": pid,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        })

df_promotions = pd.DataFrame(promotions)

# Build quick lookup: product_id -> list of (promo_id, discount, start, end)
promo_lookup = {}
for _, row in df_promotions.iterrows():
    pid = row["product_id"]
    promo_lookup.setdefault(pid, []).append({
        "promotion_id": row["promotion_id"],
        "discount": row["discount"],
        "start": datetime.strptime(row["start_date"], "%Y-%m-%d"),
        "end": datetime.strptime(row["end_date"], "%Y-%m-%d"),
    })

# ===================================================================
# 6. PURCHASING ORDER (PO)
# ===================================================================
print("[6/8] Generating Purchasing Orders ...")

warehouse_ids = df_warehouses["warehouse_id"].tolist()
product_price_map = dict(zip(df_products["product_id"], df_products["cost_price"]))
product_shelf_map = dict(zip(df_products["product_id"], df_products["shelf_life_days"]))

po_records = []
for i in range(1, NUM_PO + 1):
    pid = random.choice(product_ids)
    wh_id = random.choice(warehouse_ids)

    qty = random.randint(50, 2000)
    cost_per_unit = product_price_map[pid]
    # Supplier price can be slightly lower than retail cost (bulk discount)
    po_price_per_unit = round(cost_per_unit * random.uniform(0.70, 0.95), 2)
    po_price_total = round(po_price_per_unit * qty, 2)

    shelf_life = product_shelf_map[pid]

    # Ensure enough room: need at least 2 days for po + 2 days for shipping
    effective_shelf = max(shelf_life, 6)
    manufacturing_date = random_date(
        DATE_START - timedelta(days=30),
        DATE_END - timedelta(days=effective_shelf)
    )
    expire_date = manufacturing_date + timedelta(days=shelf_life)

    # po_date: 1-min(7, shelf_life//3) days after manufacturing
    max_po_lead = max(1, min(7, shelf_life // 3))
    po_date = manufacturing_date + timedelta(days=random.randint(1, max_po_lead))

    # arrival_date: between po_date and expire_date (leave at least 1 day before expiry)
    remaining_days = (expire_date - po_date).days
    if remaining_days <= 1:
        arrival_date = po_date  # same-day arrival for ultra-short shelf life
    else:
        ship_days = random.randint(1, max(1, min(remaining_days - 1, 21)))
        arrival_date = po_date + timedelta(days=ship_days)

    # Final safety: clamp arrival_date
    if arrival_date > expire_date:
        arrival_date = expire_date

    po_records.append({
        "po_id": f"PO-{i:05d}",
        "warehouse_id": wh_id,
        "product_id": pid,
        "po_price_total": po_price_total,
        "po_price_per_unit": po_price_per_unit,
        "unit": "ชิ้น",
        "manufacturing_date": manufacturing_date.strftime("%Y-%m-%d"),
        "po_date": po_date.strftime("%Y-%m-%d"),
        "arrival_date": arrival_date.strftime("%Y-%m-%d"),
        "expire_date": expire_date.strftime("%Y-%m-%d"),
        "qty": qty,
    })

df_po = pd.DataFrame(po_records)

# Quick lookup: po_id -> (product_id, arrival_date, expire_date, warehouse_id)
po_lookup = {}
for _, row in df_po.iterrows():
    po_lookup[row["po_id"]] = {
        "product_id": row["product_id"],
        "arrival_date": datetime.strptime(row["arrival_date"], "%Y-%m-%d"),
        "expire_date": datetime.strptime(row["expire_date"], "%Y-%m-%d"),
        "warehouse_id": row["warehouse_id"],
        "qty": row["qty"],
    }

# ===================================================================
# 7. STOCK MOVEMENT TRANSACTION
# ===================================================================
print("[7/8] Generating Stock Movement Transactions ...")

store_ids = df_stores["store_id"].tolist()
po_ids = list(po_lookup.keys())

stock_movements = []
# Track which POs have been moved to which stores (for sales reference)
po_store_map = {}  # po_id -> list of (store_id, transfer_date, qty_moved)

for i in range(1, NUM_STOCK_MOVEMENTS + 1):
    po_id = random.choice(po_ids)
    po_info = po_lookup[po_id]

    wh_id = po_info["warehouse_id"]
    store_id = random.choice(store_ids)

    receive_date = po_info["arrival_date"] + timedelta(days=random.randint(0, 2))
    transfer_date = receive_date + timedelta(days=random.randint(1, 5))

    # ห้ามโอนหลังหมดอายุ
    if transfer_date > po_info["expire_date"]:
        transfer_date = po_info["expire_date"] - timedelta(days=1)
    if transfer_date < receive_date:
        transfer_date = receive_date

    qty_moved = random.randint(10, min(200, po_info["qty"]))

    stock_movements.append({
        "stock_movement_id": f"SM-{i:05d}",
        "po_id": po_id,
        "receive_date": receive_date.strftime("%Y-%m-%d"),
        "transfer_date": transfer_date.strftime("%Y-%m-%d"),
        "store_id": store_id,
        "warehouse_id": wh_id,
        "qty": qty_moved,
    })

    po_store_map.setdefault(po_id, []).append({
        "store_id": store_id,
        "transfer_date": transfer_date,
        "expire_date": po_info["expire_date"],
        "product_id": po_info["product_id"],
    })

df_stock = pd.DataFrame(stock_movements)

# ===================================================================
# 8. SALES TRANSACTION  (ส่วนที่สำคัญที่สุด – ต้อง Consistent)
# ===================================================================
print("[8/8] Generating Sales Transactions ...")

# Build available inventory timeline: product_id -> list of
# (store_id, available_from, available_until)
inventory_timeline = []
for po_id, movements in po_store_map.items():
    for m in movements:
        inventory_timeline.append({
            "product_id": m["product_id"],
            "store_id": m["store_id"],
            "available_from": m["transfer_date"],
            "available_until": m["expire_date"],
            "po_id": po_id,
        })

product_sell_price = dict(zip(df_products["product_id"], df_products["price"]))
customer_ids = df_customers["customer_id"].tolist()

# Weight customers by membership tier for spending behaviour
customer_tier_map = dict(zip(df_customers["customer_id"], df_customers["membership_tier"]))
tier_weight = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 5}

sales = []
sales_counter = 0

# Use inventory timeline to generate consistent sales
print("    Building inventory index ...")
# Group inventory by product_id for faster lookups
inv_by_product = {}
for inv in inventory_timeline:
    inv_by_product.setdefault(inv["product_id"], []).append(inv)

print("    Generating transactions ...")
attempts = 0
max_attempts = NUM_SALES * 5

while sales_counter < NUM_SALES and attempts < max_attempts:
    attempts += 1

    # Pick a random product
    pid = random.choice(product_ids)
    if pid not in inv_by_product:
        continue

    # Pick a random inventory slot for this product
    inv_slot = random.choice(inv_by_product[pid])
    store_id = inv_slot["store_id"]
    po_id = inv_slot["po_id"]
    avail_from = inv_slot["available_from"]
    avail_until = inv_slot["available_until"]

    if avail_from >= avail_until:
        continue

    # Sale date must be within product availability, but typically soon after store arrival (7-45 days)
    max_days = random.randint(7, 45)
    sale_limit = avail_from + timedelta(days=max_days)
    sale_date_end = min(avail_until, sale_limit)
    sale_date = random_date_between(avail_from, sale_date_end)

    # ---- Determine price and promotion ----
    base_price = product_sell_price[pid]
    applied_promo_id = None
    discount_pct = 0

    # Check if any promotion applies on this date for this product
    if pid in promo_lookup:
        applicable_promos = [
            p for p in promo_lookup[pid]
            if p["start"] <= sale_date <= p["end"]
        ]
        if applicable_promos:
            chosen_promo = random.choice(applicable_promos)
            applied_promo_id = chosen_promo["promotion_id"]
            discount_pct = chosen_promo["discount"]

    actual_price = round(base_price * (1 - discount_pct / 100), 2)
    qty = random.choices(
        [1, 2, 3, 4, 5, 6, 8, 10, 12],
        weights=[35, 25, 15, 8, 6, 4, 3, 2, 2],
    )[0]

    # Pick customer (weighted by tier → higher tiers buy more often)
    cust_id = random.choices(
        customer_ids,
        weights=[tier_weight[customer_tier_map[c]] for c in customer_ids],
    )[0]

    # Add hour/minute for realistic datetime
    hour = random.choices(
        range(24),
        weights=[
            1, 0, 0, 0, 0, 1, 2, 5, 8, 10, 12, 14,  # 00-11
            15, 13, 10, 9, 10, 14, 16, 14, 10, 6, 3, 2,  # 12-23
        ],
    )[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    sale_datetime = sale_date.replace(hour=hour, minute=minute, second=second)

    sales_counter += 1
    sales.append({
        "transaction_id": f"TXN-{sales_counter:06d}",
        "datetime": sale_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "product_id": pid,
        "price": actual_price,
        "qty": qty,
        "total_amount": round(actual_price * qty, 2),    # Feature Engineering
        "customer_id": cust_id,
        "promotion_id": applied_promo_id,
        "store_id": store_id,
        "po_id": po_id,
        "discount_pct": discount_pct,                     # Feature Engineering
        "day_of_week": sale_datetime.strftime("%A"),       # Feature Engineering
        "hour_of_day": hour,                               # Feature Engineering
        "is_weekend": 1 if sale_datetime.weekday() >= 5 else 0,  # Feature Engineering
        "month": sale_datetime.month,                      # Feature Engineering
    })

    if sales_counter % 10_000 == 0:
        print(f"    ... {sales_counter:,} / {NUM_SALES:,}")

df_sales = pd.DataFrame(sales)
print(f"    Generated {len(df_sales):,} sales transactions (attempts: {attempts:,})")

# ===================================================================
# POST-PROCESSING: Update Customer Master with behavioural features
# ===================================================================
print("\n[Post] Computing customer behavioural features ...")

df_sales["datetime_parsed"] = pd.to_datetime(df_sales["datetime"])
ref_date = df_sales["datetime_parsed"].max()

cust_agg = df_sales.groupby("customer_id").agg(
    total_purchases=("transaction_id", "count"),
    total_spend=("total_amount", "sum"),
    avg_basket_size=("total_amount", "mean"),
    unique_visit_days=("datetime_parsed", lambda x: x.dt.date.nunique()),
    last_purchase_date=("datetime_parsed", "max"),
    unique_products=("product_id", "nunique"),         # Feature Engineering
    unique_stores=("store_id", "nunique"),              # Feature Engineering
    avg_qty_per_txn=("qty", "mean"),                    # Feature Engineering
    promo_usage_count=(
        "promotion_id",
        lambda x: x.notna().sum()
    ),
).reset_index()

cust_agg["days_since_last_purchase"] = (
    (ref_date - cust_agg["last_purchase_date"]).dt.days
)
cust_agg["purchase_frequency"] = (
    cust_agg["total_purchases"] / cust_agg["unique_visit_days"].clip(lower=1)
).round(2)
cust_agg["promo_sensitivity"] = (
    cust_agg["promo_usage_count"] / cust_agg["total_purchases"].clip(lower=1)
).round(4)

# Merge back into customer master
feature_cols = [
    "customer_id", "total_purchases", "total_spend", "avg_basket_size",
    "purchase_frequency", "days_since_last_purchase", "unique_products",
    "unique_stores", "avg_qty_per_txn", "promo_sensitivity",
]
cust_features = cust_agg[feature_cols].copy()
cust_features["total_spend"] = cust_features["total_spend"].round(2)
cust_features["avg_basket_size"] = cust_features["avg_basket_size"].round(2)
cust_features["avg_qty_per_txn"] = cust_features["avg_qty_per_txn"].round(2)

# Drop placeholder columns and merge real features
df_customers = df_customers.drop(columns=[
    "total_purchases", "total_spend", "avg_basket_size",
    "purchase_frequency", "days_since_last_purchase",
])
df_customers = df_customers.merge(cust_features, on="customer_id", how="left")
df_customers = df_customers.fillna({
    "total_purchases": 0,
    "total_spend": 0.0,
    "avg_basket_size": 0.0,
    "purchase_frequency": 0.0,
    "days_since_last_purchase": -1,
    "unique_products": 0,
    "unique_stores": 0,
    "avg_qty_per_txn": 0.0,
    "promo_sensitivity": 0.0,
})

# Clean up helper column from sales
df_sales = df_sales.drop(columns=["datetime_parsed"])

# ===================================================================
# SAVE ALL FILES
# ===================================================================
print("\n[Save] Writing CSV files ...")

files = {
    "01_product_master.csv": df_products,
    "02_store_master.csv": df_stores,
    "03_customer_master.csv": df_customers,
    "04_warehouse_master.csv": df_warehouses,
    "05_promotion_master.csv": df_promotions,
    "06_purchasing_order.csv": df_po,
    "07_stock_movement.csv": df_stock,
    "08_sales_transaction.csv": df_sales,
}

for filename, df in files.items():
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [OK] {filename:40s} -> {len(df):>8,} rows  x  {len(df.columns):>3} cols")

# ===================================================================
# DATA QUALITY REPORT
# ===================================================================
print("\n" + "=" * 70)
print("DATA QUALITY & CONSISTENCY REPORT")
print("=" * 70)

# 1. Referential Integrity - Sales -> Product
sales_products = set(df_sales["product_id"].unique())
master_products = set(df_products["product_id"].unique())
orphan_products = sales_products - master_products
print(f"\n[Check] Sales product_id in Product Master  : "
      f"{'PASS' if not orphan_products else f'FAIL ({len(orphan_products)} orphans)'}")

# 2. Referential Integrity - Sales -> Store
sales_stores = set(df_sales["store_id"].unique())
master_stores = set(df_stores["store_id"].unique())
orphan_stores = sales_stores - master_stores
print(f"[Check] Sales store_id in Store Master      : "
      f"{'PASS' if not orphan_stores else f'FAIL ({len(orphan_stores)} orphans)'}")

# 3. Referential Integrity - Sales -> Customer
sales_custs = set(df_sales["customer_id"].unique())
master_custs = set(df_customers["customer_id"].unique())
orphan_custs = sales_custs - master_custs
print(f"[Check] Sales customer_id in Customer Master: "
      f"{'PASS' if not orphan_custs else f'FAIL ({len(orphan_custs)} orphans)'}")

# 4. Promotion date consistency
if len(df_sales[df_sales["promotion_id"].notna()]) > 0:
    promo_sales = df_sales[df_sales["promotion_id"].notna()].copy()
    promo_sales = promo_sales.merge(
        df_promotions[["promotion_id", "product_id", "start_date", "end_date"]],
        on=["promotion_id", "product_id"],
        how="left",
    )
    promo_sales["sale_dt"] = pd.to_datetime(promo_sales["datetime"])
    promo_sales["start_dt"] = pd.to_datetime(promo_sales["start_date"])
    promo_sales["end_dt"] = pd.to_datetime(promo_sales["end_date"])
    date_violations = promo_sales[
        (promo_sales["sale_dt"].dt.date < promo_sales["start_dt"].dt.date)
        | (promo_sales["sale_dt"].dt.date > promo_sales["end_dt"].dt.date)
    ]
    print(f"[Check] Promo date consistency              : "
          f"{'PASS' if len(date_violations) == 0 else f'FAIL ({len(date_violations)} violations)'}")

# 5. PO date ordering
po_dates_ok = (
    (pd.to_datetime(df_po["manufacturing_date"]) <= pd.to_datetime(df_po["po_date"]))
    & (pd.to_datetime(df_po["po_date"]) <= pd.to_datetime(df_po["arrival_date"]))
    & (pd.to_datetime(df_po["arrival_date"]) <= pd.to_datetime(df_po["expire_date"]))
).all()
print(f"[Check] PO date ordering (mfg<=po<=arr<=exp): "
      f"{'PASS' if po_dates_ok else 'FAIL'}")

# Summary statistics
print(f"\n{'-' * 70}")
print(f"Total Products   : {NUM_PRODUCTS:>8,}")
print(f"Total Stores     : {NUM_STORES:>8,}")
print(f"Total Warehouses : {NUM_WAREHOUSES:>8,}")
print(f"Total Customers  : {NUM_CUSTOMERS:>8,}")
print(f"Total Promotions : {len(df_promotions):>8,} (rows, multi-product)")
print(f"Total POs        : {NUM_PO:>8,}")
print(f"Total Stock Moves: {len(df_stock):>8,}")
print(f"Total Sales      : {len(df_sales):>8,}")
print(f"Date Range       : {DATE_START.strftime('%Y-%m-%d')} -> {DATE_END.strftime('%Y-%m-%d')}")
print(f"Revenue (Total)  : THB {df_sales['total_amount'].sum():>15,.2f}")
print(f"Avg Basket Size  : THB {df_sales['total_amount'].mean():>10,.2f}")
print(f"{'-' * 70}")
print(f"\n[DONE] All files saved to ./{OUTPUT_DIR}/")
print("       Ready for ML pipeline (Revenue Maximization)!\n")
