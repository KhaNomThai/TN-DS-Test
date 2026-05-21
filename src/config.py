"""
config.py – Shared configuration for the ML pipeline.
"""

import os
from pathlib import Path

# ===========================================================================
# Paths
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "mock_data"
OUTPUT_DIR = PROJECT_ROOT / "model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# Data file mapping
DATA_FILES = {
    "products": DATA_DIR / "01_product_master.csv",
    "stores": DATA_DIR / "02_store_master.csv",
    "customers": DATA_DIR / "03_customer_master.csv",
    "warehouses": DATA_DIR / "04_warehouse_master.csv",
    "promotions": DATA_DIR / "05_promotion_master.csv",
    "po": DATA_DIR / "06_purchasing_order.csv",
    "stock_movement": DATA_DIR / "07_stock_movement.csv",
    "sales": DATA_DIR / "08_sales_transaction.csv",
}

# ===========================================================================
# Model Parameters
# ===========================================================================

# Demand Forecasting
FORECAST_HORIZON_DAYS = 14          # ทำนายล่วงหน้า 14 วัน
LAG_DAYS = [1, 3, 7, 14, 28]       # Lag features
ROLLING_WINDOWS = [7, 14, 28]      # Rolling statistics
TEST_SPLIT_DAYS = 60               # ใช้ 60 วันสุดท้ายเป็น test set

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

# Promotion Recommendation
RFM_QUANTILES = 4                   # จำนวน quantile สำหรับ RFM
TOP_K_RECOMMENDATIONS = 5          # จำนวนโปรโมชั่นที่แนะนำต่อลูกค้า
MF_N_FACTORS = 30                  # จำนวน latent factors
MF_LEARNING_RATE = 0.01
MF_N_EPOCHS = 50
MF_REG_LAMBDA = 0.1

# Random seed
SEED = 42
