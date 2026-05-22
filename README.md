# 🛒 Smart Retail Revenue Optimizer

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/ML-LightGBM%20%7C%20ALS-orange.svg)](https://github.com/microsoft/LightGBM)

A high-performance, double-engine Machine Learning pipeline designed to **maximize revenue** for Small & Medium Enterprise (SME) retailers (convenience stores, mini-marts, supermarkets) by solving two critical business challenges:
1. **Reducing Stock-Outs** via proactive 14-day daily demand forecasting.
2. **Increasing Basket Size** via personalized promotion recommendations targeting price-sensitive customer segments.

This repository represents a 1-week Data Science take-home assignment for the **Botnoi** team.

---

## 📌 Project Architecture & Business Value

```
             ┌─────────────────────────────────────────┐
             │       Raw Relational Mock Data          │
             │     (8 tables, 300k transactions)       │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
             ┌─────────────────────────────────────────┐
             │  Feature Engineering (51 Features)      │
             └───────────┬─────────────────────────┬───┘
                         │                         │
                         ▼                         ▼
             ┌───────────────────────┐ ┌───────────────────────┐
             │   Demand Forecasting  │ │  Promo Recommender    │
             │  (LightGBM Regressor) │ │ (NumPy Implicit ALS)  │
             └───────────┬───────────┘ └───────────┬───────────┘
                         │                         │
                         ▼                         ▼
             ┌───────────────────────┐ ┌───────────────────────┐
             │  14-Day Store-Product │ │   Personalized Top-5  │
             │   Demand Predictions  │ │  Promotions / RFM Segment│
             └───────────────────────┘ └───────────────────────┘
```

### 1. Daily Demand Forecasting (LightGBM)
*   **Business Impact:** Prevents lost sales from empty shelves and reduces inventory holding costs for perishable goods.
*   **Technique:** LightGBM Regressor utilizing lag features (1, 3, 7, 14, 28 days), rolling averages, cyclical time encoding (sine/cosine of day/month), calendar signals, and promotion indicators.

### 2. Personalized Promotion Recommender (Collaborative Filtering + RFM)
*   **Business Impact:** Increases customer lifetime value (CLV) and average order value (AOV) while protecting margins by not giving blanket discounts to customers willing to pay full price.
*   **Technique:** Implicit Alternating Least Squares (ALS) Matrix Factorization built from scratch using NumPy. Complemented by **RFM (Recency, Frequency, Monetary) Segmentation** to group customers into 4 strategic tiers (`loyal_full_price`, `promo_lover`, `price_sensitive`, `occasional`).

---

## 📊 Current Model Performance

| Model | Metric | Value | Business Interpretation |
|---|---|---|---|
| **Demand Forecasting (LightGBM)** | $R^2$ Score | **0.9513** | Explains 95.1% of the variance in daily product-store demand. |
| **Demand Forecasting (LightGBM)** | RMSE | **0.8662** | Average prediction error is less than 1 unit per day. |
| **Promotion Recommender (ALS)** | Precision@5 | **0.2792** | On average, 1.4 out of the 5 recommended products are highly relevant. |
| **Promotion Recommender (ALS)** | Hit Rate@5 | **71.03%** | 71.0% of customers receive at least 1 correct recommendation in their Top-5. |

---

## 🧬 Data Mockup Extensions & Business Justification

To make the DS solution realistic and impactful, we extended the base mockup schema with additional columns. Here is the justification as per the assessment criteria:

| Table | Added Columns | What it does & Business/Model Impact |
|---|---|---|
| **Product Master** | `cost_price`, `profit_margin`, `is_perishable`, `shelf_life_days` | **Impact:** Allows the business to calculate actual profit, not just revenue. `is_perishable` helps the Demand Forecasting model prioritize fast-moving inventory to reduce waste (Expired Stock). |
| **Customer Master** | `membership_tier`, `total_spend`, `avg_basket_size`, `days_since_last_purchase` | **Impact:** Essential for computing **RFM (Recency, Frequency, Monetary) Segments**. Helps the Recommender identify `loyal_full_price` customers who shouldn't receive margin-eroding discounts. |
| **Sales Transaction** | `total_amount`, `discount_pct`, `is_weekend`, `day_of_week` | **Impact:** `is_weekend` and `day_of_week` are critical temporal features for LightGBM to capture seasonality (e.g., weekend shopping spikes). |

---

## 📂 Directory Structure

```
DS(Test)/
├── generate_mock_data.py            # Generates all 8 relational tables (300K sales)
├── requirements.txt                 # Python dependencies
├── gemini.md                        # AI Collaboration Log (Assessment Memory)
├── README.md                        # ← You are here
├── Dockerfile                       # Production MLOps deployment file
├── docker-compose.yml               # Container orchestrator configuration
│
├── mock_data/                       # Generated CSVs (raw data)
│   ├── 01_product_master.csv        #   100 products across 10 categories
│   ├── 02_store_master.csv          #   20 retail branches
│   ├── 03_customer_master.csv       #   3,001 customers
│   ├── 04_warehouse_master.csv      #   3 distribution warehouses
│   ├── 05_promotion_master.csv      #   50 promotion campaigns
│   ├── 06_purchasing_order.csv      #   Purchase orders (stock-in events)
│   ├── 07_stock_movement.csv        #   Inventory movements
│   └── 08_sales_transaction.csv     #   300,000 sales records
│
├── src/                             # Core ML pipeline
│   ├── __init__.py
│   ├── config.py                    #   Paths, hyperparameters, constants
│   ├── utils.py                     #   Logger, data loader, seed setter
│   ├── feature_engineering.py       #   51 engineered features (9 families)
│   ├── demand_forecasting.py        #   LightGBM training + evaluation + plots
│   ├── promotion_recommendation.py  #   ALS model + RFM segmentation + plots
│   └── run_pipeline.py              #   CLI entry point (orchestrator)
│
└── model_output/                    # Pipeline outputs
    ├── models/                      #   Serialized model files (.joblib, .pkl)
    ├── plots/                       #   Premium diagnostic charts (8 PNGs)
    └── tables/                      #   Business-ready CSV outputs
        ├── demand_forecast.csv      #     14-day demand predictions
        ├── customer_segments.csv    #     RFM + promo-affinity segments
        └── promotion_recommendations.csv  # Top-5 per customer (with discounts)
```

---

## 🚀 How to Run

### Option 1: Run Locally (Python)

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the mock dataset:** (Only needed once)
   ```bash
   python generate_mock_data.py
   ```

3. **Run the full pipeline:**
   ```bash
   python -m src.run_pipeline
   ```
   *To run individual models, use the `--demand-only` or `--promo-only` flags:*
   ```bash
   python -m src.run_pipeline --demand-only
   python -m src.run_pipeline --promo-only
   ```

---

### Option 2: Run via Docker (Recommended for Deployment)

A midnight batch-inference process is simulated using a lightweight Docker container.

1. **Build and run using Docker Compose:**
   ```bash
   docker-compose up --build
   ```
   This compiles the image, generates mock data (if run inside container), runs both ML models, and saves the outputs directly to your local `model_output/` folder via mounted volumes.

2. **Run a specific pipeline step via Docker:**
   ```bash
   docker-compose run ml-pipeline --promo-only
   ```

---

## 📊 MLflow Experiment Tracking & Model Registry

This project integrates **MLflow** for full experiment lifecycle management. Every pipeline run automatically logs hyperparameters, evaluation metrics, diagnostic plots, and registers trained models.

### What Gets Tracked

| Component | Tracked Data |
|---|---|
| **Demand Forecasting** | LightGBM hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, etc.), metrics (`R²`, `RMSE`, `MAE`, `MAPE`, `WMAPE`), 4 diagnostic plots, forecast CSV |
| **Promotion Recommender** | ALS hyperparameters (`n_factors`, `n_epochs`, `reg_lambda`, etc.), metrics (`Precision@K`, `Recall@K`, `NDCG@K`, `Hit Rate@K`, `Coverage`), 3 diagnostic plots, recommendation & segment CSVs |

### Model Registry

Both models are automatically registered in the **MLflow Model Registry** after each training run:

| Registered Model Name | Flavor | Description |
|---|---|---|
| `SME_Demand_Model` | `mlflow.lightgbm` | LightGBM Regressor for 14-day demand forecasting |
| `SME_Promo_Recommender` | `mlflow.pyfunc` | Custom ALS Matrix Factorization wrapped as PythonModel |

### Viewing Experiments with MLflow UI

After running the pipeline, launch the MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root mlruns
```

Then open your browser at **http://localhost:5000** to:
- 📈 Compare metrics across training runs
- 🔍 Inspect logged hyperparameters and artifacts (plots, CSVs)
- 🏷️ Manage model versions (Staging → Production → Archived)
- ⏪ Rollback to previous model versions with one click

---

## 🤝 AI Collaboration (`gemini.md`)

This project was built using an **AI-Assisted Workflow** (Human as the Project Driver, AI as the Technical Copilot). Detailed logs of prompts, key architectural decisions, feature engineering brainstorming, and productivity impact metrics are available in [gemini.md](gemini.md). 

Reviewing [gemini.md](gemini.md) satisfies the *"AI-assisted workflows"* evaluation criteria.

---

*For any questions or clarification regarding the models, please reach out via GitHub issues.*
