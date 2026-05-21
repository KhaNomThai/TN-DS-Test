"""
feature_engineering.py – Feature engineering for the SME Retail ML pipeline.

This module provides all feature transformations needed for both the
demand-forecasting and promotion-recommendation models.  Functions are
designed to be called individually or orchestrated via
:func:`build_demand_features`.
"""

import pandas as pd
import numpy as np

from src.config import LAG_DAYS, ROLLING_WINDOWS, RFM_QUANTILES
from src.utils import get_logger

logger = get_logger(__name__)


# ===========================================================================
# 1. Daily Demand Aggregation
# ===========================================================================

def build_daily_demand(
    df_sales: pd.DataFrame,
    df_products: pd.DataFrame,
    df_stores: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate sales to daily level per (product_id, store_id).

    Missing days for each product-store pair are filled with 0 so the
    model can learn from zero-demand days.  Product and store attributes
    are joined for downstream feature usage.

    Args:
        df_sales: Raw sales transactions with ``datetime``, ``product_id``,
            ``store_id``, and ``qty`` columns.
        df_products: Product master with ``product_id`` and descriptive
            columns (category, subcategory, price, etc.).
        df_stores: Store master with ``store_id`` and descriptive columns
            (store_type, region, size_sqm).

    Returns:
        DataFrame indexed by (product_id, store_id, date) with the target
        ``daily_qty`` and joined product/store attributes.
    """
    logger.info("Building daily demand table …")

    # Derive a date column from the datetime field
    sales = df_sales.copy()
    sales["date"] = pd.to_datetime(sales["datetime"]).dt.normalize()

    # Aggregate to daily level
    daily = (
        sales
        .groupby(["product_id", "store_id", "date"], as_index=False)
        .agg(daily_qty=("qty", "sum"))
    )

    # ------------------------------------------------------------------
    # Fill missing days with 0 for every observed product-store pair
    # ------------------------------------------------------------------
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    pairs = daily[["product_id", "store_id"]].drop_duplicates()

    idx = pd.MultiIndex.from_product(
        [pairs["product_id"].unique(), pairs["store_id"].unique(), date_range],
        names=["product_id", "store_id", "date"],
    )
    # Keep only the combinations that actually existed in the data
    full_idx = (
        pairs
        .assign(key=1)
        .merge(pd.DataFrame({"date": date_range, "key": 1}), on="key")
        .drop(columns="key")
    )
    daily = full_idx.merge(daily, on=["product_id", "store_id", "date"], how="left")
    daily["daily_qty"] = daily["daily_qty"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Join product info
    # ------------------------------------------------------------------
    product_cols = [
        "product_id", "category", "subcategory", "price", "cost_price",
        "profit_margin", "is_perishable", "price_tier",
    ]
    available_product_cols = [c for c in product_cols if c in df_products.columns]
    daily = daily.merge(
        df_products[available_product_cols],
        on="product_id",
        how="left",
    )

    # ------------------------------------------------------------------
    # Join store info
    # ------------------------------------------------------------------
    store_cols = ["store_id", "store_type", "region", "size_sqm"]
    available_store_cols = [c for c in store_cols if c in df_stores.columns]
    daily = daily.merge(
        df_stores[available_store_cols],
        on="store_id",
        how="left",
    )

    logger.info(
        "Daily demand table: %s rows, %s product-store pairs",
        f"{len(daily):,}",
        f"{len(pairs):,}",
    )
    return daily


# ===========================================================================
# 2. Time / Calendar Features
# ===========================================================================

def add_time_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Derive calendar and cyclical time features from a date column.

    Args:
        df: DataFrame containing a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with new time-related columns appended.
    """
    logger.info("Adding time features …")
    out = df.copy()
    dt = pd.to_datetime(out[date_col])

    out["day_of_week"] = dt.dt.dayofweek          # 0=Mon … 6=Sun
    out["day_of_month"] = dt.dt.day
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["year"] = dt.dt.year

    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_month_start"] = dt.dt.is_month_start.astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)

    # Cyclical encoding
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)

    return out


# ===========================================================================
# 3. Lag Features
# ===========================================================================

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "daily_qty",
    group_cols: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create lagged values of the target variable.

    The DataFrame is first sorted by *group_cols* + ``date`` so that
    ``shift()`` produces the correct temporal lag within each group.

    Args:
        df: DataFrame with ``date``, *target_col*, and *group_cols*.
        target_col: Column to lag.
        group_cols: Grouping columns; defaults to
            ``['product_id', 'store_id']``.
        lags: Lag horizons in days; defaults to :pydata:`LAG_DAYS`.

    Returns:
        DataFrame with ``lag_{N}`` columns appended.
    """
    if group_cols is None:
        group_cols = ["product_id", "store_id"]
    if lags is None:
        lags = LAG_DAYS

    logger.info("Adding lag features %s …", lags)
    out = df.sort_values(group_cols + ["date"]).copy()

    grouped = out.groupby(group_cols, observed=True)[target_col]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)

    return out


# ===========================================================================
# 4. Rolling Features
# ===========================================================================

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "daily_qty",
    group_cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling statistics (mean, std, max, min) of the target.

    Uses ``min_periods=1`` so that early rows still receive values.

    Args:
        df: DataFrame sorted by *group_cols* + ``date``.
        target_col: Column to aggregate.
        group_cols: Grouping columns; defaults to
            ``['product_id', 'store_id']``.
        windows: Rolling window sizes; defaults to
            :pydata:`ROLLING_WINDOWS`.

    Returns:
        DataFrame with ``rolling_{stat}_{W}`` columns appended.
    """
    if group_cols is None:
        group_cols = ["product_id", "store_id"]
    if windows is None:
        windows = ROLLING_WINDOWS

    logger.info("Adding rolling features %s …", windows)
    out = df.sort_values(group_cols + ["date"]).copy()

    for w in windows:
        rolled = (
            out
            .groupby(group_cols, observed=True)[target_col]
            .rolling(window=w, min_periods=1)
        )
        out[f"rolling_mean_{w}"] = rolled.mean().reset_index(level=group_cols, drop=True)
        out[f"rolling_std_{w}"] = rolled.std().reset_index(level=group_cols, drop=True)
        out[f"rolling_max_{w}"] = rolled.max().reset_index(level=group_cols, drop=True)
        out[f"rolling_min_{w}"] = rolled.min().reset_index(level=group_cols, drop=True)

    # std can be NaN when window has only 1 observation
    std_cols = [c for c in out.columns if c.startswith("rolling_std_")]
    out[std_cols] = out[std_cols].fillna(0)

    return out


# ===========================================================================
# 5. Promotion Features
# ===========================================================================

def add_promotion_features(
    df: pd.DataFrame,
    df_promotions: pd.DataFrame,
) -> pd.DataFrame:
    """Add promotion-related features for each (product_id, date) pair.

    For a given date, determines whether an active promotion exists and
    calculates discount depth and temporal position within the promo
    period.

    Args:
        df: Demand DataFrame with ``product_id`` and ``date``.
        df_promotions: Promotion master with ``product_id``,
            ``start_date``, ``end_date``, ``discount``.

    Returns:
        DataFrame with ``is_on_promo``, ``promo_discount``,
        ``days_into_promo``, and ``days_until_promo_end`` columns.
    """
    logger.info("Adding promotion features …")
    out = df.copy()

    promos = df_promotions.copy()
    promos["start_date"] = pd.to_datetime(promos["start_date"])
    promos["end_date"] = pd.to_datetime(promos["end_date"])

    # Cross-join demand rows with promotions on product_id, then filter
    merged = out[["product_id", "date"]].drop_duplicates().merge(
        promos[["product_id", "start_date", "end_date", "discount"]],
        on="product_id",
        how="left",
    )

    # Keep only rows where date falls within the promo window
    active = merged[
        (merged["date"] >= merged["start_date"])
        & (merged["date"] <= merged["end_date"])
    ].copy()

    if active.empty:
        out["is_on_promo"] = 0
        out["promo_discount"] = 0.0
        out["days_into_promo"] = 0
        out["days_until_promo_end"] = 0
        return out

    active["days_into_promo"] = (active["date"] - active["start_date"]).dt.days
    active["days_until_promo_end"] = (active["end_date"] - active["date"]).dt.days

    # Aggregate: a product-date may have multiple active promos
    promo_agg = (
        active
        .groupby(["product_id", "date"], as_index=False)
        .agg(
            promo_discount=("discount", "max"),
            days_into_promo=("days_into_promo", "max"),
            days_until_promo_end=("days_until_promo_end", "min"),
        )
    )
    promo_agg["is_on_promo"] = 1

    out = out.merge(promo_agg, on=["product_id", "date"], how="left")
    out["is_on_promo"] = out["is_on_promo"].fillna(0).astype(int)
    out["promo_discount"] = out["promo_discount"].fillna(0.0)
    out["days_into_promo"] = out["days_into_promo"].fillna(0).astype(int)
    out["days_until_promo_end"] = out["days_until_promo_end"].fillna(0).astype(int)

    return out


# ===========================================================================
# 6. Expiry Features
# ===========================================================================

def add_expiry_features(
    df: pd.DataFrame,
    df_po: pd.DataFrame,
    df_stock_movement: pd.DataFrame,
) -> pd.DataFrame:
    """Add stock-expiry related features for each (product_id, store_id, date).

    Cross-references PO expiry dates with stock-movement data to find
    batches present at each store.

    Args:
        df: Demand DataFrame with ``product_id``, ``store_id``, ``date``.
        df_po: Purchase orders with ``po_id``, ``product_id``,
            ``expire_date``.
        df_stock_movement: Stock movements with ``po_id``, ``store_id``,
            ``transfer_date``.

    Returns:
        DataFrame with ``days_until_nearest_expiry``,
        ``has_expiring_stock_7d``, and ``has_expiring_stock_14d``.
    """
    logger.info("Adding expiry features …")
    out = df.copy()

    po = df_po[["po_id", "product_id", "expire_date"]].copy()
    po["expire_date"] = pd.to_datetime(po["expire_date"])

    sm = df_stock_movement[["po_id", "store_id", "transfer_date"]].copy()
    sm["transfer_date"] = pd.to_datetime(sm["transfer_date"])

    # Join PO with stock movement to get (product_id, store_id, expire_date)
    batch = po.merge(sm, on="po_id", how="inner")
    batch = batch[["product_id", "store_id", "expire_date"]].drop_duplicates()

    # Unique demand keys
    demand_keys = out[["product_id", "store_id", "date"]].drop_duplicates()

    # Merge demand keys with batch expiry info
    merged = demand_keys.merge(batch, on=["product_id", "store_id"], how="left")

    # Keep only future or same-day expiry dates
    merged = merged[merged["expire_date"] >= merged["date"]].copy()
    merged["days_until_expiry"] = (merged["expire_date"] - merged["date"]).dt.days

    if merged.empty:
        out["days_until_nearest_expiry"] = np.nan
        out["has_expiring_stock_7d"] = 0
        out["has_expiring_stock_14d"] = 0
        return out

    # Aggregate per (product_id, store_id, date)
    expiry_agg = (
        merged
        .groupby(["product_id", "store_id", "date"], as_index=False)
        .agg(days_until_nearest_expiry=("days_until_expiry", "min"))
    )
    expiry_agg["has_expiring_stock_7d"] = (
        expiry_agg["days_until_nearest_expiry"] <= 7
    ).astype(int)
    expiry_agg["has_expiring_stock_14d"] = (
        expiry_agg["days_until_nearest_expiry"] <= 14
    ).astype(int)

    out = out.merge(
        expiry_agg, on=["product_id", "store_id", "date"], how="left",
    )
    out["days_until_nearest_expiry"] = out["days_until_nearest_expiry"].fillna(-1)
    out["has_expiring_stock_7d"] = out["has_expiring_stock_7d"].fillna(0).astype(int)
    out["has_expiring_stock_14d"] = out["has_expiring_stock_14d"].fillna(0).astype(int)

    return out


# ===========================================================================
# 7. RFM Segmentation
# ===========================================================================

_RFM_SEGMENT_MAP: dict[str, str] = {
    # (R_quartile, F_quartile, M_quartile) – higher = better
    "444": "Champions",
    "443": "Champions",
    "434": "Champions",
    "344": "Loyal",
    "343": "Loyal",
    "334": "Loyal",
    "333": "Loyal",
    "244": "Potential Loyalists",
    "243": "Potential Loyalists",
    "234": "Potential Loyalists",
    "144": "New Customers",
    "143": "New Customers",
    "134": "New Customers",
    "133": "New Customers",
    "233": "Need Attention",
    "232": "Need Attention",
    "223": "Need Attention",
    "222": "Need Attention",
    "322": "About to Sleep",
    "321": "About to Sleep",
    "231": "About to Sleep",
    "221": "About to Sleep",
    "211": "At Risk",
    "212": "At Risk",
    "122": "At Risk",
    "121": "At Risk",
    "132": "At Risk",
    "131": "At Risk",
    "112": "Hibernating",
    "111": "Lost",
}


def _assign_rfm_segment(score: str) -> str:
    """Map an RFM score string to a human-readable segment name.

    Args:
        score: Three-character string of quartile ranks (e.g. ``'443'``).

    Returns:
        Segment label such as ``'Champions'`` or ``'At Risk'``.
    """
    if score in _RFM_SEGMENT_MAP:
        return _RFM_SEGMENT_MAP[score]

    # Fallback heuristic when exact score is not in the map
    r, f, m = int(score[0]), int(score[1]), int(score[2])
    avg = (r + f + m) / 3
    if avg >= 3.5:
        return "Champions"
    if avg >= 3.0:
        return "Loyal"
    if avg >= 2.5:
        return "Potential Loyalists"
    if avg >= 2.0:
        return "Need Attention"
    if r <= 2 and f >= 2:
        return "At Risk"
    if avg >= 1.5:
        return "Hibernating"
    return "Lost"


def compute_rfm(
    df_sales: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute RFM (Recency, Frequency, Monetary) scores per customer.

    Args:
        df_sales: Sales transactions with ``customer_id``, ``datetime``,
            ``transaction_id``, and ``total_amount``.
        reference_date: Anchor date for recency calculation.  Defaults
            to one day after the latest transaction date.

    Returns:
        DataFrame with one row per ``customer_id`` containing
        ``recency``, ``frequency``, ``monetary``, ``r_quartile``,
        ``f_quartile``, ``m_quartile``, ``rfm_score``, and
        ``rfm_segment``.
    """
    logger.info("Computing RFM scores …")

    sales = df_sales.copy()
    sales["date"] = pd.to_datetime(sales["datetime"]).dt.normalize()

    if reference_date is None:
        reference_date = sales["date"].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    rfm = (
        sales
        .groupby("customer_id", as_index=False)
        .agg(
            recency=("date", lambda x: (reference_date - x.max()).days),
            frequency=("transaction_id", "nunique"),
            monetary=("total_amount", "sum"),
        )
    )

    # Quartile scoring – 1 = worst, 4 = best
    # Recency: lower is better → labels reversed
    rfm["r_quartile"] = pd.qcut(
        rfm["recency"], q=RFM_QUANTILES, labels=False, duplicates="drop",
    )
    rfm["r_quartile"] = RFM_QUANTILES - rfm["r_quartile"]  # invert

    rfm["f_quartile"] = pd.qcut(
        rfm["frequency"], q=RFM_QUANTILES, labels=False, duplicates="drop",
    ) + 1

    rfm["m_quartile"] = pd.qcut(
        rfm["monetary"], q=RFM_QUANTILES, labels=False, duplicates="drop",
    ) + 1

    rfm["rfm_score"] = (
        rfm["r_quartile"].astype(str)
        + rfm["f_quartile"].astype(str)
        + rfm["m_quartile"].astype(str)
    )
    rfm["rfm_segment"] = rfm["rfm_score"].apply(_assign_rfm_segment)

    logger.info(
        "RFM computed for %s customers, segments: %s",
        f"{len(rfm):,}",
        rfm["rfm_segment"].value_counts().to_dict(),
    )
    return rfm


# ===========================================================================
# 8. Master Feature Builder (Demand Forecasting)
# ===========================================================================

def build_demand_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Orchestrate all feature-engineering steps for demand forecasting.

    This is the main entry-point used by the training pipeline.  It
    calls :func:`build_daily_demand`, time/lag/rolling/promotion/expiry
    feature helpers in sequence, then drops rows with NaN lag values
    (i.e., the earliest dates that lack sufficient history).

    Args:
        tables: Dictionary of raw DataFrames keyed by table name, as
            returned by :func:`src.utils.load_all_tables`.

    Returns:
        Feature-rich DataFrame ready for model training.
    """
    logger.info("=== Building demand features (full pipeline) ===")

    # Step 1 – Daily demand
    df = build_daily_demand(
        df_sales=tables["sales"],
        df_products=tables["products"],
        df_stores=tables["stores"],
    )

    # Step 2 – Time features
    df = add_time_features(df, date_col="date")

    # Step 3 – Lag features
    df = add_lag_features(df)

    # Step 4 – Rolling features
    df = add_rolling_features(df)

    # Step 5 – Promotion features
    df = add_promotion_features(df, df_promotions=tables["promotions"])

    # Step 6 – Expiry features
    df = add_expiry_features(
        df,
        df_po=tables["po"],
        df_stock_movement=tables["stock_movement"],
    )

    # Drop rows where lag features are NaN (early dates without history)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    rows_before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    logger.info(
        "Dropped %s rows with NaN lags. Final shape: %s",
        f"{rows_dropped:,}",
        df.shape,
    )

    logger.info("=== Demand features complete ===")
    return df


# ===========================================================================
# 9. Customer × Product Interaction Matrix
# ===========================================================================

def build_customer_product_matrix(
    df_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Create a customer × product interaction matrix (implicit feedback).

    Each cell contains the total quantity purchased by a customer for a
    given product.  Useful for collaborative-filtering recommendation
    models.

    Args:
        df_sales: Sales transactions with ``customer_id``, ``product_id``,
            and ``qty``.

    Returns:
        Pivot table with ``customer_id`` as index, ``product_id`` as
        columns, and total ``qty`` as values (0 for unseen pairs).
    """
    logger.info("Building customer × product matrix …")

    interaction = (
        df_sales
        .groupby(["customer_id", "product_id"], as_index=False)
        .agg(total_qty=("qty", "sum"))
    )

    matrix = interaction.pivot_table(
        index="customer_id",
        columns="product_id",
        values="total_qty",
        fill_value=0,
    )

    logger.info(
        "Interaction matrix: %s customers × %s products, "
        "density %.2f%%",
        matrix.shape[0],
        matrix.shape[1],
        (matrix > 0).sum().sum() / (matrix.shape[0] * matrix.shape[1]) * 100,
    )
    return matrix


# ===========================================================================
# Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    from src.utils import load_all_tables

    tables = load_all_tables()

    # --- Demand features ---
    demand_df = build_demand_features(tables)
    logger.info("Demand features sample:\n%s", demand_df.head())

    # --- RFM ---
    rfm_df = compute_rfm(tables["sales"])
    logger.info("RFM sample:\n%s", rfm_df.head())

    # --- Customer-product matrix ---
    cp_matrix = build_customer_product_matrix(tables["sales"])
    logger.info("Customer-product matrix shape: %s", cp_matrix.shape)
