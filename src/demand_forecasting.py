"""
demand_forecasting.py – Demand Forecasting model using LightGBM.

This module implements the end-to-end demand forecasting pipeline for the
SME Retail project.  It covers data splitting, categorical encoding, model
training with early stopping, multi-metric evaluation, visualisation, and
forward-looking forecast generation.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

from src.config import (
    FORECAST_HORIZON_DAYS,
    LGBM_PARAMS,
    MODELS_DIR,
    PLOTS_DIR,
    TABLES_DIR,
    SEED,
    TEST_SPLIT_DAYS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_DEMAND,
    MLFLOW_MODEL_NAME_DEMAND,
)
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

_ID_COLS = {"date", "product_id", "store_id"}
_TARGET_COL = "daily_qty"
_NON_FEATURE_COLS = _ID_COLS | {_TARGET_COL}

_CATEGORICAL_COLS = [
    "category",
    "subcategory",
    "store_type",
    "region",
    "price_tier",
    "brand",
]


# ===========================================================================
# 1. Train / Test Split
# ===========================================================================

def prepare_train_test_split(
    df_features: pd.DataFrame,
    test_days: int = TEST_SPLIT_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split.

    The last *test_days* calendar days of ``df_features`` are held out as
    the test set; everything before is used for training.

    Args:
        df_features: Feature DataFrame that must contain a ``date`` column.
        test_days: Number of trailing calendar days for the test set.

    Returns:
        Tuple of ``(df_train, df_test)`` sorted by date.
    """
    df = df_features.copy().sort_values("date").reset_index(drop=True)

    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    df_train = df[df["date"] <= cutoff_date].reset_index(drop=True)
    df_test = df[df["date"] > cutoff_date].reset_index(drop=True)

    logger.info(
        "Train/test split  |  cutoff=%s  |  train=%s rows  |  test=%s rows",
        cutoff_date.date(),
        f"{len(df_train):,}",
        f"{len(df_test):,}",
    )
    return df_train, df_test


# ===========================================================================
# 2. Feature Column Detection
# ===========================================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns, excluding IDs and target.

    Numeric columns are kept directly.  Known categorical columns that are
    present in *df* are also included (they should be label-encoded before
    modelling).

    Args:
        df: DataFrame whose columns will be inspected.

    Returns:
        Sorted list of feature column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_present = [c for c in _CATEGORICAL_COLS if c in df.columns]

    feature_cols = sorted(
        set(numeric_cols + cat_cols_present) - _NON_FEATURE_COLS
    )
    logger.info("Detected %d feature columns.", len(feature_cols))
    return feature_cols


# ===========================================================================
# 3. Categorical Encoding
# ===========================================================================

def encode_categoricals(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cat_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns.

    Encoders are fitted on *df_train* and applied to both *df_train* and
    *df_test*.  Categories that appear only in *df_test* are mapped to
    ``-1``.

    Args:
        df_train: Training DataFrame (will be modified in-place on a copy).
        df_test: Test DataFrame (same treatment).
        cat_cols: Categorical columns to encode.  Defaults to the
            intersection of ``_CATEGORICAL_COLS`` and the DataFrame columns.

    Returns:
        Tuple of ``(df_train_enc, df_test_enc, encoders)`` where
        *encoders* maps column name → fitted :class:`LabelEncoder`.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    if cat_cols is None:
        cat_cols = [c for c in _CATEGORICAL_COLS if c in df_train.columns]

    encoders: dict[str, LabelEncoder] = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))

        # Vectorized encoding: build a mapping dict, use .map()
        class_to_int = {cls: idx for idx, cls in enumerate(le.classes_)}
        df_test[col] = (
            df_test[col].astype(str).map(class_to_int).fillna(-1).astype(int)
        )
        encoders[col] = le
        logger.info(
            "Encoded '%s'  |  %d unique classes in train.", col, len(le.classes_)
        )

    return df_train, df_test, encoders


# ===========================================================================
# 4. Model Training
# ===========================================================================

def train_demand_model(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = _TARGET_COL,
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor with early stopping.

    A 10 % hold-out from *df_train* is used as the validation set for
    early-stopping (patience = 50 rounds).

    Args:
        df_train: Training data with features and target.
        feature_cols: Column names to use as predictors.
        target_col: Name of the target column.

    Returns:
        Fitted :class:`lgb.LGBMRegressor`.
    """
    set_global_seed(SEED)

    X = df_train[feature_cols]
    y = df_train[target_col]

    # 10 % chronological validation split
    val_size = max(int(len(X) * 0.1), 1)
    X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

    logger.info(
        "Training LightGBM  |  train=%s  |  val=%s  |  features=%d",
        f"{len(X_tr):,}",
        f"{len(X_val):,}",
        len(feature_cols),
    )

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = model.best_iteration_ if model.best_iteration_ else model.n_estimators
    logger.info("Training complete  |  best iteration = %d", best_iter)
    return model


# ===========================================================================
# 5. Model Evaluation
# ===========================================================================

def evaluate_model(
    model: lgb.LGBMRegressor,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = _TARGET_COL,
) -> dict[str, float]:
    """Evaluate the trained model on the test set.

    Metrics computed:
        * RMSE – Root Mean Squared Error
        * MAE  – Mean Absolute Error
        * MAPE – Mean Absolute Percentage Error (zeros excluded)
        * WMAPE – Weighted MAPE (weighted by actuals)
        * R²   – Coefficient of Determination

    Args:
        model: Fitted LightGBM model.
        df_test: Test DataFrame containing features and target.
        feature_cols: Feature column names.
        target_col: Name of the target column.

    Returns:
        Dictionary of metric name → value.
    """
    X_test = df_test[feature_cols]
    y_true = df_test[target_col].values
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE – exclude zeros to avoid division errors
    mask_nonzero = y_true != 0
    if mask_nonzero.sum() > 0:
        mape = float(
            np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
        )
    else:
        mape = float("nan")

    # Weighted MAPE
    total_actual = np.sum(np.abs(y_true))
    if total_actual > 0:
        wmape = float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)
    else:
        wmape = float("nan")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "wmape": wmape,
        "r2": r2,
    }

    logger.info("─── Evaluation Metrics ───")
    for name, value in metrics.items():
        logger.info("  %-8s: %.4f", name.upper(), value)

    return metrics


# ===========================================================================
# 6. Feature Importance
# ===========================================================================

def get_feature_importance(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract gain-based feature importance from the model.

    Args:
        model: Fitted LightGBM model.
        feature_cols: Feature column names used during training.
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns ``feature``, ``importance``,
        ``importance_pct``, sorted descending by importance.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    )
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["importance_pct"] = (
        df_imp["importance"] / df_imp["importance"].sum() * 100
    )
    return df_imp.head(top_n)


# ===========================================================================
# 7. Plotting
# ===========================================================================

def _apply_premium_style(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: str = "both",
) -> None:
    """Apply modern clean styling to a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDC3C7")
    ax.spines["bottom"].set_color("#BDC3C7")
    ax.tick_params(colors="#2C3E50", labelsize=10)

    if grid in ("both", "y"):
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#BDC3C7")
    if grid in ("both", "x"):
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#BDC3C7")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color="#2C3E50", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#2C3E50", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#2C3E50", labelpad=8)


def plot_results(
    model: lgb.LGBMRegressor,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = _TARGET_COL,
    output_dir: str | os.PathLike | None = None,
) -> None:
    """Generate and save diagnostic plots for the demand model.

    Plots created:
        1. Actual vs Predicted scatter
        2. Top-20 feature importance bar chart
        3. Residual distribution histogram
        4. Time-series comparison for the top-5 products

    Args:
        model: Fitted LightGBM model.
        df_test: Test DataFrame with features and target.
        feature_cols: Feature column names.
        target_col: Target column name.
        output_dir: Directory to save plots.  Defaults to
            :data:`src.config.PLOTS_DIR`.
    """
    if output_dir is None:
        output_dir = PLOTS_DIR
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    y_true = df_test[target_col].values
    y_pred = model.predict(df_test[feature_cols])
    residuals = y_true - y_pred

    # ── Plot 1: Actual vs Predicted ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#34495E", edgecolors="none", label="Predictions")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "--", color="#E74C3C", linewidth=1.5, label="Perfect Match")
    _apply_premium_style(ax, title="Actual vs Predicted Demand", xlabel="Actual Qty", ylabel="Predicted Qty")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved actual_vs_predicted.png")

    # ── Plot 2: Feature Importance ───────────────────────────────────────
    df_imp = get_feature_importance(model, feature_cols, top_n=20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.4, 0.8, len(df_imp)))
    bars = ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color=colors, height=0.6)

    # Add values at the end of the bars
    max_imp = df_imp["importance"].max()
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max_imp * 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}",
            ha="left",
            va="center",
            fontsize=8,
            color="#2C3E50",
        )

    _apply_premium_style(ax, title="Top-20 Feature Importance (Gain)", xlabel="Importance (Gain)", grid="x")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved feature_importance.png")

    # ── Plot 3: Residual Distribution ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=60, color="#5DADE2", edgecolor="#2E86C1", alpha=0.7, rwidth=0.85)
    ax.axvline(0, color="#E74C3C", linestyle="--", linewidth=1.5, label="Zero Bias")
    _apply_premium_style(ax, title="Residual Distribution", xlabel="Residual (Actual − Predicted)", ylabel="Frequency")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "residual_distribution.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved residual_distribution.png")

    # ── Plot 4: Time-Series for Top-5 Products ──────────────────────────
    if "date" in df_test.columns and "product_id" in df_test.columns:
        df_plot = df_test.copy()
        df_plot["predicted"] = y_pred
        df_plot["date"] = pd.to_datetime(df_plot["date"])

        # Pick top-5 products by total actual qty
        top_products = (
            df_plot.groupby("product_id")[target_col]
            .sum()
            .nlargest(5)
            .index.tolist()
        )

        fig, axes = plt.subplots(
            nrows=len(top_products), ncols=1, figsize=(14, 4 * len(top_products)),
            sharex=True,
        )
        if len(top_products) == 1:
            axes = [axes]

        for ax, pid in zip(axes, top_products):
            sub = (
                df_plot[df_plot["product_id"] == pid]
                .groupby("date")
                .agg({target_col: "sum", "predicted": "sum"})
                .sort_index()
            )
            # Smooth using 7-day rolling average to filter high-frequency daily noise
            sub_smoothed = sub.rolling(window=7, min_periods=1).mean()

            # Plot raw actual as faint background
            ax.plot(sub.index, sub[target_col], color="#BDC3C7", alpha=0.3, linewidth=0.8, label="Actual (Daily)")

            # Plot smoothed lines
            ax.plot(sub_smoothed.index, sub_smoothed[target_col], color="#1A5276", linewidth=2.0, label="Actual (7d Roll)")
            ax.plot(sub_smoothed.index, sub_smoothed["predicted"], color="#E67E22", linewidth=2.0, linestyle="--", label="Predicted (7d Roll)")

            _apply_premium_style(ax, title=f"Product: {pid}", ylabel="Daily Qty")
            ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")

        fig.suptitle("Time-Series Trends: Top-5 Products (Actual vs Predicted)", fontsize=14, fontweight="bold", color="#2C3E50", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(output_dir, "timeseries_top5.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved timeseries_top5.png")


# ===========================================================================
# 8. Forward-Looking Forecast
# ===========================================================================

def generate_forecast(
    model: lgb.LGBMRegressor,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    """Generate demand forecasts for future dates.

    For every unique ``(product_id, store_id)`` combination the model
    predicts ``daily_qty`` for each day in the forecast horizon.  The
    latest feature snapshot per combination is used as the basis (lag
    and rolling features are carried forward).

    A ``high_demand_flag`` column is set when predicted demand exceeds
    the 90th percentile across all predictions.

    Args:
        model: Fitted LightGBM model.
        df_features: Full feature DataFrame (used to derive the latest
            snapshot per product-store pair).
        feature_cols: Feature column names.
        horizon_days: Number of days into the future to forecast.

    Returns:
        DataFrame with columns ``product_id``, ``store_id``,
        ``forecast_date``, ``predicted_qty``, and ``high_demand_flag``.
    """
    logger.info("Generating %d-day demand forecast …", horizon_days)

    # Latest snapshot per (product_id, store_id)
    latest = (
        df_features.sort_values("date")
        .groupby(["product_id", "store_id"])
        .tail(1)
        .copy()
    )

    max_date = df_features["date"].max()
    forecast_dates = pd.date_range(
        start=max_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    records: list[pd.DataFrame] = []
    for fdate in forecast_dates:
        snapshot = latest.copy()
        snapshot["date"] = fdate

        # Update simple date-derived features if they exist
        if "day_of_week" in feature_cols:
            snapshot["day_of_week"] = fdate.dayofweek
        if "day_of_month" in feature_cols:
            snapshot["day_of_month"] = fdate.day
        if "month" in feature_cols and "month" in snapshot.columns:
            snapshot["month"] = fdate.month
        if "is_weekend" in feature_cols:
            snapshot["is_weekend"] = int(fdate.dayofweek >= 5)

        preds = model.predict(snapshot[feature_cols])
        snapshot["predicted_qty"] = np.clip(preds, 0, None)
        snapshot["forecast_date"] = fdate

        records.append(
            snapshot[["product_id", "store_id", "forecast_date", "predicted_qty"]]
        )

    df_forecast = pd.concat(records, ignore_index=True)

    # Flag high-demand items (> 90th percentile, with a safety minimum threshold of 0.05 to avoid flagging near-zero baseline predictions)
    threshold = df_forecast["predicted_qty"].quantile(0.90)
    effective_threshold = max(threshold, 0.05)
    df_forecast["high_demand_flag"] = (df_forecast["predicted_qty"] >= effective_threshold).astype(int)

    logger.info(
        "Forecast generated  |  %s rows  |  high-demand items: %s",
        f"{len(df_forecast):,}",
        f"{df_forecast['high_demand_flag'].sum():,}",
    )
    return df_forecast


# ===========================================================================
# 9. Main Orchestrator
# ===========================================================================

def run_demand_forecasting(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Run the full demand-forecasting pipeline.

    Steps:
        1. Build features via :func:`src.feature_engineering.build_demand_features`
        2. Train / test split
        3. Encode categoricals
        4. Train LightGBM model
        5. Evaluate on test set
        6. Generate diagnostic plots
        7. Save model with :mod:`joblib`
        8. Generate forward-looking forecast
        9. Log everything to MLflow & register model

    Args:
        tables: Dictionary of raw DataFrames keyed by table name
            (as returned by :func:`src.utils.load_all_tables`).

    Returns:
        Dictionary with keys ``model``, ``metrics``, ``forecast``,
        ``feature_cols``, and ``encoders``.
    """
    set_global_seed(SEED)

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_DEMAND)

    # 1. Build features ────────────────────────────────────────────────────
    from src.feature_engineering import build_demand_features  # noqa: E402

    logger.info("══════ Step 1/8: Building features ══════")
    df_features = build_demand_features(tables)

    # 2. Train / test split ────────────────────────────────────────────────
    logger.info("══════ Step 2/8: Train / test split ══════")
    df_train, df_test = prepare_train_test_split(df_features)

    # 3. Encode categoricals ───────────────────────────────────────────────
    logger.info("══════ Step 3/8: Encoding categoricals ══════")
    cat_cols = [c for c in _CATEGORICAL_COLS if c in df_train.columns]
    df_train, df_test, encoders = encode_categoricals(df_train, df_test, cat_cols)

    # 4. Detect features & train ───────────────────────────────────────────
    logger.info("══════ Step 4/8: Training model ══════")
    feature_cols = get_feature_columns(df_train)
    model = train_demand_model(df_train, feature_cols)

    # 5. Evaluate ──────────────────────────────────────────────────────────
    logger.info("══════ Step 5/8: Evaluating model ══════")
    metrics = evaluate_model(model, df_test, feature_cols)

    # 6. Plots ─────────────────────────────────────────────────────────────
    logger.info("══════ Step 6/8: Generating plots ══════")
    plot_results(model, df_test, feature_cols)

    # 7. Save model ────────────────────────────────────────────────────────
    logger.info("══════ Step 7/8: Saving model ══════")
    model_path = os.path.join(str(MODELS_DIR), "demand_model.joblib")
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # 8. Forecast ──────────────────────────────────────────────────────────
    logger.info("══════ Step 8/8: Generating forecast ══════")
    # Re-encode the full feature set for forecasting (vectorized)
    df_features_enc = df_features.copy()
    for col, le in encoders.items():
        class_to_int = {cls: idx for idx, cls in enumerate(le.classes_)}
        df_features_enc[col] = (
            df_features_enc[col].astype(str).map(class_to_int).fillna(-1).astype(int)
        )
    forecast = generate_forecast(model, df_features_enc, feature_cols)

    forecast_path = os.path.join(str(TABLES_DIR), "demand_forecast.csv")
    forecast.to_csv(forecast_path, index=False)
    logger.info("Forecast saved to %s", forecast_path)

    # ── 9. MLflow Tracking & Model Registry ───────────────────────────────
    logger.info("══════ Step 9: Logging to MLflow ══════")
    with mlflow.start_run(run_name="LightGBM-Demand-Forecast") as run:
        # Log hyperparameters
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("test_split_days", TEST_SPLIT_DAYS)
        mlflow.log_param("forecast_horizon_days", FORECAST_HORIZON_DAYS)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(df_train))
        mlflow.log_param("test_rows", len(df_test))

        # Log evaluation metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log diagnostic plot artifacts
        plot_files = [
            "actual_vs_predicted.png",
            "feature_importance.png",
            "residual_distribution.png",
            "timeseries_top5.png",
        ]
        for plot_file in plot_files:
            plot_path = os.path.join(str(PLOTS_DIR), plot_file)
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, artifact_path="plots")

        # Log forecast CSV
        if os.path.exists(forecast_path):
            mlflow.log_artifact(forecast_path, artifact_path="tables")

        # Register model in MLflow Model Registry
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME_DEMAND,
        )

        logger.info(
            "MLflow run logged  |  run_id=%s  |  experiment=%s",
            run.info.run_id,
            MLFLOW_EXPERIMENT_DEMAND,
        )

    return {
        "model": model,
        "metrics": metrics,
        "forecast": forecast,
        "feature_cols": feature_cols,
        "encoders": encoders,
    }


# ===========================================================================
# Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    from src.utils import load_all_tables

    tables = load_all_tables()
    results = run_demand_forecasting(tables)

    logger.info("══════ Pipeline complete ══════")
    for metric, value in results["metrics"].items():
        logger.info("  %-8s: %.4f", metric.upper(), value)
