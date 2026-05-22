"""
promotion_recommendation.py – Personalized Promotion Recommendation via
ALS-style Matrix Factorization for the SME Retail pipeline.

This module recommends promotions to individual customers by:
1.  Building an implicit-feedback interaction matrix (customer × product).
2.  Training a Matrix Factorization model with Alternating Least Squares.
3.  Generating top-K product recommendations per customer.
4.  Mapping recommended products to active / upcoming promotions.
5.  Segmenting customers by RFM + promo-affinity signals.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    MF_LEARNING_RATE,
    MF_N_EPOCHS,
    MF_N_FACTORS,
    MF_REG_LAMBDA,
    MODELS_DIR,
    PLOTS_DIR,
    TABLES_DIR,
    SEED,
    TOP_K_RECOMMENDATIONS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_PROMO,
    MLFLOW_MODEL_NAME_PROMO,
    RFM_QUANTILES,
)
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)

# ===========================================================================
# 1. Interaction Matrix Construction
# ===========================================================================


def build_interaction_matrix(
    df_sales: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Build a customer × product interaction matrix from sales data.

    Values are ``log1p(total_qty)`` to dampen skew (implicit feedback).

    Args:
        df_sales: Sales transactions with at least ``customer_id``,
            ``product_id``, and ``qty`` columns.

    Returns:
        interaction_matrix: DataFrame indexed by *customer_id* with product
            ids as columns.
        customer_to_idx: Mapping from customer_id → integer row index.
        product_to_idx: Mapping from product_id → integer column index.
    """
    logger.info("Building customer × product interaction matrix …")

    # Aggregate total quantity per (customer, product) pair
    agg = (
        df_sales.groupby(["customer_id", "product_id"])["qty"]
        .sum()
        .reset_index(name="total_qty")
    )

    # Apply log1p transform to reduce skew
    agg["interaction"] = np.log1p(agg["total_qty"])

    # Pivot to wide matrix
    interaction_matrix = agg.pivot(
        index="customer_id", columns="product_id", values="interaction"
    ).fillna(0.0)

    # Build index mappings
    customer_to_idx: dict[str, int] = {
        cid: idx for idx, cid in enumerate(interaction_matrix.index)
    }
    product_to_idx: dict[str, int] = {
        pid: idx for idx, pid in enumerate(interaction_matrix.columns)
    }

    n_customers, n_products = interaction_matrix.shape
    n_nonzero = (interaction_matrix.values > 0).sum()
    total_entries = n_customers * n_products
    sparsity = 1.0 - (n_nonzero / total_entries) if total_entries > 0 else 1.0

    logger.info(
        f"Interaction matrix: {n_customers} customers × {n_products} products  |  "
        f"non-zero={n_nonzero:,}  sparsity={sparsity:.4%}"
    )

    return interaction_matrix, customer_to_idx, product_to_idx


# ===========================================================================
# 2. Matrix Factorization (ALS)
# ===========================================================================


class MatrixFactorizationALS:
    """Alternating Least Squares Matrix Factorization for implicit feedback.

    Decomposes the interaction matrix **R ≈ U · Vᵀ** by alternating between
    closed-form updates for user factors *U* and item factors *V*.

    Attributes:
        n_factors: Dimensionality of the latent space.
        n_epochs: Number of ALS iterations.
        learning_rate: *Not used* in pure ALS but kept for API compat.
        reg_lambda: L2 regularisation strength (λ).
        seed: Random seed for weight initialisation.
        user_factors: Learned user factor matrix (n_users × n_factors).
        item_factors: Learned item factor matrix (n_items × n_factors).
        training_loss: Per-epoch MSE on observed (non-zero) entries.
    """

    def __init__(
        self,
        n_factors: int = 30,
        n_epochs: int = 50,
        learning_rate: float = 0.01,
        reg_lambda: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.seed = seed

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.training_loss: list[float] = []

    # --------------------------------------------------------------------- #
    # Fit
    # --------------------------------------------------------------------- #

    def fit(self, interaction_matrix: np.ndarray) -> "MatrixFactorizationALS":
        """Train the model via Alternating Least Squares.

        Args:
            interaction_matrix: Dense array of shape (n_users, n_items).

        Returns:
            Fitted model (self).
        """
        rng = np.random.RandomState(self.seed)
        R = np.asarray(interaction_matrix, dtype=np.float64)
        n_users, n_items = R.shape

        # Small random initialisation
        self.user_factors = rng.normal(
            scale=0.01, size=(n_users, self.n_factors)
        )
        self.item_factors = rng.normal(
            scale=0.01, size=(n_items, self.n_factors)
        )
        self.training_loss = []

        mask = R > 0  # observed entries only
        lambda_eye = self.reg_lambda * np.eye(self.n_factors)

        logger.info(
            f"ALS training: {n_users} users, {n_items} items, "
            f"{self.n_factors} factors, {self.n_epochs} epochs, "
            f"λ={self.reg_lambda}"
        )

        for epoch in range(1, self.n_epochs + 1):
            # --- Fix item factors, solve for user factors ---
            VtV = self.item_factors.T @ self.item_factors  # (k, k)
            for u in range(n_users):
                observed_items = np.where(mask[u])[0]
                if len(observed_items) == 0:
                    continue
                V_u = self.item_factors[observed_items]  # (|obs|, k)
                r_u = R[u, observed_items]               # (|obs|,)
                A = V_u.T @ V_u + lambda_eye
                b = V_u.T @ r_u
                self.user_factors[u] = np.linalg.solve(A, b)

            # --- Fix user factors, solve for item factors ---
            UtU = self.user_factors.T @ self.user_factors  # (k, k)
            for i in range(n_items):
                observed_users = np.where(mask[:, i])[0]
                if len(observed_users) == 0:
                    continue
                U_i = self.user_factors[observed_users]  # (|obs|, k)
                r_i = R[observed_users, i]               # (|obs|,)
                A = U_i.T @ U_i + lambda_eye
                b = U_i.T @ r_i
                self.item_factors[i] = np.linalg.solve(A, b)

            # --- Compute training loss (MSE on observed entries) ---
            R_hat = self.user_factors @ self.item_factors.T
            residuals = (R - R_hat)[mask]
            mse = float(np.mean(residuals ** 2))
            self.training_loss.append(mse)

            if epoch % max(1, self.n_epochs // 10) == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch:>3d}/{self.n_epochs}  MSE={mse:.6f}")

        if self.training_loss:
            logger.info(f"ALS training complete. Final MSE={self.training_loss[-1]:.6f}")
        else:
            logger.warning("ALS training completed but no training loss was recorded (possibly an empty interaction matrix).")
        return self

    # --------------------------------------------------------------------- #
    # Predict
    # --------------------------------------------------------------------- #

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict the score for a single (user, item) pair.

        Args:
            user_idx: Row index of the user.
            item_idx: Column index of the item.

        Returns:
            Predicted interaction score.
        """
        if user_idx >= len(self.user_factors) or item_idx >= len(self.item_factors):
            return 0.0
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])

    def predict_all(self) -> np.ndarray:
        """Reconstruct the full user–item score matrix.

        Returns:
            Array of shape (n_users, n_items).
        """
        return self.user_factors @ self.item_factors.T

    # --------------------------------------------------------------------- #
    # Recommend
    # --------------------------------------------------------------------- #

    def recommend_items(
        self,
        user_idx: int,
        interaction_matrix: np.ndarray,
        top_k: int = 5,
    ) -> list[int]:
        """Return top-K item indices the user has **not** interacted with.

        Args:
            user_idx: Row index of the target user.
            interaction_matrix: Original interaction array (used to exclude
                already-purchased items).
            top_k: Number of items to return.

        Returns:
            List of item column indices ordered by descending score.
        """
        scores = self.user_factors[user_idx] @ self.item_factors.T
        already_purchased = np.where(
            np.asarray(interaction_matrix[user_idx]) > 0
        )[0]
        scores[already_purchased] = -np.inf
        top_items = np.argsort(scores)[::-1][:top_k]
        return top_items.tolist()


class ALSModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapper for the custom ALS Matrix Factorization model.

    This wrapper enables MLflow to serialize, deserialize, and serve the
    custom :class:`MatrixFactorizationALS` model through the standard
    ``mlflow.pyfunc`` interface.
    """

    def __init__(self, als_model: MatrixFactorizationALS | None = None) -> None:
        self.als_model = als_model

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> np.ndarray:
        """Predict interaction scores for given user-item pairs using vectorized ops.

        Args:
            context: MLflow context.
            model_input: DataFrame with ``user_idx`` and ``item_idx`` columns.

        Returns:
            Array of predicted scores.
        """
        user_indices = model_input["user_idx"].astype(int).values
        item_indices = model_input["item_idx"].astype(int).values
        
        valid_mask = (user_indices < len(self.als_model.user_factors)) & \
                     (item_indices < len(self.als_model.item_factors))
        
        scores = np.zeros(len(model_input))
        valid_users = user_indices[valid_mask]
        valid_items = item_indices[valid_mask]
        
        if len(valid_users) > 0:
            scores[valid_mask] = np.sum(
                self.als_model.user_factors[valid_users] * 
                self.als_model.item_factors[valid_items], 
                axis=1
            )
            
        return scores


# ===========================================================================
# 3. Map Recommendations → Promotions
# ===========================================================================


def map_recommendations_to_promotions(
    recommendations: dict[Any, list[int]],
    product_to_idx: dict[Any, int],
    df_promotions: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Map recommended product indices to active/future promotions.

    Args:
        recommendations: Dict mapping *customer_id* → list of recommended
            product **indices**.
        product_to_idx: Mapping product_id → column index (used in reverse).
        df_promotions: Promotion master table with ``promotion_id``,
            ``promotion_type``, ``discount``, ``product_id``,
            ``start_date``, ``end_date``.
        reference_date: Date used to filter active/future promos.
            Defaults to today.

    Returns:
        DataFrame with columns: customer_id, product_id, product_name,
        recommended_promotion_id, discount, promo_type, score.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()

    idx_to_product = {v: k for k, v in product_to_idx.items()}

    # Active or future promotions
    active_promos = df_promotions[
        df_promotions["end_date"] >= reference_date
    ].copy()

    rows: list[dict] = []
    for customer_id, item_indices in recommendations.items():
        for rank, item_idx in enumerate(item_indices, start=1):
            product_id = idx_to_product.get(item_idx)
            if product_id is None:
                continue

            matching = active_promos[active_promos["product_id"] == product_id]
            if matching.empty:
                rows.append(
                    {
                        "customer_id": customer_id,
                        "product_id": product_id,
                        "recommended_promotion_id": None,
                        "discount": 0.0,
                        "promo_type": None,
                        "score": 1.0 / rank,
                    }
                )
            else:
                for _, promo in matching.iterrows():
                    rows.append(
                        {
                            "customer_id": customer_id,
                            "product_id": product_id,
                            "recommended_promotion_id": promo["promotion_id"],
                            "discount": promo["discount"],
                            "promo_type": promo["promotion_type"],
                            "score": 1.0 / rank,
                        }
                    )

    df_result = pd.DataFrame(rows)
    logger.info(
        f"Mapped recommendations → {len(df_result)} promotion entries "
        f"for {len(recommendations)} customers"
    )
    return df_result


# ===========================================================================
# 4. Customer Segmentation (RFM + Promo Affinity)
# ===========================================================================


def compute_customer_segments(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """Segment customers into promo-affinity groups using RFM features.

    Segments:
        * **promo_lover** – high promo_sensitivity *and* high frequency.
        * **price_sensitive** – high promo_sensitivity but lower frequency.
        * **loyal_full_price** – low promo_sensitivity and high frequency.
        * **occasional** – everyone else.

    Args:
        df_rfm: RFM DataFrame with at least ``frequency``,
            ``monetary``, ``recency`` columns.  If ``promo_sensitivity``
            is not present it is estimated from monetary quartile
            (lower monetary → higher sensitivity).

    Returns:
        Copy of *df_rfm* with an added ``segment`` column.
    """
    df = df_rfm.copy()

    # Derive promo_sensitivity if missing
    if "promo_sensitivity" not in df.columns:
        logger.info(
            "promo_sensitivity not found – estimating from monetary quartile"
        )
        monetary_q = pd.qcut(
            df["monetary"], q=RFM_QUANTILES, labels=False, duplicates="drop"
        )
        # Lower monetary quartile ⇒ higher price sensitivity
        df["promo_sensitivity"] = 1.0 - (monetary_q / monetary_q.max())

    freq_median = df["frequency"].median()
    promo_median = df["promo_sensitivity"].median()

    conditions = [
        (df["promo_sensitivity"] >= promo_median) & (df["frequency"] >= freq_median),
        (df["promo_sensitivity"] >= promo_median) & (df["frequency"] < freq_median),
        (df["promo_sensitivity"] < promo_median) & (df["frequency"] >= freq_median),
    ]
    choices = ["promo_lover", "price_sensitive", "loyal_full_price"]
    df["segment"] = np.select(conditions, choices, default="occasional")

    segment_counts = df["segment"].value_counts()
    logger.info(f"Customer segments:\n{segment_counts.to_string()}")

    return df


# ===========================================================================
# 5. Evaluation Metrics
# ===========================================================================


def evaluate_recommendations(
    model: MatrixFactorizationALS,
    interaction_matrix: np.ndarray,
    test_mask: np.ndarray,
    top_k: int = 5,
) -> dict[str, float]:
    """Evaluate recommendation quality with leave-one-out style metrics.

    Metrics computed:
        * **Precision@K** – fraction of recommended items in test set.
        * **Recall@K** – fraction of test items in recommendations.
        * **NDCG@K** – normalised discounted cumulative gain.
        * **Hit Rate@K** – fraction of users with ≥ 1 hit.
        * **Coverage** – fraction of all items ever recommended.

    Args:
        model: Trained :class:`MatrixFactorizationALS`.
        interaction_matrix: Original (full) interaction array.
        test_mask: Boolean array; ``True`` marks held-out entries.
        top_k: Number of recommended items per user.

    Returns:
        Dict of metric names to float values.
    """
    R = np.asarray(interaction_matrix)
    n_users, n_items = R.shape
    train_matrix = R.copy()
    train_matrix[test_mask] = 0.0

    precision_list: list[float] = []
    recall_list: list[float] = []
    ndcg_list: list[float] = []
    hit_list: list[int] = []
    all_recommended: set[int] = set()
    evaluated_users = 0

    for u in range(n_users):
        test_items = set(np.where(test_mask[u])[0])
        if len(test_items) == 0:
            continue

        rec_items = model.recommend_items(u, train_matrix, top_k=top_k)
        all_recommended.update(rec_items)

        hits = [1 if item in test_items else 0 for item in rec_items]
        n_hits = sum(hits)

        # Precision@K
        precision_list.append(n_hits / top_k)

        # Recall@K
        recall_list.append(n_hits / len(test_items))

        # NDCG@K
        dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
        ideal_hits = min(len(test_items), top_k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

        # Hit Rate
        hit_list.append(int(n_hits > 0))
        evaluated_users += 1

    metrics: dict[str, float] = {
        "precision_at_k": float(np.mean(precision_list)) if precision_list else 0.0,
        "recall_at_k": float(np.mean(recall_list)) if recall_list else 0.0,
        "ndcg_at_k": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "hit_rate_at_k": float(np.mean(hit_list)) if hit_list else 0.0,
        "coverage": len(all_recommended) / n_items if n_items > 0 else 0.0,
        "evaluated_users": evaluated_users,
    }

    logger.info(
        f"Evaluation (K={top_k}):  "
        f"Precision={metrics['precision_at_k']:.4f}  "
        f"Recall={metrics['recall_at_k']:.4f}  "
        f"NDCG={metrics['ndcg_at_k']:.4f}  "
        f"HitRate={metrics['hit_rate_at_k']:.4f}  "
        f"Coverage={metrics['coverage']:.4f}  "
        f"({evaluated_users} users evaluated)"
    )
    return metrics


# ===========================================================================
# 6. Train / Test Split
# ===========================================================================


def create_train_test_interaction(
    interaction_matrix: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly hold out a fraction of observed interactions for testing.

    Args:
        interaction_matrix: Dense interaction array (n_users × n_items).
        test_ratio: Fraction of non-zero entries to mask.
        seed: Random seed.

    Returns:
        train_matrix: Copy with held-out entries zeroed.
        test_mask: Boolean array (``True`` = held-out).
    """
    rng = np.random.RandomState(seed)
    R = np.asarray(interaction_matrix, dtype=np.float64)

    nonzero_users, nonzero_items = np.where(R > 0)
    n_nonzero = len(nonzero_users)
    n_test = max(1, int(n_nonzero * test_ratio))

    test_indices = rng.choice(n_nonzero, size=n_test, replace=False)

    test_mask = np.zeros_like(R, dtype=bool)
    test_mask[nonzero_users[test_indices], nonzero_items[test_indices]] = True

    train_matrix = R.copy()
    train_matrix[test_mask] = 0.0

    logger.info(
        f"Train/test split: {n_nonzero - n_test} train, {n_test} test "
        f"({test_ratio:.0%} held out)"
    )
    return train_matrix, test_mask


# ===========================================================================
# 7. Visualisation
# ===========================================================================


# ===========================================================================
# 7. Visualisation
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


def plot_recommendation_results(
    metrics: dict[str, float],
    model: MatrixFactorizationALS,
    output_dir: Path | str | None = None,
) -> None:
    """Generate diagnostic plots for the recommendation model.

    Plots:
        1. Training loss curve (MSE per epoch).
        2. Recommendation metrics bar chart (Precision, Recall, NDCG,
           Hit Rate).
        3. Distribution of predicted scores.

    Args:
        metrics: Dict returned by :func:`evaluate_recommendations`.
        model: Trained :class:`MatrixFactorizationALS`.
        output_dir: Directory to save figures. Defaults to
            ``config.PLOTS_DIR``.
    """
    if output_dir is None:
        output_dir = PLOTS_DIR
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Plot 1: Training Loss Curve ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(model.training_loss) + 1), model.training_loss, color="#4A00E0", marker="o", markersize=4, linewidth=1.5)
    _apply_premium_style(ax, title="ALS Matrix Factorization: Training Loss Curve", xlabel="Epoch", ylabel="MSE (Observed)", grid="both")
    fig.tight_layout()
    fig.savefig(output_dir / "rec_training_loss.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved training loss plot → {output_dir / 'rec_training_loss.png'}")

    # ---- Plot 2: Metrics Bar Chart ----
    metric_names = ["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"]
    display_names = ["Precision@K", "Recall@K", "NDCG@K", "Hit Rate@K"]
    values = [metrics.get(m, 0.0) for m in metric_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#1A535C"]
    bars = ax.bar(display_names, values, color=colors, width=0.55, edgecolor="none")

    # Add values on top of the bars
    max_val = max(values) if values else 0.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_val * 0.015 if max_val > 0 else 0.015),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#2C3E50",
        )
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)
    _apply_premium_style(ax, title=f"Recommendation Quality Metrics (K={TOP_K_RECOMMENDATIONS})", ylabel="Score", grid="y")
    fig.tight_layout()
    fig.savefig(output_dir / "rec_metrics.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved metrics plot → {output_dir / 'rec_metrics.png'}")

    # ---- Plot 3: Predicted Score Distribution ----
    predicted = model.predict_all()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(predicted.ravel(), bins=60, color="#5D6D7E", alpha=0.8, edgecolor="#34495E", rwidth=0.85)
    _apply_premium_style(ax, title="Distribution of Latent Reconstructed Scores", xlabel="Predicted Interaction Score", ylabel="Frequency", grid="y")
    fig.tight_layout()
    fig.savefig(output_dir / "rec_score_distribution.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved score distribution plot → {output_dir / 'rec_score_distribution.png'}")


def plot_rfm_segments(df_segments: pd.DataFrame, output_dir: Path | str | None = None) -> None:
    """Plot the distribution of RFM customer segments."""
    if output_dir is None:
        output_dir = PLOTS_DIR
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    segment_counts = df_segments["segment"].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Premium colors for segments
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(segment_counts)))
    
    bars = ax.bar(segment_counts.index, segment_counts.values, color=colors, edgecolor="none", width=0.6)
    
    # Add values on top of bars
    max_val = segment_counts.max()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_val * 0.015 if max_val > 0 else 0),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#2C3E50"
        )
        
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)
    plt.xticks(rotation=45, ha='right')
    
    _apply_premium_style(
        ax, 
        title="Customer Segment Distribution (RFM + Promo Affinity)",
        xlabel="Segment",
        ylabel="Number of Customers",
        grid="y"
    )
    
    fig.tight_layout()
    fig.savefig(output_dir / "rfm_segments.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved RFM segments plot → {output_dir / 'rfm_segments.png'}")


# ===========================================================================
# 8. Main Orchestrator
# ===========================================================================


def run_promotion_recommendation(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """End-to-end promotion recommendation pipeline.

    Steps:
        1. Build interaction matrix from sales.
        2. Create train / test split.
        3. Train MatrixFactorizationALS model.
        4. Evaluate recommendations.
        5. Compute RFM and customer segments.
        6. Generate personalised promo recommendations for all customers.
        7. Map to actual promotions.
        8. Plot results.
        9. Save outputs (recommendations CSV + model pickle).
        10. Log everything to MLflow & register model.

    Args:
        tables: Dictionary of DataFrames as returned by
            ``utils.load_all_tables()``.

    Returns:
        Dict with keys ``model``, ``metrics``, ``recommendations``,
        ``customer_segments``.
    """
    from src.feature_engineering import build_customer_product_matrix, compute_rfm

    set_global_seed(SEED)
    logger.info("=" * 60)
    logger.info("PROMOTION RECOMMENDATION PIPELINE")
    logger.info("=" * 60)

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PROMO)

    df_sales = tables["sales"]
    df_promotions = tables["promotions"]
    df_products = tables.get("products")

    # ---- 1. Build interaction matrix ----
    interaction_df, customer_to_idx, product_to_idx = build_interaction_matrix(
        df_sales
    )
    interaction_array = interaction_df.values

    # ---- 2. Train / test split ----
    train_matrix, test_mask = create_train_test_interaction(
        interaction_array, test_ratio=0.2, seed=SEED
    )

    # ---- 3. Train ALS model ----
    model = MatrixFactorizationALS(
        n_factors=MF_N_FACTORS,
        n_epochs=MF_N_EPOCHS,
        learning_rate=MF_LEARNING_RATE,
        reg_lambda=MF_REG_LAMBDA,
        seed=SEED,
    )
    model.fit(train_matrix)

    # ---- 4. Evaluate ----
    metrics = evaluate_recommendations(
        model, interaction_array, test_mask, top_k=TOP_K_RECOMMENDATIONS
    )

    # ---- 5. RFM + Customer Segments ----
    df_rfm = compute_rfm(df_sales)
    df_segments = compute_customer_segments(df_rfm)

    # ---- 6. Generate recommendations for every customer ----
    idx_to_customer = {v: k for k, v in customer_to_idx.items()}
    all_recommendations: dict[Any, list[int]] = {}

    for user_idx in range(interaction_array.shape[0]):
        customer_id = idx_to_customer[user_idx]
        rec_items = model.recommend_items(
            user_idx, interaction_array, top_k=TOP_K_RECOMMENDATIONS
        )
        all_recommendations[customer_id] = rec_items

    logger.info(
        f"Generated top-{TOP_K_RECOMMENDATIONS} recommendations for "
        f"{len(all_recommendations)} customers"
    )

    # ---- 7. Map to promotions ----
    # Use the most recent data point from sales as the reference date
    reference_date = pd.to_datetime(df_sales["datetime"]).max().normalize()
    df_promo_recs = map_recommendations_to_promotions(
        all_recommendations,
        product_to_idx,
        df_promotions,
        reference_date=reference_date,
    )

    # Enrich with product name if products table available
    if df_products is not None and "product_name" in df_products.columns:
        name_map = df_products.set_index("product_id")["product_name"].to_dict()
        df_promo_recs["product_name"] = df_promo_recs["product_id"].map(name_map)

    # ---- 8. Plot results ----
    plot_recommendation_results(metrics, model, output_dir=PLOTS_DIR)
    plot_rfm_segments(df_segments, output_dir=PLOTS_DIR)

    # ---- 9. Save outputs ----
    rec_csv_path = TABLES_DIR / "promotion_recommendations.csv"
    df_promo_recs.to_csv(rec_csv_path, index=False)
    logger.info(f"Saved recommendations → {rec_csv_path}")

    seg_csv_path = TABLES_DIR / "customer_segments.csv"
    df_segments.to_csv(seg_csv_path, index=False)
    logger.info(f"Saved customer segments → {seg_csv_path}")

    model_path = MODELS_DIR / "mf_als_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model pickle → {model_path}")

    # ── 10. MLflow Tracking & Model Registry ─────────────────────────────
    logger.info("══════ Step 10: Logging to MLflow ══════")
    with mlflow.start_run(run_name="ALS-Promo-Recommender") as run:
        # Log hyperparameters
        mlflow.log_param("n_factors", MF_N_FACTORS)
        mlflow.log_param("n_epochs", MF_N_EPOCHS)
        mlflow.log_param("learning_rate", MF_LEARNING_RATE)
        mlflow.log_param("reg_lambda", MF_REG_LAMBDA)
        mlflow.log_param("top_k", TOP_K_RECOMMENDATIONS)
        mlflow.log_param("test_ratio", 0.2)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_customers", interaction_array.shape[0])
        mlflow.log_param("n_products", interaction_array.shape[1])

        # Log evaluation metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        # Log final training loss
        if model.training_loss:
            mlflow.log_metric("final_training_mse", model.training_loss[-1])

        # Log diagnostic plot artifacts
        plot_files = [
            "rec_training_loss.png",
            "rec_metrics.png",
            "rec_score_distribution.png",
            "rfm_segments.png",
        ]
        for plot_file in plot_files:
            plot_path = str(PLOTS_DIR / plot_file)
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, artifact_path="plots")

        # Log output CSVs
        if os.path.exists(str(rec_csv_path)):
            mlflow.log_artifact(str(rec_csv_path), artifact_path="tables")
        if os.path.exists(str(seg_csv_path)):
            mlflow.log_artifact(str(seg_csv_path), artifact_path="tables")

        # Register model in MLflow Model Registry via custom PythonModel wrapper
        wrapped_model = ALSModelWrapper(als_model=model)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,
            registered_model_name=MLFLOW_MODEL_NAME_PROMO,
        )

        logger.info(
            "MLflow run logged  |  run_id=%s  |  experiment=%s",
            run.info.run_id,
            MLFLOW_EXPERIMENT_PROMO,
        )

    logger.info("Promotion recommendation pipeline complete ✓")

    return {
        "model": model,
        "metrics": metrics,
        "recommendations": df_promo_recs,
        "customer_segments": df_segments,
    }


# ===========================================================================
# Standalone execution
# ===========================================================================

if __name__ == "__main__":
    from src.utils import load_all_tables

    tables = load_all_tables()
    results = run_promotion_recommendation(tables)

    print("\n=== Recommendation Metrics ===")
    for name, value in results["metrics"].items():
        print(f"  {name:20s}: {value:.4f}")
    print(f"\nRecommendations shape: {results['recommendations'].shape}")
    print(f"Customer segments shape: {results['customer_segments'].shape}")
