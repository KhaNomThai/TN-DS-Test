"""
run_pipeline.py – Main entry point for the SME Retail ML Pipeline.

Usage:
    python -m src.run_pipeline                  # Run both models
    python -m src.run_pipeline --demand-only    # Run demand forecasting only
    python -m src.run_pipeline --promo-only     # Run promotion recommendation only
"""

import argparse
import sys
import time

from src.utils import get_logger, load_all_tables, set_global_seed
from src.config import SEED

logger = get_logger(__name__)


def main() -> None:
    """Parse CLI arguments and execute the selected pipeline(s)."""
    parser = argparse.ArgumentParser(
        description="SME Retail ML Pipeline – Maximize Revenue",
    )
    parser.add_argument(
        "--demand-only",
        action="store_true",
        help="Run only the Demand Forecasting model.",
    )
    parser.add_argument(
        "--promo-only",
        action="store_true",
        help="Run only the Promotion Recommendation model.",
    )
    args = parser.parse_args()

    run_demand = not args.promo_only
    run_promo = not args.demand_only

    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("SME Retail ML Pipeline - Maximize Revenue")
    logger.info("=" * 70)

    set_global_seed(SEED)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading data tables ...")
    t0 = time.time()
    tables = load_all_tables()
    logger.info(f"Data loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Model 1: Demand Forecasting
    # ------------------------------------------------------------------
    if run_demand:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MODEL 1: DEMAND FORECASTING (LightGBM)")
        logger.info("=" * 70)
        t1 = time.time()

        from src.demand_forecasting import run_demand_forecasting
        demand_results = run_demand_forecasting(tables)

        logger.info(f"Demand Forecasting completed in {time.time() - t1:.1f}s")
        logger.info(f"  Metrics: {demand_results['metrics']}")

    # ------------------------------------------------------------------
    # Model 2: Promotion Recommendation
    # ------------------------------------------------------------------
    if run_promo:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MODEL 2: PERSONALIZED PROMOTION RECOMMENDATION")
        logger.info("=" * 70)
        t2 = time.time()

        from src.promotion_recommendation import run_promotion_recommendation
        promo_results = run_promotion_recommendation(tables)

        logger.info(
            f"Promotion Recommendation completed in {time.time() - t2:.1f}s"
        )
        logger.info(f"  Metrics: {promo_results['metrics']}")

    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
