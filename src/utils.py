"""
utils.py – Shared utility functions for the ML pipeline.
"""

import pandas as pd
import numpy as np
import logging

from src.config import DATA_FILES, SEED

# ===========================================================================
# Logging Setup
# ===========================================================================

def get_logger(name: str) -> logging.Logger:
    """Create a formatted logger for the given module name.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ===========================================================================
# Data Loading
# ===========================================================================

def load_all_tables() -> dict[str, pd.DataFrame]:
    """Load all CSV data tables into a dictionary of DataFrames.

    Returns:
        Dictionary mapping table name to its DataFrame.  Date columns
        are automatically parsed.
    """
    logger = get_logger(__name__)
    tables = {}

    date_columns = {
        "sales": ["datetime"],
        "po": ["manufacturing_date", "po_date", "arrival_date", "expire_date"],
        "stock_movement": ["receive_date", "transfer_date"],
        "promotions": ["start_date", "end_date"],
        "customers": ["registration_date"],
        "stores": ["open_date"],
    }

    for name, path in DATA_FILES.items():
        parse_dates = date_columns.get(name, None)
        df = pd.read_csv(path, parse_dates=parse_dates)
        tables[name] = df
        logger.info(f"Loaded {name:20s}: {len(df):>8,} rows x {df.shape[1]:>3} cols")

    return tables


def set_global_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility across numpy and random.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
