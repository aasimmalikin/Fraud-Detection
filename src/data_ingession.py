import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import logging

logger = logging.getLogger(__name__)

def load_raw_data(split: str = "train") -> pd.DataFrame:
    """Load and merge transaction + identity tables."""
    trans = pd.read_csv(RAW_DATA_DIR / f"{split}_transaction.csv")
    identity = pd.read_csv(RAW_DATA_DIR / f"{split}_identity.csv")
    df = trans.merge(identity, on="TransactionID", how="left")
    logger.info("Loaded %s: shape=%s", split, df.shape)
    _validate(df)
    return df

def _validate(df: pd.DataFrame) -> None:
    assert "TransactionID" in df.columns, "Missing TransactionID"
    assert df["TransactionID"].is_unique, "Duplicate TransactionIDs"
    if "isFraud" in df.columns:
        assert df["isFraud"].isin([0, 1]).all(), "isFraud must be binary"