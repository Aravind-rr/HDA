from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.paths import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger("data.load")

REQUIRED_COLS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
}

def load_diabetic_data(csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = RAW_DIR / "diabetic_data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing file: {csv_path}\n"
            f"Put diabetic_data.csv inside data/raw/."
        )

    df = pd.read_csv(
        csv_path,
        na_values=["?", "NA", "NaN", ""],
        low_memory=False  # Fix dtype warning
    )

    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols from {csv_path.name}")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df


def make_readmit_target(df: pd.DataFrame, target_col: str = "readmitted") -> pd.DataFrame:
    out = df.copy()
    out["y_label"] = out[target_col].astype(str)
    out["y_readmit_30"] = (out["y_label"] == "<30").astype(int)
    return out