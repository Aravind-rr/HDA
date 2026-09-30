from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("analysis.treatment")


def grouped_readmission_rates(df: pd.DataFrame) -> dict:
    """
    Produce interpretable treatment effectiveness insights using grouped comparisons.
    Returns a dictionary you can directly use in PPT/report.
    """
    out = {}

    # Ensure target exists
    if "y_readmit_30" not in df.columns:
        raise ValueError("Missing y_readmit_30. Run make_readmit_target() first.")

    # 1) Insulin change impact
    if "insulin" in df.columns:
        grp = df.groupby("insulin")["y_readmit_30"].mean().sort_values(ascending=False)
        out["readmission_rate_by_insulin_status"] = grp.round(4).to_dict()

    # 2) Medication changed flag (engineered)
    if "medication_changed" in df.columns:
        grp = df.groupby("medication_changed")["y_readmit_30"].mean()
        out["readmission_rate_medication_changed"] = {
            "no_change(0)": float(grp.get(0, np.nan)),
            "changed(1)": float(grp.get(1, np.nan)),
        }

    # 3) HbA1c abnormal impact (engineered)
    if "a1c_abnormal" in df.columns:
        grp = df.groupby("a1c_abnormal")["y_readmit_30"].mean()
        out["readmission_rate_a1c_abnormal"] = {
            "normal_or_missing(0)": float(grp.get(0, np.nan)),
            "abnormal(1)": float(grp.get(1, np.nan)),
        }

    # 4) Utilization risk buckets
    if "total_visits" in df.columns:
        bins = [-1, 0, 2, 5, 1000]
        labels = ["0", "1-2", "3-5", "6+"]
        df2 = df.copy()
        df2["visit_bucket"] = pd.cut(df2["total_visits"], bins=bins, labels=labels)
        grp = df2.groupby("visit_bucket")["y_readmit_30"].mean()
        out["readmission_rate_by_prior_utilization"] = grp.round(4).to_dict()

    # 5) Time in hospital
    if "time_in_hospital" in df.columns:
        df2 = df.copy()
        df2["tih_bucket"] = pd.cut(
            df2["time_in_hospital"],
            bins=[0, 2, 4, 7, 100],
            labels=["1-2", "3-4", "5-7", "8+"],
            include_lowest=True
        )
        grp = df2.groupby("tih_bucket")["y_readmit_30"].mean()
        out["readmission_rate_by_time_in_hospital"] = grp.round(4).to_dict()

    logger.info("Treatment effectiveness summary generated.")
    return out