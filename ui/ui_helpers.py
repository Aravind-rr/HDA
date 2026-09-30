import json
from pathlib import Path
import pandas as pd


def load_treatment_summary(project_root: Path) -> dict:
    path = project_root / "reports" / "metrics" / "treatment_effectiveness_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dict_to_df(d: dict, key_col: str = "group", val_col: str = "readmission_rate") -> pd.DataFrame:
    df = pd.DataFrame(list(d.items()), columns=[key_col, val_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    return df.sort_values(val_col, ascending=False)