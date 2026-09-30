from __future__ import annotations
from pathlib import Path

# Project root = folder containing api/, src/, etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

# Prefer LightGBM model if available
LGBM_MODEL = ARTIFACTS_DIR / "lgbm_pipeline.joblib"
LGBM_META = ARTIFACTS_DIR / "lgbm_meta.json"

BASELINE_MODEL = ARTIFACTS_DIR / "baseline_logreg_pipeline.joblib"
BASELINE_META = ARTIFACTS_DIR / "baseline_logreg_meta.json"