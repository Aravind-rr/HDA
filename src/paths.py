from pathlib import Path

# Project root = folder that contains "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
MODEL_REPORTS_DIR = MODELS_DIR / "reports"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_dirs() -> None:
    """Create expected folder structure if missing."""
    dirs = [
        DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
        MODELS_DIR, ARTIFACTS_DIR, MODEL_REPORTS_DIR,
        REPORTS_DIR, FIGURES_DIR, METRICS_DIR,
        NOTEBOOKS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)