from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import requests

from src.paths import PROJECT_ROOT, ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR, RAW_DIR
from src.utils.logger import get_logger

logger = get_logger("reports.export")

API_URL = "http://127.0.0.1:8000"


def safe_copy(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Copied: {src.name} -> {dst}")
    else:
        logger.warning(f"Missing (skipped): {src}")


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def make_sample_predictions_via_api(pack_dir: Path, n: int = 25):
    """
    Fast + safe approach:
    - sample N rows from raw diabetic_data.csv
    - POST to FastAPI /batch_predict
    - save returned predictions as CSV + one example JSON
    """
    csv_path = RAW_DIR / "diabetic_data.csv"
    if not csv_path.exists():
        logger.warning("Raw dataset not found at data/raw/diabetic_data.csv; skipping sample predictions.")
        return

    # Try API health first
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if not health.get("model_loaded", False):
            logger.warning("API reachable but model not loaded; skipping sample predictions.")
            return
    except Exception as e:
        logger.warning(f"API not reachable ({e}); skipping sample predictions.")
        return

    df = pd.read_csv(csv_path, na_values=["?", "NA", "NaN", ""], low_memory=False)

    # Remove accidental target/leakage columns if present
    for bad in ["readmitted", "y_label", "y_readmit_30"]:
        if bad in df.columns:
            df = df.drop(columns=[bad])

    sample = df.sample(min(n, len(df)), random_state=42)

    # Save the sample input CSV (optional but useful)
    sample_dir = pack_dir / "sample_outputs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_input_path = sample_dir / "sample_input.csv"
    sample.to_csv(sample_input_path, index=False)

    # Call API batch endpoint
    try:
        files = {
            "file": ("sample_input.csv", sample.to_csv(index=False).encode("utf-8"), "text/csv")
        }
        resp = requests.post(
            f"{API_URL}/batch_predict",
            files=files,
            params={"explain": False},
            timeout=60
        )
        resp.raise_for_status()
        out = resp.json()
    except Exception as e:
        logger.warning(f"Batch predict call failed ({e}); skipping sample predictions.")
        return

    results = out.get("results", [])
    if not results:
        logger.warning("API returned no results; skipping sample predictions.")
        return

    # Save batch predictions CSV
    batch_df = pd.DataFrame(results)
    batch_path = sample_dir / "sample_batch_predictions.csv"
    batch_df.to_csv(batch_path, index=False)
    logger.info(f"Saved sample batch predictions -> {batch_path}")

    # Save single example JSON (first record)
    single_obj = {
        "api_health": health,
        "input_features_example": sample.iloc[0].to_dict(),
        "output_example": results[0],
    }
    single_path = sample_dir / "sample_single_prediction.json"
    save_json(single_obj, single_path)
    logger.info(f"Saved sample single prediction -> {single_path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "submission_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_dir = out_dir / f"diabetes_readmission_pack_{timestamp}"
    pack_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Copy key artifacts
    # -------------------------
    safe_copy(ARTIFACTS_DIR / "lgbm_pipeline.joblib", pack_dir / "models" / "lgbm_pipeline.joblib")
    safe_copy(ARTIFACTS_DIR / "lgbm_meta.json", pack_dir / "models" / "lgbm_meta.json")

    safe_copy(ARTIFACTS_DIR / "baseline_logreg_pipeline.joblib", pack_dir / "models" / "baseline_logreg_pipeline.joblib")
    safe_copy(ARTIFACTS_DIR / "baseline_logreg_meta.json", pack_dir / "models" / "baseline_logreg_meta.json")

    safe_copy(METRICS_DIR / "lgbm_metrics.json", pack_dir / "metrics" / "lgbm_metrics.json")
    safe_copy(METRICS_DIR / "baseline_logreg_metrics.json", pack_dir / "metrics" / "baseline_logreg_metrics.json")

    safe_copy(METRICS_DIR / "treatment_effectiveness_summary.json", pack_dir / "analysis" / "treatment_effectiveness_summary.json")
    safe_copy(METRICS_DIR / "local_explanation_example.json", pack_dir / "analysis" / "local_explanation_example.json")

    safe_copy(FIGURES_DIR / "shap_summary.png", pack_dir / "figures" / "shap_summary.png")

    # -------------------------
    # Sample predictions (via API, avoids slow model unpickling)
    # -------------------------
    make_sample_predictions_via_api(pack_dir, n=25)

    # -------------------------
    # Create PPT/report-ready summary
    # -------------------------
    lgbm = load_json_if_exists(METRICS_DIR / "lgbm_metrics.json")
    base = load_json_if_exists(METRICS_DIR / "baseline_logreg_metrics.json")

    lines = []
    lines.append("Diabetes 30-Day Readmission Predictive Analytics — Results Summary")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    def add_model(name: str, metrics: dict):
        if not metrics:
            lines.append(f"{name}: metrics not found.")
            lines.append("")
            return
        test = metrics.get("test", {})
        lines.append(f"{name} (Test Set):")
        lines.append(f"  ROC-AUC:     {test.get('roc_auc', 'NA')}")
        lines.append(f"  PR-AUC:      {test.get('pr_auc', 'NA')}")
        lines.append(f"  Recall:      {test.get('recall', 'NA')}")
        lines.append(f"  Precision:   {test.get('precision', 'NA')}")
        lines.append(f"  Specificity: {test.get('specificity', 'NA')}")
        lines.append(f"  Confusion Matrix: {test.get('confusion_matrix', {})}")
        lines.append("")

    add_model("LightGBM", lgbm)
    add_model("Baseline Logistic Regression", base)

    lines.append("Interpretability Artifacts:")
    lines.append("  - figures/shap_summary.png (global SHAP feature importance)")
    lines.append("  - analysis/local_explanation_example.json (local explanation example)")
    lines.append("")
    lines.append("Treatment Effectiveness Artifacts:")
    lines.append("  - analysis/treatment_effectiveness_summary.json (grouped readmission rates)")
    lines.append("")
    lines.append("Sample Prediction Outputs:")
    lines.append("  - sample_outputs/sample_input.csv")
    lines.append("  - sample_outputs/sample_batch_predictions.csv")
    lines.append("  - sample_outputs/sample_single_prediction.json")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Model emphasizes recall due to healthcare risk sensitivity (avoid false negatives).")
    lines.append("  - Operating threshold tuned using validation set.")

    summary_path = pack_dir / "RESULTS_SUMMARY.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Created summary -> {summary_path}")

    # -------------------------
    # Zip it
    # -------------------------
    zip_path = out_dir / f"diabetes_readmission_pack_{timestamp}.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
        for p in pack_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(pack_dir))
    logger.info(f"Created ZIP -> {zip_path}")

    print("\n✅ Submission pack created!")
    print(f"Folder: {pack_dir}")
    print(f"ZIP:    {zip_path}")


if __name__ == "__main__":
    main()