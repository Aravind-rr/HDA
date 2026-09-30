from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from src.paths import RAW_DIR, ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR
from src.utils.io import load_joblib, load_json
from src.utils.logger import get_logger

logger = get_logger("analysis.calibration")


def ensure_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the engineered features required by the trained pipeline exist:
      - a1c_abnormal
      - medication_changed
      - total_visits

    We compute them from the raw columns if missing.
    """
    df = df.copy()

    # a1c_abnormal: 1 if A1Cresult is >7 or >8 else 0
    if "a1c_abnormal" not in df.columns:
        if "A1Cresult" in df.columns:
            df["a1c_abnormal"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
        else:
            df["a1c_abnormal"] = 0

    # medication_changed: 1 if change == "Ch"
    if "medication_changed" not in df.columns:
        if "change" in df.columns:
            df["medication_changed"] = (df["change"] == "Ch").astype(int)
        else:
            df["medication_changed"] = 0

    # total_visits = outpatient + emergency + inpatient
    if "total_visits" not in df.columns:
        outp = df["number_outpatient"] if "number_outpatient" in df.columns else 0
        emer = df["number_emergency"] if "number_emergency" in df.columns else 0
        inp = df["number_inpatient"] if "number_inpatient" in df.columns else 0
        df["total_visits"] = (outp + emer + inp).astype(int)

    return df


def decision_curve_analysis(y_true, y_prob, thresholds):
    n = len(y_true)
    net_benefits = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefits.append(nb)
    return np.array(net_benefits)


def main():
    logger.info("Loading raw dataset...")

    csv_path = RAW_DIR / "diabetic_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, na_values=["?", "NA", "NaN", ""], low_memory=False)

    # Target: 30-day readmit
    df["y_readmit_30"] = (df["readmitted"] == "<30").astype(int)

    # Drop leakage raw target column before modeling
    df = df.drop(columns=["readmitted"], errors="ignore")

    # Ensure engineered cols exist (required by your pipeline)
    df = ensure_engineered_features(df)

    y = df["y_readmit_30"].values
    X = df.drop(columns=["y_readmit_30", "y_label"], errors="ignore")

    logger.info("Loading trained model...")

    model_path = ARTIFACTS_DIR / "lgbm_pipeline.joblib"
    meta_path = ARTIFACTS_DIR / "lgbm_meta.json"
    if not model_path.exists():
        model_path = ARTIFACTS_DIR / "baseline_logreg_pipeline.joblib"
        meta_path = ARTIFACTS_DIR / "baseline_logreg_meta.json"

    model = load_joblib(model_path)
    meta = load_json(meta_path)

    threshold = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "Model")

    # Split (stratified)
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------
    # Calibration Curve
    # -------------------------
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Curve")
    plt.tight_layout()

    cal_path = FIGURES_DIR / "calibration_curve.png"
    plt.savefig(cal_path)
    plt.close()
    logger.info(f"Saved calibration curve -> {cal_path}")

    # -------------------------
    # Brier Score
    # -------------------------
    brier = brier_score_loss(y_test, y_prob)

    # -------------------------
    # Decision Curve
    # -------------------------
    thresholds = np.linspace(0.01, 0.99, 50)
    net_benefit = decision_curve_analysis(y_test, y_prob, thresholds)

    prevalence = float(np.mean(y_test))
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    plt.figure()
    plt.plot(thresholds, net_benefit, label="Model")
    plt.plot(thresholds, treat_all, linestyle="--", label="Treat All")
    plt.plot(thresholds, treat_none, linestyle="--", label="Treat None")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.tight_layout()

    dca_path = FIGURES_DIR / "decision_curve.png"
    plt.savefig(dca_path)
    plt.close()
    logger.info(f"Saved decision curve -> {dca_path}")

    # -------------------------
    # Save metrics
    # -------------------------
    output = {
        "model_name": model_name,
        "brier_score": float(brier),
        "operating_threshold": float(threshold),
        "prevalence_test": prevalence,
    }

    metrics_path = METRICS_DIR / "calibration_decision_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved calibration metrics -> {metrics_path}")

    print("\n✅ Step 13B Complete")
    print(f"Calibration curve: {cal_path}")
    print(f"Decision curve:    {dca_path}")
    print(f"Brier score:       {brier:.4f}")


if __name__ == "__main__":
    main()