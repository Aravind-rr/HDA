from __future__ import annotations

import pandas as pd

from src.data.load import load_diabetic_data, make_readmit_target
from src.data.clean import basic_cleaning
from src.data.features import engineer_features
from src.utils.io import load_joblib, save_json
from src.paths import ARTIFACTS_DIR, METRICS_DIR
from src.modeling.interpret import shap_global_summary, shap_local_explanation
from src.analysis.treatment_effectiveness import grouped_readmission_rates


def main():
    # Load trained model (LightGBM) if you ran Step 7
    model_path = ARTIFACTS_DIR / "lgbm_pipeline.joblib"
    if not model_path.exists():
        # fallback to baseline logistic
        model_path = ARTIFACTS_DIR / "baseline_logreg_pipeline.joblib"

    pipe = load_joblib(model_path)

    # Load + prep data
    df = load_diabetic_data()
    df = make_readmit_target(df)
    df = basic_cleaning(df)
    df = engineer_features(df)

    # Create a sample for SHAP (keep it small for speed)
    X = df.drop(columns=["y_readmit_30", "y_label", "readmitted"], errors="ignore")
    X_sample = X.sample(2000, random_state=42)

    # Global SHAP plot
    shap_global_summary(pipe, X_sample)

    # Local explanation for one patient
    one = X_sample.iloc[[0]]
    local = shap_local_explanation(pipe, one, top_k=10)
    save_json(local, METRICS_DIR / "local_explanation_example.json")

    # Treatment effectiveness analysis
    treatment_summary = grouped_readmission_rates(df)
    save_json(treatment_summary, METRICS_DIR / "treatment_effectiveness_summary.json")

    print("✅ Step 8 outputs generated:")
    print("- reports/figures/shap_summary.png")
    print("- reports/metrics/local_explanation_example.json")
    print("- reports/metrics/treatment_effectiveness_summary.json")


if __name__ == "__main__":
    main()