from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils.io import load_joblib, load_json
from src.modeling.interpret import shap_local_explanation


class RiskPredictor:
    def __init__(self, model_path: Path, meta_path: Path, model_name: str):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model_name = model_name

        self.pipeline = load_joblib(model_path)
        meta = load_json(meta_path)
        self.threshold = float(meta.get("threshold", 0.5))

        # Best-effort: try to capture expected raw feature names
        # ColumnTransformer stores columns internally; we just need consistent input columns.
        self.expected_columns = None

    def predict_one(self, features: dict, explain: bool = True, top_k: int = 8) -> dict:
        """
        features: raw dict of input fields.
        Returns probability, high_risk boolean, and top SHAP contributors (if explain=True).
        """
        X = pd.DataFrame([features])

        # IMPORTANT: Never include target columns if user mistakenly sends them
        for bad in ["readmitted", "y_label", "y_readmit_30"]:
            if bad in X.columns:
                X = X.drop(columns=[bad])

        proba = float(self.pipeline.predict_proba(X)[:, 1][0])
        high_risk = proba >= self.threshold

        resp = {
            "probability_readmit_30": proba,
            "high_risk": bool(high_risk),
            "threshold": float(self.threshold),
            "model_name": self.model_name,
        }

        if explain:
            try:
                local = shap_local_explanation(self.pipeline, X, top_k=top_k)
                resp["top_contributors"] = local["top_contributors"]
            except Exception:
                # If SHAP fails in deployment, still return prediction
                resp["top_contributors"] = None

        return resp