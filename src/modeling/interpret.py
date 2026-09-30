from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.paths import FIGURES_DIR, ensure_dirs


def _get_preprocess_and_model(pipe):
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    return preprocess, model


def _transform_input(pipe, X: pd.DataFrame):
    preprocess, _ = _get_preprocess_and_model(pipe)
    X_proc = preprocess.transform(X)

    if hasattr(preprocess, "get_feature_names_out"):
        feature_names = list(preprocess.get_feature_names_out())
    else:
        feature_names = [f"f_{i}" for i in range(X_proc.shape[1])]

    # convert sparse to dense if needed
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    return X_proc, feature_names


def shap_global_summary(
    pipe,
    X: pd.DataFrame,
    max_samples: int = 1000,
    save_path: str | Path | None = None,
):
    """
    Create and save SHAP global summary plot for the model pipeline.
    """
    ensure_dirs()

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42).copy()
    else:
        X = X.copy()

    X_proc, feature_names = _transform_input(pipe, X)
    _, model = _get_preprocess_and_model(pipe)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    if save_path is None:
        save_path = FIGURES_DIR / "shap_summary.png"
    else:
        save_path = Path(save_path)

    plt.figure()
    shap.summary_plot(shap_values, X_proc, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return str(save_path)


def shap_local_explanation(
    pipe,
    X: pd.DataFrame,
    row_index: int = 0,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return top contributing features for one row using SHAP values.
    """
    X = X.copy().reset_index(drop=True)
    X_proc, feature_names = _transform_input(pipe, X)
    _, model = _get_preprocess_and_model(pipe)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    row_vals = shap_values[row_index]
    pairs = list(zip(feature_names, row_vals))
    pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    return [
        {"feature": str(f), "shap_value": float(v)}
        for f, v in pairs
    ]


def explain_single_pipeline(
    pipe,
    X: pd.DataFrame,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper used by API.
    """
    return shap_local_explanation(pipe, X, row_index=0, top_k=top_k)