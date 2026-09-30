from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.config import Config
from src.data.load import load_diabetic_data, make_readmit_target
from src.data.clean import basic_cleaning
from src.data.features import engineer_features
from src.paths import ARTIFACTS_DIR, METRICS_DIR, ensure_dirs
from src.utils.io import load_joblib, load_json, save_json


def _fairness_report(df: pd.DataFrame, y_true, y_pred, group_col: str) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    groups = df[group_col].fillna('Unknown').astype(str)

    for group in sorted(groups.unique()):
        mask = groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        report[group] = {
            'count': int(len(yt)),
            'recall': float(recall_score(yt, yp, zero_division=0)),
            'precision': float(precision_score(yt, yp, zero_division=0)),
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) else 0.0,
            'positive_prediction_rate': float(pd.Series(yp).mean()),
        }
    return report


def main():
    ensure_dirs()
    cfg = Config()

    df = load_diabetic_data()
    df = make_readmit_target(df, target_col=cfg.TARGET_COL)
    df = basic_cleaning(df)
    df = engineer_features(df)

    X = df.drop(columns=['y_readmit_30', 'y_label', cfg.TARGET_COL], errors='ignore')
    y = df['y_readmit_30'].values

    model_path = ARTIFACTS_DIR / 'lgbm_pipeline.joblib'
    meta_path = ARTIFACTS_DIR / 'lgbm_meta.json'
    if not model_path.exists():
        model_path = ARTIFACTS_DIR / 'baseline_logreg_pipeline.joblib'
        meta_path = ARTIFACTS_DIR / 'baseline_logreg_meta.json'

    pipe = load_joblib(model_path)
    meta = load_json(meta_path)
    threshold = float(meta.get('threshold', 0.5))

    prob = pipe.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    outputs = {
        'gender': _fairness_report(df, y, pred, 'gender') if 'gender' in df.columns else {},
        'race': _fairness_report(df, y, pred, 'race') if 'race' in df.columns else {},
        'age': _fairness_report(df, y, pred, 'age') if 'age' in df.columns else {},
    }

    save_json(outputs, METRICS_DIR / 'fairness_report.json')
    print('Saved fairness report -> reports/metrics/fairness_report.json')


if __name__ == '__main__':
    main()