from __future__ import annotations

from dataclasses import asdict

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import Config
from src.data.clean import basic_cleaning
from src.data.features import engineer_features
from src.data.load import load_diabetic_data
from src.modeling.evaluate import evaluate_binary, find_threshold_recall_precision
from src.modeling.preprocess import build_preprocessor
from src.paths import ARTIFACTS_DIR, METRICS_DIR, ensure_dirs
from src.utils.io import save_joblib, save_json
from src.utils.logger import get_logger

logger = get_logger("model.train_los")


def make_los_target(df, threshold_days: int = 7):
    df = df.copy()
    df["y_los_high"] = (df["time_in_hospital"] >= threshold_days).astype(int)
    return df


def build_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def main(target_recall: float = 0.80, min_precision: float = 0.25, threshold_days: int = 7):
    ensure_dirs()
    cfg = Config()

    df = load_diabetic_data()
    df = basic_cleaning(df)
    df = engineer_features(df)
    df = make_los_target(df, threshold_days=threshold_days)

    y = df["y_los_high"].values
    X = df.drop(columns=["y_los_high", "time_in_hospital"], errors="ignore")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.SEED,
        stratify=y,
    )

    valid_size_adjusted = cfg.VALID_SIZE / (1.0 - cfg.TEST_SIZE)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval,
        y_trainval,
        test_size=valid_size_adjusted,
        random_state=cfg.SEED,
        stratify=y_trainval,
    )

    logger.info(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    tmp = X_train.copy()
    tmp["y_los_high"] = y_train
    preprocessor = build_preprocessor(tmp, target_col="y_los_high")

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", build_lgbm(cfg.SEED)),
        ]
    )

    pipe.fit(X_train, y_train)

    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    thr = find_threshold_recall_precision(
        y_valid,
        valid_proba,
        target_recall=target_recall,
        min_precision=min_precision,
    )
    valid_metrics = evaluate_binary(y_valid, valid_proba, threshold=thr)

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_binary(y_test, test_proba, threshold=thr)

    model_path = ARTIFACTS_DIR / "los_lgbm_pipeline.joblib"
    meta_path = ARTIFACTS_DIR / "los_lgbm_meta.json"
    metrics_path = METRICS_DIR / "los_lgbm_metrics.json"

    save_joblib(pipe, model_path)
    save_json(
        {
            "config": asdict(cfg),
            "threshold": float(thr),
            "target_recall": float(target_recall),
            "min_precision": float(min_precision),
            "threshold_days": int(threshold_days),
            "artifact": model_path.name,
            "model_name": "LightGBM_LOS",
            "notes": "Regularized LOS model to reduce overconfident probabilities",
        },
        meta_path,
    )
    save_json({"valid": valid_metrics, "test": test_metrics}, metrics_path)

    logger.info(
        f"Chosen threshold: {thr:.4f} "
        f"(recall≥{target_recall}, precision≥{min_precision})"
    )
    logger.info(
        f"VALID | Recall={valid_metrics['recall']:.4f} "
        f"Prec={valid_metrics['precision']:.4f} "
        f"ROC-AUC={valid_metrics['roc_auc']:.4f}"
    )
    logger.info(
        f"TEST  | Recall={test_metrics['recall']:.4f} "
        f"Prec={test_metrics['precision']:.4f} "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}"
    )
    logger.info(f"Saved LOS model -> {model_path}")
    logger.info(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()