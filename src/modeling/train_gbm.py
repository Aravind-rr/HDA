from __future__ import annotations

from dataclasses import asdict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from src.config import Config
from src.data.load import load_diabetic_data, make_readmit_target
from src.data.clean import basic_cleaning
from src.data.features import engineer_features
from src.modeling.preprocess import build_preprocessor
from src.modeling.evaluate import evaluate_binary, find_threshold_recall_precision
from src.paths import ensure_dirs, ARTIFACTS_DIR, METRICS_DIR
from src.utils.io import save_joblib, save_json
from src.utils.logger import get_logger

logger = get_logger("model.train_gbm")


def build_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def main(target_recall: float = 0.80, min_precision: float = 0.15):
    ensure_dirs()
    cfg = Config()

    df = load_diabetic_data()
    df = make_readmit_target(df, target_col=cfg.TARGET_COL)
    df = basic_cleaning(df)
    df = engineer_features(df)

    y = df["y_readmit_30"].values
    X = df.drop(columns=["y_readmit_30", "y_label", cfg.TARGET_COL], errors="ignore")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, stratify=y
    )

    valid_size_adjusted = cfg.VALID_SIZE / (1.0 - cfg.TEST_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval,
        test_size=valid_size_adjusted,
        random_state=cfg.SEED,
        stratify=y_trainval
    )

    logger.info(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    tmp_df = X_train.copy()
    tmp_df["y_readmit_30"] = y_train
    preprocessor = build_preprocessor(tmp_df, target_col="y_readmit_30")

    model = build_lgbm(cfg.SEED)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    thr = find_threshold_recall_precision(
        y_valid, valid_proba, target_recall=target_recall, min_precision=min_precision
    )

    valid_metrics = evaluate_binary(y_valid, valid_proba, threshold=thr)
    logger.info(f"Chosen threshold: {thr:.4f} (recall≥{target_recall}, precision≥{min_precision})")
    logger.info(f"VALID | Recall={valid_metrics['recall']:.4f} Prec={valid_metrics['precision']:.4f} ROC-AUC={valid_metrics['roc_auc']:.4f}")

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_binary(y_test, test_proba, threshold=thr)
    logger.info(f"TEST  | Recall={test_metrics['recall']:.4f} Prec={test_metrics['precision']:.4f} ROC-AUC={test_metrics['roc_auc']:.4f}")

    model_path = ARTIFACTS_DIR / "lgbm_pipeline.joblib"
    meta_path = ARTIFACTS_DIR / "lgbm_meta.json"
    metrics_path = METRICS_DIR / "lgbm_metrics.json"

    save_joblib(pipe, model_path)
    save_json(
        {
            "config": asdict(cfg),
            "threshold": thr,
            "target_recall": target_recall,
            "min_precision": min_precision,
            "artifact": str(model_path.name),
            "model": "LightGBM",
        },
        meta_path
    )
    save_json({"valid": valid_metrics, "test": test_metrics}, metrics_path)

    logger.info(f"Saved model -> {model_path}")
    logger.info(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()