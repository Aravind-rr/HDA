import os
import warnings
import pandas as pd
from joblib import dump

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier, log_evaluation

from src.modeling.preprocess import load_and_prepare_data, build_preprocessor

warnings.filterwarnings("ignore")


def print_separator(title=None, width=80):
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def print_confusion_details(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nTN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    return tn, fp, fn, tp


def evaluate_model(y_true, y_pred, y_prob=None, label="EVALUATION"):
    print_separator(label)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy      : {acc:.6f}")
    print(f"Precision     : {prec:.6f}")
    print(f"Recall        : {rec:.6f}")
    print(f"F1 Score      : {f1:.6f}")

    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            print(f"ROC-AUC       : {roc_auc:.6f}")
            print(f"PR-AUC        : {pr_auc:.6f}")
        except Exception as e:
            print(f"Could not calculate ROC-AUC / PR-AUC: {e}")

    print_confusion_details(y_true, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=6, zero_division=0))


def main():
    print_separator("STEP 1: LOADING DATASET")
    X, y = load_and_prepare_data()

    print(f"Feature shape : {X.shape}")
    print(f"Target shape  : {y.shape}")

    print("\nTarget distribution:")
    print(pd.Series(y).value_counts().sort_index())
    print("\nMeaning:")
    print("0 = Not readmitted within 30 days")
    print("1 = Readmitted within 30 days")

    print_separator("STEP 2: TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")

    print_separator("STEP 3: PREPROCESSING")
    preprocessor = build_preprocessor(X)

    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed X_train shape : {X_train_processed.shape}")
    print(f"Processed X_test shape  : {X_test_processed.shape}")

    print_separator("STEP 4: MODEL TRAINING")

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )

    print("Training started...")
    print("Iteration logs below:\n")

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_train_processed, y_train), (X_test_processed, y_test)],
        eval_names=["train", "valid"],
        eval_metric="binary_logloss",
        callbacks=[log_evaluation(period=1)]
    )

    print("\nTraining completed.")

    if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
        print(f"Best iteration: {model.best_iteration_}")

    print_separator("STEP 5: PREDICTIONS")

    y_train_prob = model.predict_proba(X_train_processed)[:, 1]
    y_test_prob = model.predict_proba(X_test_processed)[:, 1]

    y_train_pred = (y_train_prob >= 0.5).astype(int)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    print("Predictions completed.")

    evaluate_model(y_train, y_train_pred, y_train_prob, label="TRAIN METRICS")
    evaluate_model(y_test, y_test_pred, y_test_prob, label="TEST METRICS")

    print_separator("STEP 6: SAVING MODEL")
    os.makedirs("models", exist_ok=True)

    model_path = os.path.join("models", "lgbm_pipeline.joblib")
    dump(
        {
            "preprocessor": preprocessor,
            "model": model
        },
        model_path
    )

    print("Model saved successfully at:")
    print(model_path)

    print_separator("DONE")



if __name__ == "__main__":
    main()