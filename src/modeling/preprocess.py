import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "diabetic_data.csv")


def load_and_prepare_data():
    """
    Loads the diabetes dataset and prepares features and target.

    Target:
        1 -> readmitted within 30 days ("<30")
        0 -> not readmitted within 30 days ("NO" or ">30")
    """
    print(f"Reading dataset from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            f"Please check whether 'diabetic_data.csv' exists inside data/raw/"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # IMPORTANT:
    # Use np.nan, not pd.NA, to avoid sklearn ambiguity errors
    df = df.replace("?", np.nan)

    if "readmitted" not in df.columns:
        raise KeyError("Column 'readmitted' not found in dataset.")

    # Keep only expected target values
    df = df[df["readmitted"].isin(["NO", ">30", "<30"])].copy()

    # Binary target
    df["target"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Drop columns that are identifiers / high-missing / less useful
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
        "readmitted",
    ]
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert known numeric columns safely
    numeric_like_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    for col in numeric_like_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df.drop(columns=["target"], errors="ignore")
    y = df["target"].astype(int)

    return X, y


def build_preprocessor(X):
    """
    Builds preprocessing pipeline:
    - numeric: median imputation + scaling
    - categorical: most frequent imputation + one-hot encoding
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Safety: if any column is missed by dtype detection, force it into categorical
    covered = set(categorical_cols + numerical_cols)
    extra_cols = [col for col in X.columns if col not in covered]
    categorical_cols.extend(extra_cols)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    return preprocessor