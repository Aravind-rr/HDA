from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Reproducibility
    SEED: int = 42

    # Target definition (UCI dataset uses readmitted: '<30', '>30', 'NO')
    TARGET_COL: str = "readmitted"
    POSITIVE_CLASS: str = "<30"  # 30-day readmission

    # Splits
    TEST_SIZE: float = 0.15
    VALID_SIZE: float = 0.15  # split remaining train into train/valid

    # Modeling priorities (healthcare)
    PRIMARY_METRICS: tuple = ("recall", "roc_auc")

    # Threshold selection
    DEFAULT_THRESHOLD: float = 0.50