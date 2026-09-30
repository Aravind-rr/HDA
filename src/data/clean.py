import pandas as pd

LEAKAGE_COLUMNS = [
    "encounter_id",
    "patient_nbr",
]

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop leakage IDs
    - Drop duplicates
    - Remove invalid gender
    """
    df = df.copy()

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove invalid gender (UCI has 'Unknown/Invalid')
    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"]

    # Drop leakage identifiers
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    return df