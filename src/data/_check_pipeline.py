from src.data.load import load_diabetic_data, make_readmit_target
from src.data.clean import basic_cleaning
from src.data.features import engineer_features
from src.modeling.preprocess import build_preprocessor


def main():
    df = load_diabetic_data()
    df = make_readmit_target(df)
    df = basic_cleaning(df)
    df = engineer_features(df)

    preprocessor = build_preprocessor(df)

    X = df.drop(columns=["y_readmit_30", "y_label"], errors="ignore")
    y = df["y_readmit_30"]

    X_processed = preprocessor.fit_transform(X)

    print("Processed Shape:", X_processed.shape)
    print("Target distribution:")
    print(y.value_counts())


if __name__ == "__main__":
    main()