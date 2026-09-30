from src.paths import ensure_dirs
from src.data.load import load_diabetic_data, load_ids_mapping, make_readmit_target

def main():
    ensure_dirs()

    df = load_diabetic_data()
    _ = load_ids_mapping()

    df2 = make_readmit_target(df)
    print(df2[["readmitted", "y_readmit_30"]].value_counts().head(10))

if __name__ == "__main__":
    main()