from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data.clean import basic_cleaning
from src.data.features import engineer_features
from src.data.load import load_diabetic_data, make_readmit_target
from src.paths import METRICS_DIR, ensure_dirs
from src.utils.io import save_json


def main(n_clusters: int = 4):
    ensure_dirs()

    df = load_diabetic_data()
    df = make_readmit_target(df)
    df = basic_cleaning(df)
    df = engineer_features(df)

    features = [
        'time_in_hospital',
        'num_medications',
        'number_inpatient',
        'number_emergency',
        'number_outpatient',
        'number_diagnoses',
        'total_visits',
        'care_complexity_score',
    ]
    features = [c for c in features if c in df.columns]

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['patient_segment'] = km.fit_predict(Xs)

    summary = (
        df.groupby('patient_segment')
        .agg(
            count=('patient_segment', 'size'),
            avg_readmit_30=('y_readmit_30', 'mean'),
            avg_stay=('time_in_hospital', 'mean'),
            avg_meds=('num_medications', 'mean'),
            avg_total_visits=('total_visits', 'mean'),
            avg_complexity=('care_complexity_score', 'mean'),
        )
        .round(4)
        .to_dict(orient='index')
    )

    save_json(summary, METRICS_DIR / 'segment_summary.json')
    print('Saved segments -> reports/metrics/segment_summary.json')


if __name__ == '__main__':
    main()