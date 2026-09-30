import pandas as pd


def _safe_diag_group(code) -> str:
    """Map ICD-like diagnosis code into a broad clinical family."""
    try:
        code = str(code).strip()
        if not code or code.lower() == 'nan':
            return 'other'
        if code.startswith('250'):
            return 'diabetes'
        value = float(code)
        if 390 <= value < 460 or value == 785:
            return 'circulatory'
        if 460 <= value < 520 or value == 786:
            return 'respiratory'
        if 520 <= value < 580 or value == 787:
            return 'digestive'
        if 800 <= value < 1000:
            return 'injury'
        return 'other'
    except Exception:
        return 'other'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expanded healthcare feature engineering.
    Keeps old features and adds operational + population-health signals.
    """
    df = df.copy()

    # Existing features
    if 'change' in df.columns:
        df['medication_changed'] = (df['change'] == 'Ch').astype(int)

    if 'A1Cresult' in df.columns:
        df['a1c_abnormal'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)

    util_cols = ['number_inpatient', 'number_outpatient', 'number_emergency']
    if all(col in df.columns for col in util_cols):
        df['total_visits'] = df[util_cols].fillna(0).sum(axis=1)

    # New healthcare features
    if 'total_visits' in df.columns:
        df['high_utilization'] = (df['total_visits'] >= 3).astype(int)
        df['very_high_utilization'] = (df['total_visits'] >= 6).astype(int)

    if 'num_medications' in df.columns:
        meds = pd.to_numeric(df['num_medications'], errors='coerce').fillna(0)
        df['polypharmacy'] = (meds >= 10).astype(int)

    if 'time_in_hospital' in df.columns:
        stay = pd.to_numeric(df['time_in_hospital'], errors='coerce').fillna(0)
        df['long_stay_flag'] = (stay >= 7).astype(int)
        df['very_long_stay_flag'] = (stay >= 10).astype(int)

    if 'number_emergency' in df.columns:
        emerg = pd.to_numeric(df['number_emergency'], errors='coerce').fillna(0)
        df['emergency_heavy'] = (emerg >= 2).astype(int)

    if 'number_inpatient' in df.columns:
        inp = pd.to_numeric(df['number_inpatient'], errors='coerce').fillna(0)
        df['prior_inpatient_flag'] = (inp >= 1).astype(int)
        df['frequent_inpatient_flag'] = (inp >= 2).astype(int)

    if 'number_diagnoses' in df.columns:
        ndiag = pd.to_numeric(df['number_diagnoses'], errors='coerce').fillna(0)
        df['high_comorbidity'] = (ndiag >= 6).astype(int)

    if 'num_lab_procedures' in df.columns:
        nlab = pd.to_numeric(df['num_lab_procedures'], errors='coerce').fillna(0)
        df['high_lab_burden'] = (nlab >= 40).astype(int)

    if 'insulin' in df.columns:
        df['insulin_active'] = df['insulin'].isin(['Down', 'Steady', 'Up']).astype(int)
        df['insulin_escalated'] = (df['insulin'] == 'Up').astype(int)

    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df.columns:
            df[f'{diag_col}_group'] = df[diag_col].apply(_safe_diag_group)

    # Composite clinical complexity score
    score_inputs = [
        'a1c_abnormal',
        'polypharmacy',
        'high_utilization',
        'long_stay_flag',
        'high_comorbidity',
        'emergency_heavy',
        'prior_inpatient_flag',
    ]
    present = [c for c in score_inputs if c in df.columns]
    if present:
        df['care_complexity_score'] = df[present].fillna(0).sum(axis=1)

    return df