from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictRequest, PredictResponse, BatchPredictResponse, BatchPredictRow
from src.data.features import engineer_features
from src.inference.recommendations import generate_recommendations
from src.paths import ARTIFACTS_DIR, METRICS_DIR
from src.utils.io import load_joblib, load_json
from src.utils.logger import get_logger

logger = get_logger('api')

app = FastAPI(title='Universal Healthcare Intelligence Platform API', version='2.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

READMIT_PIPE = None
READMIT_THRESHOLD = 0.5
READMIT_MODEL_NAME = 'Unknown'

LOS_PIPE = None
LOS_THRESHOLD = 0.5
LOS_MODEL_NAME = 'Unavailable'


def _load_models():
    global READMIT_PIPE, READMIT_THRESHOLD, READMIT_MODEL_NAME
    global LOS_PIPE, LOS_THRESHOLD, LOS_MODEL_NAME

    lgbm_model = ARTIFACTS_DIR / 'lgbm_pipeline.joblib'
    lgbm_meta = ARTIFACTS_DIR / 'lgbm_meta.json'
    base_model = ARTIFACTS_DIR / 'baseline_logreg_pipeline.joblib'
    base_meta = ARTIFACTS_DIR / 'baseline_logreg_meta.json'

    if lgbm_model.exists() and lgbm_meta.exists():
        READMIT_PIPE = load_joblib(lgbm_model)
        meta = load_json(lgbm_meta)
        READMIT_MODEL_NAME = meta.get('model_name', meta.get('model', 'LightGBM'))
        READMIT_THRESHOLD = float(meta.get('threshold', 0.5))
    elif base_model.exists() and base_meta.exists():
        READMIT_PIPE = load_joblib(base_model)
        meta = load_json(base_meta)
        READMIT_MODEL_NAME = meta.get('model_name', 'LogisticRegression')
        READMIT_THRESHOLD = float(meta.get('threshold', 0.5))
    else:
        raise FileNotFoundError('No readmission model artifacts found')

    los_model = ARTIFACTS_DIR / 'los_lgbm_pipeline.joblib'
    los_meta = ARTIFACTS_DIR / 'los_lgbm_meta.json'
    if los_model.exists() and los_meta.exists():
        LOS_PIPE = load_joblib(los_model)
        meta = load_json(los_meta)
        LOS_MODEL_NAME = meta.get('model_name', 'LightGBM_LOS')
        LOS_THRESHOLD = float(meta.get('threshold', 0.5))


@app.on_event('startup')
def startup():
    try:
        _load_models()
        logger.info(f'Loaded readmission model: {READMIT_MODEL_NAME} @ {READMIT_THRESHOLD}')
        if LOS_PIPE is not None:
            logger.info(f'Loaded LOS model: {LOS_MODEL_NAME} @ {LOS_THRESHOLD}')
    except Exception as e:
        logger.exception(f'Failed to load model(s): {e}')


def _prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['readmitted', 'y_label', 'y_readmit_30', 'y_los_high']:
        if c in df.columns:
            df = df.drop(columns=[c], errors='ignore')
    return engineer_features(df)


def _ensure_columns(pipe, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        preprocess = pipe.named_steps.get('preprocess') if pipe is not None else None
        if preprocess is not None and hasattr(preprocess, 'feature_names_in_'):
            for c in list(preprocess.feature_names_in_):
                if c not in df.columns:
                    if c in [
                        'race', 'gender', 'age', 'insulin', 'A1Cresult', 'change',
                        'diag_1', 'diag_2', 'diag_3',
                        'diag_1_group', 'diag_2_group', 'diag_3_group'
                    ]:
                        df[c] = 'Unknown'
                    else:
                        df[c] = 0
    except Exception:
        pass
    return df


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'readmission_model_loaded': READMIT_PIPE is not None,
        'readmission_model_name': READMIT_MODEL_NAME,
        'readmission_threshold': READMIT_THRESHOLD,
        'los_model_loaded': LOS_PIPE is not None,
        'los_model_name': LOS_MODEL_NAME,
        'los_threshold': LOS_THRESHOLD,
    }


@app.get('/fairness')
def fairness():
    path = METRICS_DIR / 'fairness_report.json'
    if not path.exists():
        raise HTTPException(status_code=404, detail='fairness_report.json not found. Run: python -m src.analysis.fairness')
    return load_json(path)


@app.get('/segments')
def segments():
    path = METRICS_DIR / 'segment_summary.json'
    if not path.exists():
        raise HTTPException(status_code=404, detail='segment_summary.json not found. Run: python -m src.analysis.segmentation')
    return load_json(path)


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest, explain: bool = Query(default=False)):
    if READMIT_PIPE is None:
        raise HTTPException(status_code=500, detail='Readmission model not loaded')

    X = pd.DataFrame([req.features])
    X = _prepare_input(X)
    X_readmit = _ensure_columns(READMIT_PIPE, X)

    readmit_prob = float(READMIT_PIPE.predict_proba(X_readmit)[:, 1][0])
    high_risk = bool(readmit_prob >= READMIT_THRESHOLD)

    los_prob = None
    if LOS_PIPE is not None:
        X_los = _ensure_columns(LOS_PIPE, X)
        los_prob = float(LOS_PIPE.predict_proba(X_los)[:, 1][0])

    enriched = X.iloc[0].to_dict()
    bundle = generate_recommendations(enriched, readmit_prob, los_prob)

    top_contributors: List[Dict[str, float]] = []
    if explain:
        try:
            from src.modeling.interpret import explain_single_pipeline
            top_contributors = explain_single_pipeline(READMIT_PIPE, X_readmit, top_k=5)
        except Exception as e:
            logger.exception(f'Explain failed: {e}')

    return PredictResponse(
        probability_readmit_30=readmit_prob,
        high_risk=high_risk,
        threshold=READMIT_THRESHOLD,
        model_name=READMIT_MODEL_NAME,
        risk_bucket=bundle['risk_bucket'],
        probability_long_stay=los_prob,
        los_model_name=LOS_MODEL_NAME if LOS_PIPE is not None else None,
        los_threshold=LOS_THRESHOLD if LOS_PIPE is not None else None,
        recommendations=bundle['recommendations'],
        top_contributors=top_contributors,
    )


@app.post('/batch_predict', response_model=BatchPredictResponse)
def batch_predict(file: UploadFile = File(...), explain: bool = Query(default=False)):
    if READMIT_PIPE is None:
        raise HTTPException(status_code=500, detail='Readmission model not loaded')
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
        df = _prepare_input(df)
        X_readmit = _ensure_columns(READMIT_PIPE, df)
        readmit_probs = READMIT_PIPE.predict_proba(X_readmit)[:, 1]

        los_probs = None
        if LOS_PIPE is not None:
            X_los = _ensure_columns(LOS_PIPE, df)
            los_probs = LOS_PIPE.predict_proba(X_los)[:, 1]

        rows = []
        for i, p in enumerate(readmit_probs):
            risk_bundle = generate_recommendations(
                df.iloc[i].to_dict(),
                float(p),
                float(los_probs[i]) if los_probs is not None else None
            )
            rows.append(
                BatchPredictRow(
                    row_index=i,
                    probability_readmit_30=float(p),
                    high_risk=bool(p >= READMIT_THRESHOLD),
                    threshold=READMIT_THRESHOLD,
                    model_name=READMIT_MODEL_NAME,
                    risk_bucket=risk_bundle['risk_bucket'],
                    probability_long_stay=float(los_probs[i]) if los_probs is not None else None,
                )
            )
        return BatchPredictResponse(count=len(rows), results=rows)
    except Exception as e:
        logger.exception(f'Batch predict failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))