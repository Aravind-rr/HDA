from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description='Raw patient features as key-value pairs')


class ShapContributor(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    probability_readmit_30: float
    high_risk: bool
    threshold: float
    model_name: str
    risk_bucket: str
    probability_long_stay: Optional[float] = None
    los_model_name: Optional[str] = None
    los_threshold: Optional[float] = None
    recommendations: List[str] = []
    top_contributors: List[ShapContributor] = []


class BatchPredictRow(BaseModel):
    row_index: int
    probability_readmit_30: float
    high_risk: bool
    threshold: float
    model_name: str
    risk_bucket: str
    probability_long_stay: Optional[float] = None


class BatchPredictResponse(BaseModel):
    count: int
    results: List[BatchPredictRow]