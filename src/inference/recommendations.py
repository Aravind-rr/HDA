from __future__ import annotations

from typing import Any, Dict, List, Optional


def _risk_bucket(prob: float) -> str:
    if prob >= 0.70:
        return 'very_high'
    if prob >= 0.45:
        return 'high'
    if prob >= 0.20:
        return 'moderate'
    return 'low'


def generate_recommendations(
    features: Dict[str, Any],
    readmit_prob: float,
    los_prob: Optional[float] = None
) -> Dict[str, Any]:
    """Rule-based care coordination suggestions for demo/viva/dashboard use."""
    recommendations: List[str] = []

    if readmit_prob >= 0.45:
        recommendations.append('Schedule early follow-up within 7 days after discharge.')
        recommendations.append('Arrange discharge counseling and confirm medication adherence.')

    if int(features.get('a1c_abnormal', 0)) == 1:
        recommendations.append('Escalate diabetes education and glycemic control review.')

    if int(features.get('polypharmacy', 0)) == 1:
        recommendations.append('Perform medication reconciliation and polypharmacy review.')

    if int(features.get('high_utilization', 0)) == 1 or int(features.get('very_high_utilization', 0)) == 1:
        recommendations.append('Assign case manager for closer care coordination and follow-up calls.')

    if int(features.get('emergency_heavy', 0)) == 1:
        recommendations.append('Review emergency visit triggers and reinforce outpatient escalation plan.')

    if int(features.get('long_stay_flag', 0)) == 1:
        recommendations.append('Assess discharge readiness and post-discharge support needs carefully.')

    if los_prob is not None and los_prob >= 0.50:
        recommendations.append('Consider early multidisciplinary rounding to reduce prolonged stay risk.')

    if str(features.get('diag_1_group', '')) == 'circulatory':
        recommendations.append('Monitor cardiovascular comorbidity closely during discharge planning.')

    if not recommendations:
        recommendations.append('Standard discharge follow-up advised.')

    return {
        'risk_bucket': _risk_bucket(readmit_prob),
        'recommendations': recommendations,
    }