import pandas as pd
import requests
import streamlit as st
from pathlib import Path

from ui_helpers import dict_to_df, load_treatment_summary

API_URL = 'http://127.0.0.1:8000'
PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title='Universal Healthcare Intelligence Platform', layout='wide')
st.title('Universal Healthcare Intelligence Platform')
st.caption('Readmission risk + prolonged stay + fairness + patient segments')


def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {'raw_text': resp.text}


def patient_payload():
    return {
        'features': {
            'race': 'Caucasian',
            'gender': 'Male',
            'age': '[60-70)',
            'time_in_hospital': 5,
            'num_lab_procedures': 40,
            'num_procedures': 1,
            'num_medications': 15,
            'number_outpatient': 0,
            'number_emergency': 1,
            'number_inpatient': 1,
            'number_diagnoses': 7,
            'A1Cresult': '>8',
            'insulin': 'Steady',
            'change': 'Ch',
            'diag_1': '250.83',
            'diag_2': '401',
            'diag_3': '276',
        }
    }


with st.sidebar:
    st.subheader('API Status')
    try:
        health = requests.get(f'{API_URL}/health', timeout=10)
        st.json(safe_json(health))
    except Exception as e:
        st.error(f'API not reachable: {e}')


t1, t2, t3, t4, t5 = st.tabs([
    'Risk Prediction',
    'Batch Prediction',
    'Fairness Audit',
    'Patient Segments',
    'Treatment Insights',
])

with t1:
    st.subheader('Single Patient Risk Prediction')
    payload = patient_payload()['features']

    c1, c2, c3 = st.columns(3)
    with c1:
        payload['race'] = st.selectbox('Race', ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other', 'Unknown'])
        payload['gender'] = st.selectbox('Gender', ['Male', 'Female'])
        payload['age'] = st.selectbox('Age', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
    with c2:
        payload['time_in_hospital'] = st.number_input('Time in Hospital', 1, 50, int(payload['time_in_hospital']))
        payload['num_lab_procedures'] = st.number_input('Lab Procedures', 0, 200, int(payload['num_lab_procedures']))
        payload['num_procedures'] = st.number_input('Procedures', 0, 50, int(payload['num_procedures']))
        payload['num_medications'] = st.number_input('Medications', 0, 100, int(payload['num_medications']))
    with c3:
        payload['number_outpatient'] = st.number_input('Outpatient Visits', 0, 50, int(payload['number_outpatient']))
        payload['number_emergency'] = st.number_input('Emergency Visits', 0, 50, int(payload['number_emergency']))
        payload['number_inpatient'] = st.number_input('Inpatient Visits', 0, 50, int(payload['number_inpatient']))
        payload['number_diagnoses'] = st.number_input('Number of Diagnoses', 1, 20, int(payload['number_diagnoses']))

    payload['A1Cresult'] = st.selectbox('HbA1c', ['None', 'Norm', '>7', '>8'])
    payload['insulin'] = st.selectbox('Insulin', ['No', 'Down', 'Steady', 'Up'])
    payload['change'] = st.selectbox('Medication Change', ['No', 'Ch'])
    payload['diag_1'] = st.text_input('diag_1', payload['diag_1'])
    payload['diag_2'] = st.text_input('diag_2', payload['diag_2'])
    payload['diag_3'] = st.text_input('diag_3', payload['diag_3'])
    explain = st.checkbox('Include SHAP explanation', value=True)

    if st.button('Predict'):
        resp = requests.post(f'{API_URL}/predict', json={'features': payload}, params={'explain': explain}, timeout=90)
        data = safe_json(resp)
        st.json(data)
        if resp.status_code < 400:
            st.metric('30-day readmission probability', f"{data['probability_readmit_30']:.4f}")
            if data.get('probability_long_stay') is not None:
                st.metric('Prolonged stay probability', f"{data['probability_long_stay']:.4f}")
            st.write('Risk bucket:', data['risk_bucket'])
            st.write('Recommendations:')
            for rec in data.get('recommendations', []):
                st.write('-', rec)
            if data.get('top_contributors'):
                st.dataframe(pd.DataFrame(data['top_contributors']), width='stretch')

with t2:
    st.subheader('Batch Prediction')
    file = st.file_uploader('Upload CSV', type=['csv'])
    if file is not None and st.button('Run Batch Prediction'):
        files = {'file': (file.name, file.getvalue(), 'text/csv')}
        resp = requests.post(f'{API_URL}/batch_predict', files=files, timeout=180)
        data = safe_json(resp)
        st.json(data)
        if resp.status_code < 400:
            df = pd.DataFrame(data['results'])
            st.dataframe(df, width='stretch')

with t3:
    st.subheader('Fairness Audit')
    resp = requests.get(f'{API_URL}/fairness', timeout=30)
    data = safe_json(resp)
    if resp.status_code >= 400:
        st.warning(data)
    else:
        st.json(data)
        for group_name, report in data.items():
            if report:
                st.markdown(f'### {group_name.title()}')
                st.dataframe(
                    pd.DataFrame(report).T.reset_index().rename(columns={'index': group_name}),
                    width='stretch'
                )

with t4:
    st.subheader('Patient Segments')
    resp = requests.get(f'{API_URL}/segments', timeout=30)
    data = safe_json(resp)
    if resp.status_code >= 400:
        st.warning(data)
    else:
        st.json(data)
        df = pd.DataFrame(data).T.reset_index().rename(columns={'index': 'segment'})
        st.dataframe(df, width='stretch')

with t5:
    st.subheader('Treatment Effectiveness & Pattern Insights')
    summary = load_treatment_summary(PROJECT_ROOT)
    if not summary:
        st.info('Run: python -m src.analysis.run_step8')
    else:
        st.json(summary)
        if 'readmission_rate_by_prior_utilization' in summary:
            df_util = dict_to_df(summary['readmission_rate_by_prior_utilization'], 'visit_bucket', 'readmission_rate')
            st.dataframe(df_util, width='stretch')
            st.bar_chart(df_util.set_index('visit_bucket'))