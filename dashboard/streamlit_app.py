from __future__ import annotations

import base64
import os
import sys

import pandas as pd
import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import fetch_prediction_history

st.set_page_config(page_title='Salary Prediction Dashboard', layout='wide')
st.title('Salary Prediction Dashboard')
st.caption('This dashboard reads prediction history from Supabase only.')

history = fetch_prediction_history(limit=200)
frame = pd.DataFrame(history) if history else pd.DataFrame()

if frame.empty:
    st.warning('No prediction records found yet. You can still use the form below to trigger a saved prediction.')
else:
    col1, col2, col3 = st.columns(3)
    col1.metric('Predictions stored', len(frame))
    col2.metric('Average salary', f"${frame['predicted_salary_usd'].mean():,.0f}")
    col3.metric('Top job title', frame.groupby('job_title')['predicted_salary_usd'].mean().idxmax())

    st.subheader('Prediction history')
    st.dataframe(
        frame[['created_at', 'job_title', 'experience_level', 'company_location', 'remote_ratio', 'predicted_salary_usd']],
        use_container_width=True,
    )

    st.subheader('Latest analyst narrative')
    latest = frame.sort_values('created_at', ascending=False).iloc[0]
    st.write(latest['llm_analysis'])

    st.subheader('Latest visualization')
    if latest.get('chart_base64'):
        st.image(base64.b64decode(latest['chart_base64']))
    else:
        st.info('No chart stored for the latest run.')

st.subheader('On-demand prediction (uses deployed API)')
with st.form('prediction_form'):
    cols = st.columns(4)
    work_year = cols[0].number_input('Work year', min_value=2020, max_value=2030, value=2023)
    experience_level = cols[1].selectbox('Experience level', ['EN', 'MI', 'SE', 'EX'])
    employment_type = cols[2].selectbox('Employment type', ['FT', 'PT', 'CT', 'FL'])
    company_size = cols[3].selectbox('Company size', ['S', 'M', 'L'])
    job_title = st.text_input('Job title', value='Data Scientist')
    use_llm = st.checkbox('Use Ollama analysis (slower)', value=False)
    r1, r2, r3 = st.columns(3)
    employee_residence = r1.text_input('Employee residence', value='US', max_chars=2)
    remote_ratio = r2.selectbox('Remote ratio', [0, 50, 100])
    company_location = r3.text_input('Company location', value='US', max_chars=2)
    api_base = st.text_input('API base URL', value='http://127.0.0.1:8000')
    submitted = st.form_submit_button('Get prediction')

if submitted:
    params = {
        'work_year': int(work_year),
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'employee_residence': employee_residence.upper(),
        'remote_ratio': remote_ratio,
        'company_location': company_location.upper(),
        'company_size': company_size,
        'use_llm': use_llm,
    }
    try:
        response = requests.get(f'{api_base}/predict/full', params=params, timeout=60 if use_llm else 30)
        response.raise_for_status()
        payload = response.json()
        st.success(f"Predicted salary: ${payload['prediction']:,.0f}")
        st.write(payload['analysis'])
        if payload.get('chart_base64'):
            st.image(base64.b64decode(payload['chart_base64']))
        elif payload.get('chart_path'):
            st.caption(f"Chart saved on API server: {payload['chart_path']}")
        else:
            st.info('No chart returned by the API.')
    except requests.RequestException as exc:
        st.error(f'API request failed: {exc}')
