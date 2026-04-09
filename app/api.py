from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI

from src.database import fetch_prediction_history, save_prediction_record
from src.llm_analysis import create_salary_chart, generate_llm_analysis, image_to_base64
from src.model_service import predict_salary
from src.schemas import PredictionInput, PredictionResponse

app = FastAPI(title='Salary Prediction API', version='1.0.0')


@app.get('/health')
def healthcheck() -> dict:
    return {'status': 'ok'}


@app.get('/predict', response_model=PredictionResponse)
def predict(
    work_year: int,
    experience_level: str,
    employment_type: str,
    job_title: str,
    employee_residence: str,
    remote_ratio: int,
    company_location: str,
    company_size: str,
    save_result: Optional[bool] = True,
):
    payload = PredictionInput(
        work_year=work_year,
        experience_level=experience_level,
        employment_type=employment_type,
        job_title=job_title,
        employee_residence=employee_residence,
        remote_ratio=remote_ratio,
        company_location=company_location,
        company_size=company_size,
    )
    predicted_salary = predict_salary(payload)
    history = fetch_prediction_history(limit=50)

    run_id = str(uuid.uuid4())
    current_record = {**payload.model_dump(), 'predicted_salary_usd': predicted_salary, 'run_id': run_id}
    analysis_text = generate_llm_analysis(current_record, history)
    chart_path = create_salary_chart(history + [current_record], run_id)
    current_record['llm_analysis'] = analysis_text
    current_record['chart_base64'] = image_to_base64(chart_path)

    if save_result:
        save_prediction_record(current_record)

    return PredictionResponse(
        input_data=payload,
        predicted_salary_usd=predicted_salary,
        model_version='decision_tree_v1',
        confidence_note='Tree-based model trained on ds_salaries.csv; generalization depends on feature similarity to training data.',
    )


@app.get('/predict/full')
def predict_full(
    work_year: int,
    experience_level: str,
    employment_type: str,
    job_title: str,
    employee_residence: str,
    remote_ratio: int,
    company_location: str,
    company_size: str,
):
    payload = PredictionInput(
        work_year=work_year,
        experience_level=experience_level,
        employment_type=employment_type,
        job_title=job_title,
        employee_residence=employee_residence,
        remote_ratio=remote_ratio,
        company_location=company_location,
        company_size=company_size,
    )
    predicted_salary = predict_salary(payload)
    history = fetch_prediction_history(limit=50)
    run_id = str(uuid.uuid4())
    current_record = {**payload.model_dump(), 'predicted_salary_usd': predicted_salary, 'run_id': run_id}
    analysis_text = generate_llm_analysis(current_record, history)
    chart_path = create_salary_chart(history + [current_record], run_id)
    current_record['llm_analysis'] = analysis_text
    current_record['chart_base64'] = image_to_base64(chart_path)
    db_result = save_prediction_record(current_record)

    return {
        'prediction': predicted_salary,
        'analysis': analysis_text,
        'chart_path': chart_path,
        'saved': db_result,
        'input': payload.model_dump(),
    }
