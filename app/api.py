from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI

from src.database import fetch_prediction_history, save_prediction_record
from src.llm_analysis import create_salary_chart, fallback_analysis, generate_llm_analysis, image_to_base64
from src.model_service import predict_salary
from src.schemas import PredictionInput, PredictionResponse

app = FastAPI(title='Salary Prediction API', version='1.0.0')


def _build_payload(
    work_year: int,
    experience_level: str,
    employment_type: str,
    job_title: str,
    employee_residence: str,
    remote_ratio: int,
    company_location: str,
    company_size: str,
) -> PredictionInput:
    return PredictionInput(
        work_year=work_year,
        experience_level=experience_level,
        employment_type=employment_type,
        job_title=job_title,
        employee_residence=employee_residence,
        remote_ratio=remote_ratio,
        company_location=company_location,
        company_size=company_size,
    )


def _build_record(
    payload: PredictionInput,
    predicted_salary: float,
    history: list[dict[str, Any]],
    use_llm: bool,
) -> tuple[dict[str, Any], str]:
    run_id = str(uuid.uuid4())
    current_record = {
        **payload.model_dump(),
        'predicted_salary_usd': predicted_salary,
        'run_id': run_id,
    }
    chart_path = create_salary_chart(history + [current_record], run_id)
    current_record['chart_base64'] = image_to_base64(chart_path)
    current_record['llm_analysis'] = (
        generate_llm_analysis(current_record, history) if use_llm else fallback_analysis(current_record, history)
    )
    return current_record, chart_path


@app.get('/health')
def healthcheck() -> dict[str, str]:
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
    save_result: bool = True,
) -> PredictionResponse:
    payload = _build_payload(
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

    if save_result:
        history = fetch_prediction_history(limit=50)
        record, _ = _build_record(payload, predicted_salary, history, use_llm=False)
        save_prediction_record(record)

    return PredictionResponse(
        input_data=payload,
        predicted_salary_usd=predicted_salary,
        model_version='random_forest_v1',
        confidence_note='Random-forest model trained on ds_salaries.csv with normalized role and region features; generalization depends on feature similarity to training data.',
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
    use_llm: bool = False,
) -> dict[str, Any]:
    payload = _build_payload(
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
    current_record, chart_path = _build_record(payload, predicted_salary, history, use_llm=use_llm)
    db_result = save_prediction_record(current_record)

    return {
        'prediction': predicted_salary,
        'analysis': current_record['llm_analysis'],
        'chart_path': chart_path,
        'chart_base64': current_record['chart_base64'],
        'saved': db_result,
        'input': payload.model_dump(),
    }
