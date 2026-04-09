from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import requests

from src.config import CHARTS_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL


def create_salary_chart(history: list[dict], run_id: str) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    chart_path = CHARTS_DIR / f'{run_id}.png'

    plt.figure(figsize=(8, 4.5))
    if not frame.empty and {'job_title', 'predicted_salary_usd'}.issubset(frame.columns):
        summary = (
            frame.groupby('job_title', as_index=False)['predicted_salary_usd']
            .mean()
            .sort_values('predicted_salary_usd', ascending=False)
            .head(8)
        )
        plt.bar(summary['job_title'], summary['predicted_salary_usd'])
        plt.xticks(rotation=35, ha='right')
        plt.ylabel('Predicted salary (USD)')
        plt.title('Average predicted salary by job title')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No history available yet', ha='center', va='center')
        plt.axis('off')
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    return str(chart_path)


def image_to_base64(path: str) -> str:
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def fallback_analysis(current_record: dict, history: list[dict]) -> str:
    predicted = current_record['predicted_salary_usd']
    all_values = [item['predicted_salary_usd'] for item in history if 'predicted_salary_usd' in item]
    avg_salary = mean(all_values) if all_values else predicted
    delta = predicted - avg_salary
    direction = 'above' if delta >= 0 else 'below'
    return (
        f"The predicted salary is ${predicted:,.0f}. Compared with the average stored prediction of "
        f"${avg_salary:,.0f}, this result is {abs(delta):,.0f} USD {direction} the current benchmark. "
        f"The main drivers are likely seniority ({current_record['experience_level']}), role fit ({current_record['job_title']}), "
        f"and geography ({current_record['company_location']}/{current_record['employee_residence']}). "
        "Use the bar chart to compare how this role sits relative to the top titles seen in prior prediction runs."
    )


def generate_llm_analysis(current_record: dict, history: list[dict]) -> str:
    prompt = {
        'current_prediction': current_record,
        'historical_predictions': history[-20:],
        'task': 'Write a concise data-analyst narrative in 120-180 words about the salary prediction and the salary landscape. Mention likely drivers and how to interpret the chart.',
    }
    try:
        response = requests.post(
            f'{OLLAMA_BASE_URL}/api/generate',
            json={'model': OLLAMA_MODEL, 'prompt': json.dumps(prompt), 'stream': False},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get('response', '').strip()
        if text:
            return text
    except Exception:
        pass
    return fallback_analysis(current_record, history)
