# Salary Prediction Application

End-to-end ML assignment solution built around a **Decision Tree regressor**, a **FastAPI prediction service**, an **Ollama-powered analyst narrative**, **Supabase persistence**, and a **Streamlit dashboard**.

## What this project includes

- Dataset cleaning and model training pipeline
- Decision Tree regression model for salary prediction in USD
- GET-based FastAPI endpoint with input validation
- Python client that exercises multiple combinations of valid inputs
- LLM analysis layer using Ollama, with a deterministic fallback if Ollama is unavailable
- Supabase integration for storing predictions, narratives, and charts
- Streamlit dashboard that reads from Supabase only
- SQL schema and deployment-ready project structure

## Folder structure

```text
salary_prediction_app/
├── app/
│   └── api.py
├── artifacts/
├── dashboard/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── ds_salaries.csv
├── sql/
│   └── schema.sql
├── src/
│   ├── client.py
│   ├── config.py
│   ├── database.py
│   ├── llm_analysis.py
│   ├── model_service.py
│   ├── preprocess.py
│   ├── schemas.py
│   └── train.py
├── .env.example
├── README.md
└── requirements.txt
```

## 1) Setup

```bash
git clone <your-repo-url>
cd salary_prediction_app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Supabase credentials to `.env`.

## 2) Train the model

```bash
python -m src.train
```

This writes:
- `artifacts/salary_model.joblib`
- `artifacts/training_metadata.json`
- `artifacts/metrics.json`

## 3) Run the API

```bash
uvicorn app.api:app --reload
```

### Example prediction request

```bash
curl "http://127.0.0.1:8000/predict/full?work_year=2023&experience_level=SE&employment_type=FT&job_title=Data%20Scientist&employee_residence=US&remote_ratio=100&company_location=US&company_size=M"
```

## 4) Exercise the API with the client script

```bash
python -m src.client
```

The client generates multiple valid input combinations from training metadata and calls the deployed API.

## 5) Create the Supabase table

Run the SQL in `sql/schema.sql` inside the Supabase SQL editor.

## 6) Launch the Streamlit dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard reads prediction history from Supabase only. The form can trigger new predictions through the deployed FastAPI endpoint.

## Input validation rules

- `work_year`: 2020–2030
- `experience_level`: EN, MI, SE, EX
- `employment_type`: FT, PT, CT, FL
- `employee_residence` and `company_location`: 2-letter country codes
- `remote_ratio`: 0, 50, 100 (the dataset distribution)
- `company_size`: S, M, L

## Suggested deployment split

- **FastAPI**: Render, Railway, Fly.io, or EC2
- **Streamlit dashboard**: Streamlit Community Cloud or Render
- **Supabase**: hosted Postgres + storage
- **Ollama**: local machine or a self-hosted box where the model is installed

## Notes

- The Ollama step falls back to a deterministic analyst summary if the local model is unavailable.
- For unseen job titles, the API defaults to `Data Scientist` so the pipeline still returns a prediction instead of failing.
- The `/predict` route returns the prediction response only; `/predict/full` includes narrative generation, chart creation, and Supabase persistence.

## Deliverables checklist

- [ ] FastAPI endpoint deployed
- [ ] Streamlit URL deployed
- [ ] README polished and updated with your final URLs
- [ ] Supabase project configured
- [ ] Ollama model installed locally

## Git workflow

Use Git CLI only:

```bash
git init
git add .
git commit -m "Complete salary prediction application"
git remote add origin <repo-url>
git push -u origin main
```
