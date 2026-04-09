import uuid
from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_KEY

print(SUPABASE_KEY[:12])
print(len(SUPABASE_KEY))
print("..." in SUPABASE_KEY)

def test_insert():
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    payload = {
        "run_id": str(uuid.uuid4()),
        "work_year": 2013,
        "experience_level": "EX",
        "employment_type": "PT",
        "job_title": "Software Engineer",
        "employee_residence": "US",
        "remote_ratio": 50,
        "company_location": "US",
        "company_size": "B",
        "predicted_salary_usd": 130000,
        "llm_analysis": "test insert",
        "chart_base64": None,
    }
    response = client.table("prediction_runs").insert(payload).execute()
    print(response)
    return response
if __name__ == "__main__":
    test_insert()