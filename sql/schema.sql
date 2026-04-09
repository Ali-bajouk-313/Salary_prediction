create table if not exists prediction_runs (
    id bigint generated always as identity primary key,
    run_id uuid not null unique,
    created_at timestamptz not null default now(),
    work_year int not null,
    experience_level text not null,
    employment_type text not null,
    job_title text not null,
    employee_residence text not null,
    remote_ratio int not null,
    company_location text not null,
    company_size text not null,
    predicted_salary_usd numeric not null,
    llm_analysis text,
    chart_base64 text
);
