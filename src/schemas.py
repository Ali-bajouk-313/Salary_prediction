from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ExperienceLevel = Literal['EN', 'EX', 'MI', 'SE']
EmploymentType = Literal['CT', 'FL', 'FT', 'PT']
CompanySize = Literal['L', 'M', 'S']


class PredictionInput(BaseModel):
    work_year: int = Field(ge=2020, le=2030)
    experience_level: ExperienceLevel
    employment_type: EmploymentType
    job_title: str = Field(min_length=2, max_length=120)
    employee_residence: str = Field(min_length=2, max_length=2)
    remote_ratio: int = Field(ge=0, le=100)
    company_location: str = Field(min_length=2, max_length=2)
    company_size: CompanySize

    @field_validator('employee_residence', 'company_location')
    @classmethod
    def uppercase_country_code(cls, value: str) -> str:
        return value.upper()

    @field_validator('job_title')
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return ' '.join(value.split())


class PredictionResponse(BaseModel):
    input_data: PredictionInput
    predicted_salary_usd: float
    model_version: str
    confidence_note: str
