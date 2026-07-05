"""Pydantic request/response models for the API.

Field names mirror the Telco feature contract in config.py. Categorical fields
are constrained with Literals so invalid categories are rejected at the edge
with a 422 rather than silently mishandled by the model.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """One customer record to score."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.5,
                "TotalCharges": 445.25,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: bool
    risk_band: Literal["low", "medium", "high"]
    decision_threshold: float
    model_version: str


class BatchRequest(BaseModel):
    records: list[CustomerFeatures]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
