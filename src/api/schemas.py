"""Pydantic data schemas for FastAPI request and response validation."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Schema representing an individual customer profile for churn prediction."""

    customerID: Optional[str] = Field(default=None, description="Optional customer unique identifier", examples=["7590-VHVEG"])
    gender: str = Field(..., description="Customer gender: 'Male' or 'Female'", examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is a senior citizen (1: Yes, 0: No)", examples=[0])
    Partner: str = Field(..., description="Whether customer has a partner ('Yes' or 'No')", examples=["Yes"])
    Dependents: str = Field(..., description="Whether customer has dependents ('Yes' or 'No')", examples=["No"])
    tenure: int = Field(..., ge=0, description="Number of months customer has stayed with company", examples=[1])
    PhoneService: str = Field(..., description="Whether customer has phone service ('Yes' or 'No')", examples=["No"])
    MultipleLines: str = Field(..., description="Whether customer has multiple lines ('Yes', 'No', 'No phone service')", examples=["No phone service"])
    InternetService: str = Field(..., description="Customer internet service provider ('DSL', 'Fiber optic', 'No')", examples=["DSL"])
    OnlineSecurity: str = Field(..., description="Whether customer has online security ('Yes', 'No', 'No internet service')", examples=["No"])
    OnlineBackup: str = Field(..., description="Whether customer has online backup ('Yes', 'No', 'No internet service')", examples=["Yes"])
    DeviceProtection: str = Field(..., description="Whether customer has device protection ('Yes', 'No', 'No internet service')", examples=["No"])
    TechSupport: str = Field(..., description="Whether customer has tech support ('Yes', 'No', 'No internet service')", examples=["No"])
    StreamingTV: str = Field(..., description="Whether customer has streaming TV ('Yes', 'No', 'No internet service')", examples=["No"])
    StreamingMovies: str = Field(..., description="Whether customer has streaming movies ('Yes', 'No', 'No internet service')", examples=["No"])
    Contract: str = Field(..., description="Contract term ('Month-to-month', 'One year', 'Two year')", examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., description="Whether customer has paperless billing ('Yes' or 'No')", examples=["Yes"])
    PaymentMethod: str = Field(
        ...,
        description="Payment method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')",
        examples=["Electronic check"],
    )
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly fee charged to customer ($)", examples=[29.85])
    TotalCharges: Union[float, str] = Field(..., description="Total amount charged over tenure ($)", examples=[29.85])

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for individual customer churn prediction."""

    customer_id: Optional[str] = Field(None, description="Customer identifier if provided")
    churn_prediction: int = Field(..., description="Binary churn prediction (1 = Will Churn, 0 = Will Stay)")
    churn_label: str = Field(..., description="Readable churn label ('Churn' or 'Stay')")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of churning")
    risk_level: str = Field(..., description="Churn risk category ('Low', 'Medium', 'High')")
    recommended_action: str = Field(..., description="Actionable retention recommendation based on churn risk")


class BatchCustomerInput(BaseModel):
    """Schema for batch prediction requests."""

    customers: List[CustomerInput] = Field(..., min_length=1, description="List of customer profiles to evaluate")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    total_customers: int = Field(..., description="Total number of evaluated customers")
    predicted_churners: int = Field(..., description="Total count of predicted churners")
    predicted_retained: int = Field(..., description="Total count of predicted retained customers")
    churn_rate_percent: float = Field(..., description="Percentage of customers flagged at churn risk")
    predictions: List[PredictionResponse] = Field(..., description="Individual customer predictions")


class HealthResponse(BaseModel):
    """Schema for health status response."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_name: Optional[str] = Field(None, examples=["gradient_boosting"])
    version: str = Field(..., examples=["1.0.0"])


class ModelMetadataResponse(BaseModel):
    """Schema for detailed model metadata and metrics."""

    model_name: str
    trained_at: str
    cv_roc_auc_score: float
    best_hyperparameters: Dict[str, Any]
    test_metrics: Dict[str, float]
    business_impact: Dict[str, Any]
    all_models_summary: List[Dict[str, Any]]
