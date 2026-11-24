"""Integration tests for the FastAPI REST API service endpoints."""

from fastapi.testclient import TestClient
import pytest

from src.api.app import app, load_artifacts


@pytest.fixture(scope="module")
def client():
    """Create a test client instance with pre-loaded model artifacts."""
    load_artifacts()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test the root welcome endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] is not None


def test_model_info_endpoint(client):
    """Test model information and metrics retrieval endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "cv_roc_auc_score" in data
    assert "test_metrics" in data
    assert data["test_metrics"]["roc_auc"] > 0.80
    assert "all_models_summary" in data


def test_predict_single_customer_valid(client):
    """Test single customer churn prediction with valid payload."""
    payload = {
        "customerID": "TEST-1001",
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

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "TEST-1001"
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Churn", "Stay"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert len(data["recommended_action"]) > 10


def test_predict_risk_stratification(client):
    """Verify that a high-risk profile yields a higher churn probability than a loyal low-risk profile."""
    high_risk_customer = {
        "customerID": "HIGH-RISK-01",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
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
        "MonthlyCharges": 105.5,
        "TotalCharges": 105.5,
    }

    loyal_customer = {
        "customerID": "LOYAL-01",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 70,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 3150.0,
    }

    resp_high = client.post("/predict", json=high_risk_customer)
    resp_loyal = client.post("/predict", json=loyal_customer)

    assert resp_high.status_code == 200
    assert resp_loyal.status_code == 200

    prob_high = resp_high.json()["churn_probability"]
    prob_loyal = resp_loyal.json()["churn_probability"]

    # High risk should have substantially higher churn probability than loyal customer
    assert prob_high > prob_loyal
    assert prob_high > 0.50
    assert prob_loyal < 0.20


def test_predict_batch_endpoint(client):
    """Test batch prediction endpoint with multiple customer records."""
    batch_payload = {
        "customers": [
            {
                "customerID": "CUST-001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0,
            },
            {
                "customerID": "CUST-002",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
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
                "MonthlyCharges": 98.0,
                "TotalCharges": 196.0,
            },
        ]
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert data["predicted_churners"] + data["predicted_retained"] == 2
    assert 0.0 <= data["churn_rate_percent"] <= 100.0


def test_predict_validation_error(client):
    """Test request validation error when required field is missing."""
    invalid_payload = {
        "gender": "Female",
        # Missing tenure, Contract, MonthlyCharges, etc.
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Pydantic unprocessable entity validation error
