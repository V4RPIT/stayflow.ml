"""Live verification script demonstrating API endpoints with realistic customer profiles."""

import json
from fastapi.testclient import TestClient
from src.api.app import app, load_artifacts


def main():
    load_artifacts()
    client = TestClient(app)

    print("=" * 60)
    print("1. Testing GET /health")
    resp = client.get("/health")
    print("Status:", resp.status_code)
    print("Response:", json.dumps(resp.json(), indent=2))

    print("\n" + "=" * 60)
    print("2. Testing GET /model-info")
    resp = client.get("/model-info")
    print("Status:", resp.status_code)
    model_info = resp.json()
    print("Model:", model_info["model_name"])
    print("CV ROC-AUC:", model_info["cv_roc_auc_score"])
    print("Test Metrics:", json.dumps(model_info["test_metrics"], indent=2))
    print("Business Impact Summary (Optimal Threshold):", json.dumps(model_info["business_impact"]["optimal_threshold"], indent=2))

    print("\n" + "=" * 60)
    print("3. Testing POST /predict (High-Risk Customer: Month-to-month, Fiber Optic, Short Tenure)")
    high_risk_payload = {
        "customerID": "HIGH-RISK-01",
        "gender": "Female",
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
        "MonthlyCharges": 105.50,
        "TotalCharges": 105.50,
    }
    resp = client.post("/predict", json=high_risk_payload)
    print("Status:", resp.status_code)
    print("Response:", json.dumps(resp.json(), indent=2))

    print("\n" + "=" * 60)
    print("4. Testing POST /predict (Low-Risk Customer: Two-Year Contract, High Tenure, Bundled Services)")
    low_risk_payload = {
        "customerID": "LOYAL-01",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
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
        "MonthlyCharges": 55.00,
        "TotalCharges": 3575.00,
    }
    resp = client.post("/predict", json=low_risk_payload)
    print("Status:", resp.status_code)
    print("Response:", json.dumps(resp.json(), indent=2))

    print("\n" + "=" * 60)
    print("5. Testing POST /predict/batch (Batch Prediction of Both Customers)")
    batch_payload = {"customers": [high_risk_payload, low_risk_payload]}
    resp = client.post("/predict/batch", json=batch_payload)
    print("Status:", resp.status_code)
    print("Response:", json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
