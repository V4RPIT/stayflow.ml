"""Unit tests for data loading, cleaning, and feature engineering transformers."""

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.data_loader import clean_raw_data, load_raw_data, split_data
from src.preprocessor import ChurnFeatureEngineer, build_full_preprocessing_pipeline


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provide a minimal raw dataframe fixture with typical edge cases."""
    return pd.DataFrame(
        {
            "customerID": ["001-A", "002-B", "003-C", "004-D", "005-E", "006-F"],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No"],
            "tenure": [1, 0, 48, 12, 24, 60],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes", "No"],
            "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["No", "No", "Yes", "Yes", "No internet service", "No"],
            "OnlineBackup": ["Yes", "No", "Yes", "No", "No internet service", "Yes"],
            "DeviceProtection": ["No", "No", "Yes", "No", "No internet service", "No"],
            "TechSupport": ["No", "No", "Yes", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "Yes", "Yes", "No", "No internet service", "No"],
            "StreamingMovies": ["No", "No", "Yes", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
                "Mailed check",
                "Electronic check",
            ],
            "MonthlyCharges": [29.85, 70.0, 55.5, 85.0, 20.0, 40.0],
            "TotalCharges": ["29.85", " ", "2600.0", "1020.0", "480.0", "2400.0"],  # Notice blank space for tenure=0
            "Churn": ["No", "Yes", "No", "Yes", "No", "Yes"],
        }
    )


def test_load_raw_data():
    """Verify raw dataset loading from path."""
    df = load_raw_data(RAW_DATA_PATH)
    assert not df.empty
    assert "Churn" in df.columns
    assert len(df) > 5000


def test_clean_raw_data(sample_raw_data):
    """Verify missing value handling and target mapping."""
    df_clean = clean_raw_data(sample_raw_data)

    # TotalCharges with blank space should be imputed to 0.0
    assert df_clean["TotalCharges"].iloc[1] == 0.0
    assert not df_clean["TotalCharges"].isna().any()

    # Target column should be mapped to integers 0 and 1
    assert set(df_clean["Churn"].unique()).issubset({0, 1})
    assert df_clean["Churn"].iloc[0] == 0
    assert df_clean["Churn"].iloc[1] == 1


def test_split_data(sample_raw_data):
    """Verify stratified split mechanics."""
    df_clean = clean_raw_data(sample_raw_data)
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.33, random_state=42)

    assert len(X_train) + len(X_test) == len(df_clean)
    assert len(y_train) + len(y_test) == len(df_clean)
    assert "Churn" not in X_train.columns


def test_churn_feature_engineer(sample_raw_data):
    """Verify custom feature engineering outputs."""
    df_clean = clean_raw_data(sample_raw_data)
    engineer = ChurnFeatureEngineer(drop_id=True)
    df_trans = engineer.transform(df_clean)

    # CustomerID dropped
    assert "customerID" not in df_trans.columns

    # 1. tenure_group
    assert "tenure_group" in df_trans.columns
    assert df_trans["tenure_group"].iloc[0] == "0-12m"
    assert df_trans["tenure_group"].iloc[2] == "25-48m"

    # 2. avg_monthly_spend
    assert "avg_monthly_spend" in df_trans.columns
    assert np.isclose(df_trans["avg_monthly_spend"].iloc[0], 29.85 / 2.0)

    # 3. monthly_to_total_ratio
    assert "monthly_to_total_ratio" in df_trans.columns
    assert df_trans["monthly_to_total_ratio"].iloc[0] > 0

    # 4. total_services_count
    assert "total_services_count" in df_trans.columns
    # Row 0: OnlineBackup='Yes' -> 1 service
    # Row 2: PhoneService='Yes', MultipleLines='Yes', OnlineSecurity='Yes', OnlineBackup='Yes',
    # DeviceProtection='Yes', TechSupport='Yes', StreamingTV='Yes', StreamingMovies='Yes' -> 8 services
    assert df_trans["total_services_count"].iloc[0] == 1
    assert df_trans["total_services_count"].iloc[2] == 8


def test_full_preprocessing_pipeline(sample_raw_data):
    """Verify full preprocessing pipeline handles scaling and one-hot encoding without NaNs."""
    df_clean = clean_raw_data(sample_raw_data)
    X = df_clean.drop(columns=["Churn"])

    pipeline = build_full_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)

    assert isinstance(X_processed, np.ndarray)
    assert not np.isnan(X_processed).any()
    assert X_processed.shape[0] == len(X)
    assert X_processed.shape[1] > 20  # Expanded after one-hot encoding
