"""Preprocessing and feature engineering transformers for the customer churn pipeline."""

import logging
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    ID_COLUMN,
)

logger = logging.getLogger(__name__)

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn transformer that creates domain-engineered features:
    1. 'tenure_group': Segmenting customer tenure into lifecycle buckets ('0-12m', '13-24m', etc.).
    2. 'avg_monthly_spend': Ratio of TotalCharges to effective tenure (TotalCharges / (tenure + 1)).
    3. 'monthly_to_total_ratio': Ratio of MonthlyCharges to TotalCharges (MonthlyCharges / (TotalCharges + 1)).
    4. 'total_services_count': Number of active value-add telecom services subscribed to.
    """

    def __init__(self, drop_id: bool = True):
        self.drop_id = drop_id

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from raw customer input.

        Args:
            X: Input pandas DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with newly engineered features.
        """
        # Ensure we work on a copy
        df = X.copy()

        # Handle customerID column if present
        if self.drop_id and ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])

        # Ensure TotalCharges is numeric, fill NaNs with 0.0
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        # Ensure tenure and MonthlyCharges are numeric
        if "tenure" in df.columns:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

        # Feature 1: tenure_group (Lifecycle stage)
        # 0-12m: New / Trial / High Risk
        # 13-24m: Early Tenure
        # 25-48m: Mid Tenure
        # 49-60m: Established
        # 61-72m: Loyal / Long-term
        bins = [-1, 12, 24, 48, 60, 100]
        labels = ["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"]
        df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

        # Feature 2: avg_monthly_spend (historical average monthly charge)
        # Avoid division by zero by adding 1 month
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1.0)

        # Feature 3: monthly_to_total_ratio (relative magnitude of monthly bill to cumulative spend)
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)

        # Feature 4: total_services_count (number of active telecom services)
        active_services = pd.Series(0, index=df.index, dtype=int)
        for col in SERVICE_COLUMNS:
            if col in df.columns:
                active_services += (df[col].astype(str).str.lower() == "yes").astype(int)
        df["total_services_count"] = active_services

        # Ensure SeniorCitizen is treated as categorical/string for one-hot encoding consistency
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

        return df


def build_column_transformer() -> ColumnTransformer:
    """Build the Scikit-learn ColumnTransformer for numerical scaling and categorical encoding.

    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    # Numerical pipeline: Imputation + Standard Scaling
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: Imputation + One-Hot Encoding
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ALL_NUMERICAL_FEATURES),
            ("cat", cat_pipeline, ALL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_full_preprocessing_pipeline() -> Pipeline:
    """Assemble the complete preprocessing pipeline combining custom feature engineering,
    missing value imputation, standard scaling, and one-hot encoding.

    Returns:
        Pipeline: Complete preprocessing pipeline.
    """
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("column_transformer", build_column_transformer()),
        ]
    )
