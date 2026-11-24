"""Project configuration, feature definitions, hyperparameter grids, and business constants."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_churn_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE_PATH = MODEL_DIR / "best_churn_pipeline.joblib"
METADATA_FILE_PATH = MODEL_DIR / "model_metadata.json"
REPORT_DIR = PROJECT_ROOT / "reports"

# Ensure runtime directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

# Feature Schema Definitions
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Raw numerical features expected in incoming data
BASE_NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Engineered numerical features created by ChurnFeatureEngineer
ENGINEERED_NUMERICAL_FEATURES = [
    "avg_monthly_spend",
    "monthly_to_total_ratio",
    "total_services_count",
]

# All numerical features fed into scaling step
ALL_NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

# Categorical features
BASE_CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered categorical feature
ENGINEERED_CATEGORICAL_FEATURES = [
    "tenure_group",
]

# All categorical features fed into One-Hot Encoding step
ALL_CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

# Hyperparameter search grids
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_iter": [1000],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__class_weight": [None, "balanced"],
        "classifier__random_state": [42],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 150],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5],
        "classifier__subsample": [0.8, 1.0],
        "classifier__random_state": [42],
    },
}

# Business Impact Parameters
BUSINESS_PARAMS = {
    "clv_saved_per_customer": 500.0,     # Customer Lifetime Value preserved if retained ($)
    "retention_offer_cost": 50.0,        # Cost to offer incentive / discount per targeted customer ($)
    "offer_acceptance_rate": 0.50,       # Probability that an at-risk customer accepts the incentive
    "organic_churn_cost": 500.0,         # Cost if an at-risk customer churns without retention attempt ($)
}
