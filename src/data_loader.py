"""Data loading, cleaning, validation, and splitting module."""

import logging
from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN, ID_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load raw churn dataset from a CSV file.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file is empty or missing expected target.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    logger.info("Successfully loaded raw dataset with shape: %s", df.shape)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw churn dataset:
    - Trims leading/trailing whitespaces in string columns.
    - Safely converts TotalCharges to numeric, replacing whitespace with NaN.
    - Imputes missing TotalCharges with 0.0 (corresponds to brand-new customers with tenure == 0).
    - Maps target column 'Churn' ('Yes' -> 1, 'No' -> 0).

    Args:
        df: Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    # Strip whitespaces from string-like columns
    for col in df_clean.columns:
        if pd.api.types.is_string_dtype(df_clean[col]) or df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str).str.strip()

    # Convert TotalCharges to numeric, coercing invalid spaces to NaN
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        null_count = df_clean["TotalCharges"].isna().sum()
        if null_count > 0:
            logger.info("Imputing %d missing TotalCharges values with 0.0 (tenure=0)", null_count)
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Convert Churn target to binary integer (1 for Yes, 0 for No) if present
    if TARGET_COLUMN in df_clean.columns:
        if pd.api.types.is_string_dtype(df_clean[TARGET_COLUMN]) or df_clean[TARGET_COLUMN].dtype == "object":
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(str).str.strip().map({"Yes": 1, "No": 0})
            logger.info("Target distribution:\n%s", df_clean[TARGET_COLUMN].value_counts())

    return df_clean


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified split of the dataset into training and testing sets.

    Args:
        df: Cleaned DataFrame containing features and target.
        target_col: Name of the target column.
        test_size: Fraction of samples assigned to the test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split dataset: Train=%d samples (churn=%.2f%%), Test=%d samples (churn=%.2f%%)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test
