"""Model training and cross-validated hyperparameter tuning module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    METADATA_FILE_PATH,
    MODEL_FILE_PATH,
    PARAM_GRIDS,
)
from src.preprocessor import build_full_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_candidate_models() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Return dictionary of candidate classifiers paired with their parameter grids.

    Returns:
        Dict: Mapping model_name -> (estimator_instance, param_grid)
    """
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            PARAM_GRIDS["logistic_regression"],
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            PARAM_GRIDS["random_forest"],
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            PARAM_GRIDS["gradient_boosting"],
        ),
    }
    return candidates


def train_and_tune_model(
    model_name: str,
    base_model: Any,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
) -> GridSearchCV:
    """Construct an end-to-end pipeline and optimize hyperparameters with GridSearchCV.

    Args:
        model_name: Name identifier for the model.
        base_model: Scikit-learn classification estimator.
        param_grid: Parameter grid dictionary for tuning.
        X_train: Training features DataFrame.
        y_train: Training target labels Series.
        cv_folds: Number of stratified cross-validation folds.
        scoring: Evaluation metric to optimize during CV search.

    Returns:
        GridSearchCV: Fitted GridSearchCV object containing the best estimator.
    """
    logger.info("Starting hyperparameter tuning for %s (cv=%d, metric=%s)...", model_name, cv_folds, scoring)

    # Build full pipeline: Preprocessing + Classifier
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_full_preprocessing_pipeline()),
            ("classifier", base_model),
        ]
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    logger.info(
        "Finished tuning %s. Best %s score: %.4f | Best params: %s",
        model_name,
        scoring,
        grid_search.best_score_,
        grid_search.best_params_,
    )
    return grid_search


def train_and_tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
) -> Dict[str, GridSearchCV]:
    """Train and tune all candidate models.

    Args:
        X_train: Training features.
        y_train: Training target.
        cv_folds: Number of CV folds.
        scoring: Metric to optimize.

    Returns:
        Dict: Mapping model_name -> fitted GridSearchCV instance.
    """
    candidates = get_candidate_models()
    fitted_searches = {}

    for name, (estimator, grid) in candidates.items():
        search = train_and_tune_model(
            model_name=name,
            base_model=estimator,
            param_grid=grid,
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds,
            scoring=scoring,
        )
        fitted_searches[name] = search

    return fitted_searches


def save_model_pipeline(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: Path = MODEL_FILE_PATH,
    metadata_path: Path = METADATA_FILE_PATH,
) -> None:
    """Serialize the trained end-to-end pipeline and its metadata to disk.

    Args:
        pipeline: Trained Scikit-learn Pipeline.
        metadata: Metadata dictionary containing model info and metrics.
        model_path: Path to target .joblib file.
        metadata_path: Path to target .json metadata file.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    logger.info("Saved serialized model pipeline to: %s", model_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved model metadata to: %s", metadata_path)
