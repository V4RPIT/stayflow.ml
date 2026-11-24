"""Unit tests for model training, tuning configurations, and evaluation metrics."""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from src.config import METADATA_FILE_PATH, MODEL_FILE_PATH
from src.evaluate import (
    compute_business_impact,
    find_optimal_business_threshold,
)
from src.train import get_candidate_models


def test_candidate_models_structure():
    """Verify candidate models dictionary contains all required classifiers."""
    candidates = get_candidate_models()
    expected_models = {"logistic_regression", "random_forest", "gradient_boosting"}
    assert expected_models.issubset(set(candidates.keys()))

    for name, (estimator, grid) in candidates.items():
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict_proba")
        assert isinstance(grid, dict)
        assert len(grid) > 0


def test_compute_business_impact():
    """Verify financial cost-benefit calculation mechanics."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.4, 0.1, 0.85, 0.3])

    # With threshold 0.5:
    # y_pred = [1, 1, 1, 0, 0, 0, 1, 0]
    # TP: 3 (indices 0, 1, 6)
    # FP: 1 (index 2)
    # FN: 1 (index 4)
    # TN: 3 (indices 3, 5, 7)
    params = {
        "clv_saved_per_customer": 500.0,
        "retention_offer_cost": 50.0,
        "offer_acceptance_rate": 0.50,
        "organic_churn_cost": 500.0,
    }

    metrics = compute_business_impact(y_true, y_prob, threshold=0.5, params=params)

    assert metrics["tp"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 3
    assert metrics["customers_targeted"] == 4  # TP + FP
    assert metrics["campaign_cost"] == 4 * 50.0  # $200
    assert metrics["customers_saved"] == 3 * 0.50  # 1.5 customers
    assert metrics["gross_value_saved"] == 1.5 * 500.0  # $750
    assert metrics["net_value_added"] == 750.0 - 200.0  # $550
    assert np.isclose(metrics["roi_percent"], (550.0 / 200.0) * 100)  # 275%


def test_find_optimal_business_threshold():
    """Verify threshold optimization finds a threshold yielding maximum net profit."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.4, 0.1, 0.85, 0.3])

    best_thresh, best_metrics = find_optimal_business_threshold(y_true, y_prob)
    assert 0.05 <= best_thresh <= 0.95
    assert best_metrics["net_value_added"] >= 0


def test_saved_model_artifact():
    """Verify serialized model pipeline artifact exists, loads properly, and performs inference."""
    assert MODEL_FILE_PATH.exists()
    assert METADATA_FILE_PATH.exists()

    pipeline = joblib.load(MODEL_FILE_PATH)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")

    with open(METADATA_FILE_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert "model_name" in metadata
    assert "cv_roc_auc_score" in metadata
    assert metadata["cv_roc_auc_score"] > 0.80
    assert "test_metrics" in metadata
    assert metadata["test_metrics"]["roc_auc"] > 0.80
