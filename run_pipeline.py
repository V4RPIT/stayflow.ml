"""End-to-end execution pipeline for Customer Churn Prediction.

Runs:
1. Data loading and cleaning
2. Stratified train/test splitting
3. Training & hyperparameter tuning for Logistic Regression, Random Forest, and Gradient Boosting
4. Rigorous evaluation (ROC-AUC, PR-AUC, F1, Business Cost-Benefit analysis)
5. Model serialization to disk for REST API serving
6. Generating report artifacts and plots
"""

import datetime
import logging
from typing import Any, Dict
import pandas as pd

from src.config import (
    METADATA_FILE_PATH,
    MODEL_FILE_PATH,
    RAW_DATA_PATH,
    REPORT_DIR,
)
from src.data_loader import clean_raw_data, load_raw_data, split_data
from src.evaluate import evaluate_all_models, plot_evaluation_curves
from src.train import save_model_pipeline, train_and_tune_all_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("run_pipeline")


def run() -> Dict[str, Any]:
    """Execute the end-to-end churn prediction pipeline."""
    logger.info("=== STEP 1: Data Ingestion & Cleaning ===")
    df_raw = load_raw_data(RAW_DATA_PATH)
    df_clean = clean_raw_data(df_raw)

    logger.info("=== STEP 2: Stratified Train-Test Split ===")
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)

    logger.info("=== STEP 3: Model Training & Hyperparameter Tuning ===")
    fitted_searches = train_and_tune_all_models(X_train, y_train, cv_folds=5, scoring="roc_auc")

    # Extract best estimator pipelines
    best_pipelines = {name: search.best_estimator_ for name, search in fitted_searches.items()}

    logger.info("=== STEP 4: Comprehensive Model & Business Evaluation ===")
    summary_df, best_model_name, all_eval_results = evaluate_all_models(best_pipelines, X_test, y_test)

    # Generate visualization plots
    plot_path = REPORT_DIR / "model_evaluation_visualizations.png"
    plot_evaluation_curves(best_pipelines, X_test, y_test, output_path=plot_path)

    logger.info("=== STEP 5: Serializing Best Model Pipeline ===")
    best_pipeline = best_pipelines[best_model_name]
    best_search = fitted_searches[best_model_name]
    best_eval = all_eval_results[best_model_name]

    metadata = {
        "model_name": best_model_name,
        "trained_at": datetime.datetime.now().isoformat(),
        "cv_roc_auc_score": float(best_search.best_score_),
        "best_hyperparameters": best_search.best_params_,
        "test_metrics": {
            "roc_auc": best_eval["roc_auc"],
            "pr_auc": best_eval["pr_auc"],
            "accuracy": best_eval["accuracy"],
            "balanced_accuracy": best_eval["balanced_accuracy"],
            "precision": best_eval["precision"],
            "recall": best_eval["recall"],
            "f1_score": best_eval["f1_score"],
            "brier_score": best_eval["brier_score"],
        },
        "business_impact": {
            "default_threshold": best_eval["default_threshold_business"],
            "optimal_threshold": best_eval["optimal_threshold_business"],
        },
        "all_models_summary": summary_df.to_dict(orient="records"),
    }

    save_model_pipeline(
        pipeline=best_pipeline,
        metadata=metadata,
        model_path=MODEL_FILE_PATH,
        metadata_path=METADATA_FILE_PATH,
    )

    logger.info("=== STEP 6: Verification & Sample Inference ===")
    sample_customers = X_test.head(3)
    sample_preds = best_pipeline.predict(sample_customers)
    sample_probs = best_pipeline.predict_proba(sample_customers)[:, 1]

    for idx, (churn_pred, churn_prob) in enumerate(zip(sample_preds, sample_probs)):
        cust_id = sample_customers.iloc[idx].get("customerID", f"Sample-{idx+1}")
        logger.info(
            "Inference check - Customer: %s -> Churn Prediction: %d | Probability: %.4f",
            cust_id,
            churn_pred,
            churn_prob,
        )

    logger.info("Pipeline execution completed successfully!")
    return metadata


if __name__ == "__main__":
    run()
