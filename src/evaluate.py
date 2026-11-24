"""Model evaluation, business impact analysis, and visualization module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import BUSINESS_PARAMS, REPORT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_business_impact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    params: Dict[str, float] = BUSINESS_PARAMS,
) -> Dict[str, float]:
    """Calculate the business cost-benefit metrics for a given classification threshold.

    Business Mechanics:
    - Positive class (1) = Customer intends to churn.
    - Targeted customers (predicted 1) receive a retention incentive costing `retention_offer_cost`.
    - True Positives (TP): Customer actually intended to churn.
      A fraction `offer_acceptance_rate` accept the offer, preventing churn and saving `clv_saved_per_customer`.
      Gross value saved = TP * offer_acceptance_rate * clv_saved_per_customer
    - False Positives (FP): Customer was going to stay anyway.
      Incentive cost is spent needlessly = FP * retention_offer_cost
    - Net Value Added = Gross Value Saved - Total Campaign Cost ((TP + FP) * retention_offer_cost)
    - Campaign ROI = (Net Value Added / Total Campaign Cost) * 100%

    Args:
        y_true: True binary target values.
        y_prob: Predicted churn probabilities.
        threshold: Decision threshold for predicting churn.
        params: Business parameters dict.

    Returns:
        Dict: Financial impact metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    clv = params["clv_saved_per_customer"]
    offer_cost = params["retention_offer_cost"]
    acceptance_rate = params["offer_acceptance_rate"]

    customers_targeted = int(tp + fp)
    campaign_cost = float(customers_targeted * offer_cost)
    customers_saved = float(tp * acceptance_rate)
    gross_value_saved = float(customers_saved * clv)
    net_value_added = float(gross_value_saved - campaign_cost)
    roi_percent = float((net_value_added / campaign_cost) * 100) if campaign_cost > 0 else 0.0

    # Churn capture rate (Recall of actual churners)
    churn_capture_rate = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "customers_targeted": customers_targeted,
        "campaign_cost": campaign_cost,
        "customers_saved": customers_saved,
        "gross_value_saved": gross_value_saved,
        "net_value_added": net_value_added,
        "roi_percent": roi_percent,
        "churn_capture_rate": churn_capture_rate,
    }


def find_optimal_business_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    params: Dict[str, float] = BUSINESS_PARAMS,
    thresholds: np.ndarray = np.linspace(0.05, 0.95, 91),
) -> Tuple[float, Dict[str, float]]:
    """Sweep decision thresholds to identify the one that maximizes net financial value added.

    Args:
        y_true: True binary labels.
        y_prob: Predicted churn probabilities.
        params: Business cost and benefit parameters.
        thresholds: Array of thresholds to evaluate.

    Returns:
        Tuple: (optimal_threshold, best_business_metrics)
    """
    best_value = -float("inf")
    best_metrics = {}
    best_thresh = 0.5

    for thresh in thresholds:
        metrics = compute_business_impact(y_true, y_prob, threshold=thresh, params=params)
        if metrics["net_value_added"] > best_value:
            best_value = metrics["net_value_added"]
            best_metrics = metrics
            best_thresh = float(thresh)

    return best_thresh, best_metrics


def evaluate_single_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate all standard machine learning metrics and business impact for a single model.

    Args:
        model: Trained pipeline with predict and predict_proba.
        X_test: Test features DataFrame.
        y_test: Test target Series.
        model_name: Identifier name of the model.
        threshold: Decision threshold for binary classification.

    Returns:
        Dict: Complete evaluation metrics.
    """
    y_true = np.asarray(y_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    # Business impact at default threshold
    default_biz = compute_business_impact(y_true, y_prob, threshold=threshold)

    # Optimal threshold for maximum business ROI
    optimal_thresh, optimal_biz = find_optimal_business_threshold(y_true, y_prob)

    results = {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "brier_score": brier,
        "confusion_matrix": default_biz["tp"],
        "default_threshold_business": default_biz,
        "optimal_threshold": optimal_thresh,
        "optimal_threshold_business": optimal_biz,
    }
    return results


def evaluate_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Evaluate all trained models, rank them, and select the best performer based on business impact and ROC-AUC.

    Args:
        models_dict: Dictionary mapping model names to fitted pipelines.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple: (comparison_dataframe, best_model_name, evaluation_results_dict)
    """
    all_results = {}
    summary_rows = []

    for name, model in models_dict.items():
        res = evaluate_single_model(model, X_test, y_test, model_name=name)
        all_results[name] = res

        summary_rows.append(
            {
                "Model": name,
                "ROC-AUC": round(res["roc_auc"], 4),
                "PR-AUC": round(res["pr_auc"], 4),
                "Accuracy": round(res["accuracy"], 4),
                "Balanced Acc": round(res["balanced_accuracy"], 4),
                "Precision": round(res["precision"], 4),
                "Recall": round(res["recall"], 4),
                "F1-Score": round(res["f1_score"], 4),
                "Net Benefit ($) [Thresh=0.5]": round(res["default_threshold_business"]["net_value_added"], 2),
                "Campaign ROI (%)": round(res["default_threshold_business"]["roi_percent"], 1),
                "Optimal Thresh": round(res["optimal_threshold"], 2),
                "Max Net Benefit ($)": round(res["optimal_threshold_business"]["net_value_added"], 2),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    # Select best model: highest net benefit at optimal threshold, breaking ties with ROC-AUC
    best_model_name = summary_df.sort_values(
        by=["Max Net Benefit ($)", "ROC-AUC"], ascending=[False, False]
    ).iloc[0]["Model"]

    logger.info("Evaluation Summary:\n%s", summary_df.to_string(index=False))
    logger.info("Best model selected based on business impact: %s", best_model_name)

    # Save comparison artifacts
    comparison_json_path = REPORT_DIR / "model_comparison.json"
    comparison_csv_path = REPORT_DIR / "model_comparison.csv"
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    summary_df.to_csv(comparison_csv_path, index=False)

    return summary_df, best_model_name, all_results


def plot_evaluation_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path = REPORT_DIR / "model_evaluation_visualizations.png",
) -> None:
    """Generate high-resolution evaluation figures:
    1. ROC Curves for all models
    2. Precision-Recall Curves for all models
    3. Net Financial Benefit vs Decision Threshold

    Args:
        models_dict: Dictionary mapping model names to fitted pipelines.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save the PNG file.
    """
    y_true = np.asarray(y_test)
    thresholds = np.linspace(0.05, 0.95, 91)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = {"logistic_regression": "#1f77b4", "random_forest": "#2ca02c", "gradient_boosting": "#ff7f0e"}

    # 1. ROC Curves
    ax_roc = axes[0]
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=colors.get(name), lw=2)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right")

    # 2. Precision-Recall Curves
    ax_pr = axes[1]
    baseline_churn_rate = y_true.mean()
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        ax_pr.plot(recall, precision, label=f"{name} (PR-AUC = {ap_score:.3f})", color=colors.get(name), lw=2)

    ax_pr.axhline(baseline_churn_rate, color="k", linestyle="--", alpha=0.6, label=f"Baseline ({baseline_churn_rate:.2f})")
    ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="upper right")

    # 3. Business Net Value Added Curve
    ax_biz = axes[2]
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        net_values = [
            compute_business_impact(y_true, y_prob, threshold=th)["net_value_added"]
            for th in thresholds
        ]
        best_val = max(net_values)
        best_th = thresholds[np.argmax(net_values)]
        ax_biz.plot(thresholds, net_values, label=f"{name} (Peak ${best_val:,.0f} @ {best_th:.2f})", color=colors.get(name), lw=2)

    ax_biz.axhline(0, color="k", linestyle="--", alpha=0.6, label="Break-Even ($0)")
    ax_biz.set_title("Business Net Value Added ($) vs Threshold", fontsize=12, fontweight="bold")
    ax_biz.set_xlabel("Decision Threshold", fontsize=10)
    ax_biz.set_ylabel("Net Financial Value Added ($)", fontsize=10)
    ax_biz.grid(True, alpha=0.3)
    ax_biz.legend(loc="lower center")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved evaluation visualization to: %s", output_path)
