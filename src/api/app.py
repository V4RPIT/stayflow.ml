"""Production FastAPI REST API application for serving the Customer Churn Prediction Pipeline."""

from contextlib import asynccontextmanager
import json
import logging
from typing import Any, Dict, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchCustomerInput,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
)
from src.config import METADATA_FILE_PATH, MODEL_FILE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("churn_api")

# Model artifacts holder in app memory
app_state: Dict[str, Any] = {
    "pipeline": None,
    "metadata": None,
}


def load_artifacts() -> None:
    """Load trained pipeline and metadata from disk into memory."""
    if MODEL_FILE_PATH.exists():
        logger.info("Loading model pipeline from %s", MODEL_FILE_PATH)
        app_state["pipeline"] = joblib.load(MODEL_FILE_PATH)
    else:
        logger.warning("Model file not found at %s. Please run training pipeline first.", MODEL_FILE_PATH)

    if METADATA_FILE_PATH.exists():
        logger.info("Loading model metadata from %s", METADATA_FILE_PATH)
        with open(METADATA_FILE_PATH, "r", encoding="utf-8") as f:
            app_state["metadata"] = json.load(f)
    else:
        logger.warning("Metadata file not found at %s.", METADATA_FILE_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for FastAPI startup and shutdown."""
    load_artifacts()
    yield
    app_state.clear()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-ready machine learning service for predicting customer churn in subscription businesses. "
        "Engineered with Scikit-learn, FastAPI, and Pydantic v2."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin web client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_risk_and_action(churn_prob: float) -> tuple[str, str]:
    """Determine risk level and actionable retention recommendation based on churn probability."""
    if churn_prob >= 0.70:
        risk_level = "High"
        recommendation = (
            "Urgent: Customer at critical churn risk. Proactively trigger high-impact retention intervention: "
            "offer a 12-month contract renewal discount or assign a dedicated customer success agent."
        )
    elif churn_prob >= 0.40:
        risk_level = "Medium"
        recommendation = (
            "Caution: Moderate churn risk detected. Engage via personalized satisfaction check-in, "
            "feature adoption guides, or targeted loyalty incentive."
        )
    else:
        risk_level = "Low"
        recommendation = (
            "Healthy: Low churn likelihood. Ideal candidate for loyalty recognition, upselling premium features, "
            "or referral program enrollment."
        )
    return risk_level, recommendation


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """Root welcome endpoint providing service information."""
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Service health check endpoint."""
    is_loaded = app_state["pipeline"] is not None
    model_name = app_state["metadata"].get("model_name") if app_state["metadata"] else None

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=model_name,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelMetadataResponse, tags=["Model Info"])
def get_model_info() -> ModelMetadataResponse:
    """Retrieve detailed model architecture, cross-validation scores, test metrics, and business impact analysis."""
    if not app_state["metadata"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not loaded. Ensure training pipeline has been executed.",
        )
    return ModelMetadataResponse(**app_state["metadata"])


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerInput) -> PredictionResponse:
    """Predict churn probability and binary classification for an individual customer."""
    pipeline = app_state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not initialized. Run training pipeline first.",
        )

    try:
        input_data = customer.model_dump()
        df_input = pd.DataFrame([input_data])

        # Execute prediction through full Scikit-learn Pipeline
        pred_prob = float(pipeline.predict_proba(df_input)[0, 1])
        pred_binary = int(pipeline.predict(df_input)[0])

        risk_level, action = get_risk_and_action(pred_prob)

        return PredictionResponse(
            customer_id=customer.customerID,
            churn_prediction=pred_binary,
            churn_label="Churn" if pred_binary == 1 else "Stay",
            churn_probability=round(pred_prob, 4),
            risk_level=risk_level,
            recommended_action=action,
        )
    except Exception as exc:
        logger.error("Error during single prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_churn_batch(batch: BatchCustomerInput) -> BatchPredictionResponse:
    """Perform vectorized batch churn predictions for multiple customer profiles."""
    pipeline = app_state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not initialized. Run training pipeline first.",
        )

    try:
        data_records = [c.model_dump() for c in batch.customers]
        df_batch = pd.DataFrame(data_records)

        probs = pipeline.predict_proba(df_batch)[:, 1]
        preds = pipeline.predict(df_batch)

        predictions = []
        churn_count = 0

        for i, customer in enumerate(batch.customers):
            p_val = float(probs[i])
            b_val = int(preds[i])
            if b_val == 1:
                churn_count += 1

            risk, rec = get_risk_and_action(p_val)
            predictions.append(
                PredictionResponse(
                    customer_id=customer.customerID,
                    churn_prediction=b_val,
                    churn_label="Churn" if b_val == 1 else "Stay",
                    churn_probability=round(p_val, 4),
                    risk_level=risk,
                    recommended_action=rec,
                )
            )

        total = len(predictions)
        churn_pct = round((churn_count / total) * 100, 2) if total > 0 else 0.0

        return BatchPredictionResponse(
            total_customers=total,
            predicted_churners=churn_count,
            predicted_retained=total - churn_count,
            churn_rate_percent=churn_pct,
            predictions=predictions,
        )
    except Exception as exc:
        logger.error("Error during batch prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        )
