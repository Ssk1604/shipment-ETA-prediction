"""
FastAPI Application for Aiqon Shipment ETA Prediction & Delay Propagation System.

Provides robust REST endpoints for:
- System Health Monitoring
- Shipment Registry Exploration
- Real-Time ETA & Delay Risk Prediction
- Multi-Stage Delay Propagation Simulation
- Machine Learning Model Metrics & Feature Explainability
"""

import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    FeatureImportanceItem,
    HealthResponse,
    ModelMetricsResponse,
    PredictionResponse,
    ShipmentInput,
    SimulationRequest,
    SimulationResponse,
)
from app.services import get_shipment_service

app = FastAPI(
    title="Aiqon Logistics AI - Shipment ETA & Delay Propagation API",
    description=(
        "Production-grade ML API predicting maritime container ETA, delay risk probabilities, "
        "and modeling physical multi-stage delay propagation with schedule slack absorption."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
def health_check():
    """Confirms API service is alive and ML models are loaded in memory."""
    service = get_shipment_service()
    models_ready = (
        service.engine is not None
        and service.engine.reg_pipeline is not None
        and service.engine.clf_pipeline is not None
    )
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_ready,
    )


@app.get(
    "/shipments",
    tags=["Shipments"],
    summary="Retrieve shipment records",
)
def get_shipments(
    limit: int = Query(50, ge=1, le=250, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    route: Optional[str] = Query(None, description="Filter by route code (e.g. ASIA-EUR-01)"),
    is_delayed: Optional[int] = Query(None, ge=0, le=1, description="Filter by delay status (1 = delayed >=24h, 0 = on-time)"),
):
    """Retrieves paginated and filtered historical/operational shipments."""
    service = get_shipment_service()
    return service.get_shipments(limit=limit, offset=offset, route=route, is_delayed=is_delayed)


@app.get(
    "/shipments/{shipment_id}",
    tags=["Shipments"],
    summary="Retrieve single shipment by ID",
)
def get_shipment(shipment_id: str):
    """Retrieves full telemetry and operational parameters for a specific shipment."""
    service = get_shipment_service()
    record = service.get_shipment_by_id(shipment_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipment with ID '{shipment_id}' was not found.",
        )
    return record


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction & Explainability"],
    summary="Predict shipment ETA and delay risk",
)
def predict_eta(payload: ShipmentInput):
    """
    Predicts:
    1. Predicted delay in hours (RandomForestRegressor).
    2. Dynamic estimated delivery timestamp (ETA).
    3. Delay risk probability (RandomForestClassifier).
    4. Risk category tier (LOW, MEDIUM, HIGH).
    5. Local stage-level delay attribution waterfall.
    """
    service = get_shipment_service()
    try:
        prediction = service.predict(payload.model_dump())
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post(
    "/simulate-delay",
    response_model=SimulationResponse,
    tags=["Simulation & Propagation"],
    summary="Simulate bottleneck disruption and downstream delay propagation",
)
def simulate_delay(payload: SimulationRequest):
    """
    Simulates injecting additional disruption hours at a designated stage (port, customs, or transit).
    Applies the mathematical schedule slack absorption to determine true downstream ETA impact.
    """
    service = get_shipment_service()
    try:
        shipment_data = payload.shipment_data.model_dump() if payload.shipment_data else None
        result = service.simulate(
            disruption_stage=payload.disruption_stage,
            additional_hours=payload.additional_disruption_hours,
            shipment_id=payload.shipment_id,
            shipment_data=shipment_data,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation error: {str(e)}",
        )


@app.get(
    "/model/metrics",
    response_model=ModelMetricsResponse,
    tags=["Model Insights"],
    summary="Retrieve model evaluation metrics",
)
def get_model_metrics():
    """Returns test set evaluation metrics for Baseline, Model A, and Model B."""
    service = get_shipment_service()
    try:
        return service.get_metrics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load metrics: {str(e)}",
        )


@app.get(
    "/model/feature-importance",
    response_model=List[FeatureImportanceItem],
    tags=["Model Insights"],
    summary="Retrieve global feature importance",
)
def get_feature_importance():
    """Returns normalized Gini feature importances across all engineered variables."""
    service = get_shipment_service()
    try:
        return service.get_feature_importance()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load feature importance: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
