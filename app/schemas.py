"""
Pydantic Schemas for FastAPI Request/Response Validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    models_loaded: bool = Field(..., json_schema_extra={"example": True})


class ShipmentInput(BaseModel):
    shipment_id: Optional[str] = Field("SHP-INPUT", json_schema_extra={"example": "SHP-2026-0001"})
    route: str = Field("ASIA-EUR-01", json_schema_extra={"example": "ASIA-EUR-01"})
    origin: str = Field("CNSHA", json_schema_extra={"example": "CNSHA"})
    destination: str = Field("NLRTM", json_schema_extra={"example": "NLRTM"})
    distance_km: float = Field(19500.0, ge=100.0, le=50000.0, json_schema_extra={"example": 19500.0})
    carrier_rating: float = Field(0.85, ge=0.0, le=1.0, json_schema_extra={"example": 0.88})
    scheduled_departure: Optional[str] = Field("2026-02-01T08:00:00", json_schema_extra={"example": "2026-02-01T08:00:00"})
    departure_delay_hours: float = Field(0.0, ge=-24.0, le=200.0, json_schema_extra={"example": 4.5})
    scheduled_port_arrival: Optional[str] = Field(None, json_schema_extra={"example": "2026-02-24T18:00:00"})
    actual_port_arrival: Optional[str] = Field(None, json_schema_extra={"example": "2026-02-25T02:00:00"})
    port_arrival_delay_hours: Optional[float] = Field(None, json_schema_extra={"example": 8.0})
    port_congestion: float = Field(0.35, ge=0.0, le=1.0, json_schema_extra={"example": 0.45})
    weather_severity: float = Field(2.0, ge=1.0, le=5.0, json_schema_extra={"example": 3.2})
    port_processing_hours: float = Field(28.0, ge=1.0, le=300.0, json_schema_extra={"example": 36.0})
    customs_processing_hours: float = Field(16.0, ge=1.0, le=300.0, json_schema_extra={"example": 24.0})
    scheduled_delivery: Optional[str] = Field("2026-02-28T12:00:00", json_schema_extra={"example": "2026-02-28T12:00:00"})


class StageAttribution(BaseModel):
    stage: str
    delay_contribution_hours: float
    description: str


class PredictionExplanation(BaseModel):
    shipment_id: str
    predicted_delay_hours: float
    delay_risk_category: str
    stages: List[StageAttribution]
    net_explained_delay_hours: float
    methodology_note: str


class PredictionResponse(BaseModel):
    shipment_id: str
    predicted_delay_hours: float
    predicted_eta: Optional[str]
    scheduled_delivery: Optional[str]
    delay_probability: float
    is_delayed_24h: int
    risk_tier: str
    explanation: Optional[PredictionExplanation] = None


class BatchPredictionItem(BaseModel):
    shipment_id: str
    predicted_delay_hours: float
    predicted_eta: Optional[str]
    scheduled_delivery: Optional[str]
    delay_probability: float
    is_delayed_24h: int
    risk_tier: str


class BatchPredictionRequest(BaseModel):
    shipments: List[ShipmentInput] = Field(..., min_length=1, max_length=5000)


class BatchPredictionResponse(BaseModel):
    total_processed: int
    processing_time_ms: float
    predictions: List[BatchPredictionItem]


class SimulationRequest(BaseModel):
    shipment_id: Optional[str] = Field(None, description="Optional ID of existing shipment to simulate")
    shipment_data: Optional[ShipmentInput] = Field(None, description="Custom shipment data if ID not provided")
    disruption_stage: str = Field(..., pattern="^(port|customs|transit)$", json_schema_extra={"example": "port"})
    additional_disruption_hours: float = Field(..., ge=0.0, le=300.0, json_schema_extra={"example": 24.0})


class SimulationResponse(BaseModel):
    shipment_id: str
    stage_disrupted: str
    additional_delay_added_hours: float
    absorbed_by_schedule_slack_hours: float
    net_effective_delay_increase_hours: float
    original_predicted_delay_hours: float
    revised_predicted_delay_hours: float
    original_eta: Optional[str]
    revised_eta: Optional[str]
    risk_tier: str


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ModelMetricsResponse(BaseModel):
    baseline_regression: Dict[str, Any]
    model_a_regression: Dict[str, Any]
    model_b_classification: Dict[str, Any]
    top_features: List[FeatureImportanceItem]
    split_info: Dict[str, Any]
