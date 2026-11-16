"""
Integration and End-to-End API tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_get_shipments():
    response = client.get("/shipments?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10
    if len(data) > 0:
        assert "shipment_id" in data[0]


def test_get_shipment_by_id_found():
    response = client.get("/shipments/SHP-2026-0001")
    assert response.status_code == 200
    data = response.json()
    assert data["shipment_id"] == "SHP-2026-0001"


def test_get_shipment_by_id_not_found():
    response = client.get("/shipments/SHP-DOES-NOT-EXIST")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_post_predict_valid():
    payload = {
        "shipment_id": "API-TEST-01",
        "route": "ASIA-EUR-01",
        "origin": "CNSHA",
        "destination": "NLRTM",
        "distance_km": 19500.0,
        "carrier_rating": 0.88,
        "scheduled_departure": "2026-04-01T08:00:00",
        "departure_delay_hours": 4.0,
        "port_arrival_delay_hours": 6.0,
        "port_congestion": 0.35,
        "weather_severity": 2.5,
        "port_processing_hours": 30.0,
        "customs_processing_hours": 18.0,
        "scheduled_delivery": "2026-04-28T12:00:00",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_delay_hours" in data
    assert "predicted_eta" in data
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "explanation" in data


def test_post_predict_batch():
    batch_payload = {
        "shipments": [
            {
                "shipment_id": f"BATCH-{i:03d}",
                "route": "ASIA-EUR-01",
                "origin": "CNSHA",
                "destination": "NLRTM",
                "distance_km": 19500.0,
                "carrier_rating": 0.85,
                "scheduled_departure": "2026-04-01T08:00:00",
                "departure_delay_hours": float(i * 2),
                "port_congestion": 0.4,
                "weather_severity": 2.5,
                "port_processing_hours": 30.0,
                "customs_processing_hours": 18.0,
                "scheduled_delivery": "2026-04-28T12:00:00",
            }
            for i in range(5)
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 5
    assert len(data["predictions"]) == 5
    assert "processing_time_ms" in data
    assert data["predictions"][0]["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]


def test_post_predict_invalid_validation():
    payload = {
        "shipment_id": "API-TEST-INVALID",
        "distance_km": -500.0,  # Invalid negative distance
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_simulate_delay_with_existing_shipment():
    payload = {
        "shipment_id": "SHP-2026-0001",
        "disruption_stage": "port",
        "additional_disruption_hours": 18.0,
    }
    response = client.post("/simulate-delay", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["stage_disrupted"] == "port"
    assert data["additional_delay_added_hours"] == 18.0
    assert data["revised_predicted_delay_hours"] >= data["original_predicted_delay_hours"]


def test_simulate_delay_with_unknown_shipment():
    payload = {
        "shipment_id": "SHP-UNKNOWN-999",
        "disruption_stage": "customs",
        "additional_disruption_hours": 10.0,
    }
    response = client.post("/simulate-delay", json=payload)
    assert response.status_code == 400


def test_model_metrics_endpoint():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "baseline_regression" in data
    assert "model_a_regression" in data
    assert "model_b_classification" in data


def test_feature_importance_endpoint():
    response = client.get("/model/feature-importance")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "feature" in data[0]
    assert "importance" in data[0]
