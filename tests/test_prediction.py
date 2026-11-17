"""
Unit tests for Machine Learning Inference and Prediction Pipeline.
"""

import pytest
from src.predict import ShipmentInferenceEngine, get_inference_engine


@pytest.fixture
def engine():
    return get_inference_engine()


def test_predict_single_structure(engine):
    sample_shipment = {
        "shipment_id": "TEST-001",
        "route": "ASIA-EUR-01",
        "origin": "CNSHA",
        "destination": "NLRTM",
        "distance_km": 19500.0,
        "carrier_rating": 0.85,
        "scheduled_departure": "2026-03-01T08:00:00",
        "departure_delay_hours": 3.0,
        "scheduled_port_arrival": "2026-03-24T12:00:00",
        "actual_port_arrival": "2026-03-24T18:00:00",
        "port_arrival_delay_hours": 6.0,
        "port_congestion": 0.4,
        "weather_severity": 2.5,
        "port_processing_hours": 28.0,
        "customs_processing_hours": 18.0,
        "scheduled_delivery": "2026-03-28T12:00:00",
    }
    pred = engine.predict_single(sample_shipment)
    assert "predicted_delay_hours" in pred
    assert "predicted_delay_formatted" in pred
    assert "predicted_eta" in pred
    assert "delay_probability" in pred
    assert pred["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "explanation" in pred
    assert len(pred["explanation"]["stages"]) == 5


def test_predict_missing_telemetry(engine):
    # Test that missing port_arrival_delay_hours is imputed properly by pipeline
    sample_shipment = {
        "shipment_id": "TEST-MISSING",
        "route": "TPAC-01",
        "origin": "CNNBO",
        "destination": "USLAX",
        "distance_km": 11200.0,
        "carrier_rating": 0.70,
        "scheduled_departure": "2026-03-05T08:00:00",
        "departure_delay_hours": 12.0,
        "scheduled_port_arrival": "2026-03-18T12:00:00",
        "actual_port_arrival": None,
        "port_arrival_delay_hours": None,
        "port_congestion": 0.6,
        "weather_severity": 3.8,
        "port_processing_hours": 42.0,
        "customs_processing_hours": 30.0,
        "scheduled_delivery": "2026-03-22T12:00:00",
    }
    pred = engine.predict_single(sample_shipment)
    assert pred["predicted_delay_hours"] is not None
    assert pred["delay_probability"] >= 0.0


def test_predict_batch_throughput(engine):
    batch = [
        {
            "shipment_id": f"BULK-{i:04d}",
            "route": "ASIA-EUR-01",
            "origin": "CNSHA",
            "destination": "NLRTM",
            "distance_km": 19500.0,
            "carrier_rating": 0.85,
            "scheduled_departure": "2026-03-01T08:00:00",
            "departure_delay_hours": float(i % 10),
            "port_arrival_delay_hours": float((i % 10) + 2),
            "port_congestion": 0.4,
            "weather_severity": 2.5,
            "port_processing_hours": 28.0,
            "customs_processing_hours": 18.0,
            "scheduled_delivery": "2026-03-28T12:00:00",
        }
        for i in range(100)
    ]
    results = engine.predict_batch(batch)
    assert len(results) == 100
    assert results[0]["shipment_id"] == "BULK-0000"
    assert "predicted_delay_hours" in results[0]
    assert results[99]["shipment_id"] == "BULK-0099"
