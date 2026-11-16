"""
Comprehensive Unit Tests for the Four Advanced Feature Engineering Techniques:
1. Interaction Feature: congestion_x_missed_slot
2. Schedule Slack / Buffer Features: remaining_buffer_hours & unabsorbed_delay_hours
3. Missing Data Features: is_port_arrival_missing & training simulation/imputation
4. Route Variability Encoding & Distance Interactions: route_encoded, distance_km, weather_x_distance
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    LogisticsFeatureExtractor,
    build_preprocessor,
    create_feature_pipeline,
    prepare_data_splits,
    simulate_training_missingness,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ROUTE_DISTANCE_MAP,
)


# ==============================================================================
# 1. Congestion Penalty & Missed Slot Logic Tests
# ==============================================================================
def test_congestion_x_missed_slot_below_four_hours():
    """When arrival delay <= 4 hours, penalty is 1.0 (no penalty multiplier)."""
    extractor = LogisticsFeatureExtractor().fit(pd.DataFrame())
    
    # Arrival delay = 2.0h (< 4h), port_congestion = 0.5
    # Expected: 0.5 * 1.0 = 0.5
    df = pd.DataFrame([{
        "port_congestion": 0.5,
        "port_arrival_delay_hours": 2.0,
        "departure_delay_hours": 1.0,
    }])
    transformed = extractor.transform(df)
    assert transformed["congestion_x_missed_slot"].iloc[0] == pytest.approx(0.5, rel=1e-3)


def test_congestion_x_missed_slot_equal_to_four_hours():
    """When arrival delay == 4.0 hours, penalty is 1.0 (boundary condition: delay > 4 triggers penalty)."""
    extractor = LogisticsFeatureExtractor().fit(pd.DataFrame())
    
    # Arrival delay = 4.0h, port_congestion = 0.6
    # Expected: 0.6 * 1.0 = 0.6
    df = pd.DataFrame([{
        "port_congestion": 0.6,
        "port_arrival_delay_hours": 4.0,
        "departure_delay_hours": 2.0,
    }])
    transformed = extractor.transform(df)
    assert transformed["congestion_x_missed_slot"].iloc[0] == pytest.approx(0.6, rel=1e-3)


def test_congestion_x_missed_slot_above_four_hours():
    """When arrival delay > 4.0 hours, missed-slot penalty is 2.0 (doubles congestion impact)."""
    extractor = LogisticsFeatureExtractor().fit(pd.DataFrame())
    
    # Arrival delay = 6.5h (> 4h), port_congestion = 0.4
    # Expected: 0.4 * 2.0 = 0.8
    df = pd.DataFrame([{
        "port_congestion": 0.4,
        "port_arrival_delay_hours": 6.5,
        "departure_delay_hours": 3.0,
    }])
    transformed = extractor.transform(df)
    assert transformed["congestion_x_missed_slot"].iloc[0] == pytest.approx(0.8, rel=1e-3)


# ==============================================================================
# 2. Schedule Buffer & Slack Accounting Tests
# ==============================================================================
def test_buffer_accounting_delay_smaller_than_buffer():
    """
    Case A: Delay (8h) < Buffer (20h)
    remaining_buffer_hours = max(0, 20 - 8) = 12h
    unabsorbed_delay_hours = max(0, 8 - 20) = 0h
    """
    extractor = LogisticsFeatureExtractor(default_buffer_hours=20.0).fit(pd.DataFrame())
    df = pd.DataFrame([{
        "port_arrival_delay_hours": 8.0,
        "departure_delay_hours": 4.0,
    }])
    transformed = extractor.transform(df)
    assert transformed["remaining_buffer_hours"].iloc[0] == pytest.approx(12.0, rel=1e-3)
    assert transformed["unabsorbed_delay_hours"].iloc[0] == pytest.approx(0.0, rel=1e-3)


def test_buffer_accounting_delay_equal_to_buffer():
    """
    Case B: Delay (20h) == Buffer (20h)
    remaining_buffer_hours = max(0, 20 - 20) = 0h
    unabsorbed_delay_hours = max(0, 20 - 20) = 0h
    """
    extractor = LogisticsFeatureExtractor(default_buffer_hours=20.0).fit(pd.DataFrame())
    df = pd.DataFrame([{
        "port_arrival_delay_hours": 20.0,
        "departure_delay_hours": 10.0,
    }])
    transformed = extractor.transform(df)
    assert transformed["remaining_buffer_hours"].iloc[0] == pytest.approx(0.0, rel=1e-3)
    assert transformed["unabsorbed_delay_hours"].iloc[0] == pytest.approx(0.0, rel=1e-3)


def test_buffer_accounting_delay_greater_than_buffer():
    """
    Case C: Delay (35h) > Buffer (20h)
    remaining_buffer_hours = max(0, 20 - 35) = 0h
    unabsorbed_delay_hours = max(0, 35 - 20) = 15h
    """
    extractor = LogisticsFeatureExtractor(default_buffer_hours=20.0).fit(pd.DataFrame())
    df = pd.DataFrame([{
        "port_arrival_delay_hours": 35.0,
        "departure_delay_hours": 15.0,
    }])
    transformed = extractor.transform(df)
    assert transformed["remaining_buffer_hours"].iloc[0] == pytest.approx(0.0, rel=1e-3)
    assert transformed["unabsorbed_delay_hours"].iloc[0] == pytest.approx(15.0, rel=1e-3)


def test_buffer_derived_from_timestamps():
    """Verifies that initial_buffer_hours is dynamically derived from scheduled timetable."""
    extractor = LogisticsFeatureExtractor(nominal_vessel_speed_kmh=35.0).fit(pd.DataFrame())
    # Distance = 7000 km -> Nominal transit = 7000 / 35 = 200 hours.
    # Scheduled transit = 224 hours.
    # Initial buffer = 224 - 200 = 24 hours.
    # If delay = 10 hours: remaining buffer = 14h, unabsorbed = 0h.
    t0 = datetime(2026, 3, 1, 0, 0, 0)
    t1 = t0 + timedelta(hours=224)
    df = pd.DataFrame([{
        "distance_km": 7000.0,
        "scheduled_departure": t0.isoformat(),
        "scheduled_port_arrival": t1.isoformat(),
        "port_arrival_delay_hours": 10.0,
    }])
    transformed = extractor.transform(df)
    assert transformed["remaining_buffer_hours"].iloc[0] == pytest.approx(14.0, rel=1e-2)
    assert transformed["unabsorbed_delay_hours"].iloc[0] == pytest.approx(0.0, rel=1e-2)


# ==============================================================================
# 3. Missing Telemetry Simulation & Imputation Tests
# ==============================================================================
def test_simulate_training_missingness_reproducible_ten_percent():
    """Verifies that exactly ~10% of training port arrivals are removed with seed 42."""
    df = pd.DataFrame({
        "actual_port_arrival": [f"2026-03-{i+1:02d}T12:00:00" for i in range(100)],
        "port_arrival_delay_hours": [float(i) for i in range(100)],
    })
    
    masked1 = simulate_training_missingness(df, missing_rate=0.10, random_seed=42)
    masked2 = simulate_training_missingness(df, missing_rate=0.10, random_seed=42)
    
    # 10% of 100 rows is 10 rows
    assert masked1["actual_port_arrival"].isna().sum() == 10
    assert masked1["port_arrival_delay_hours"].isna().sum() == 10
    # Reproducibility check
    pd.testing.assert_frame_equal(masked1, masked2)


def test_is_port_arrival_missing_indicator_and_imputation():
    """Verifies that missingness flag is 1 for missing rows and 0 for present rows, and delay is imputed."""
    train_data = pd.DataFrame([
        {"actual_port_arrival": "2026-03-01T12:00:00", "port_arrival_delay_hours": 4.0, "departure_delay_hours": 1.0},
        {"actual_port_arrival": "2026-03-02T12:00:00", "port_arrival_delay_hours": 6.0, "departure_delay_hours": 2.0},
    ])
    extractor = LogisticsFeatureExtractor().fit(train_data)
    
    test_df = pd.DataFrame([
        {"actual_port_arrival": "2026-03-05T12:00:00", "port_arrival_delay_hours": 5.0, "departure_delay_hours": 1.0},
        {"actual_port_arrival": None, "port_arrival_delay_hours": None, "departure_delay_hours": 3.0},
    ])
    transformed = extractor.transform(test_df)
    
    # Check flags
    assert transformed["is_port_arrival_missing"].iloc[0] == 0.0
    assert transformed["is_port_arrival_missing"].iloc[1] == 1.0
    
    # Check imputation: row 1 was missing, but transformed port_arrival_delay_hours is imputed (not NaN)
    assert not np.isnan(transformed["port_arrival_delay_hours"].iloc[1])


# ==============================================================================
# 4. Route Variability, Distance & Weather Interaction Tests
# ==============================================================================
def test_route_encoded_and_unseen_routes():
    """Verifies target encoding produces numeric values and falls back safely for unseen routes."""
    train_df = pd.DataFrame({
        "route": ["ASIA-EUR-01", "ASIA-EUR-01", "TPAC-01", "TPAC-01"],
    })
    y_train = pd.Series([10.0, 14.0, 25.0, 35.0])
    
    extractor = LogisticsFeatureExtractor().fit(train_df, y=y_train)
    
    test_df = pd.DataFrame({
        "route": ["ASIA-EUR-01", "NEW-UNKNOWN-ROUTE"],
    })
    transformed = extractor.transform(test_df)
    
    # Must be numeric
    assert np.issubdtype(transformed["route_encoded"].dtype, np.number)
    # Known route
    assert transformed["route_encoded"].iloc[0] < 20.0
    # Unseen route falls back to global mean
    assert transformed["route_encoded"].iloc[1] == pytest.approx(y_train.mean(), rel=1e-1)


def test_distance_km_lookup_and_numeric():
    """Verifies distance_km is numeric and uses route lookup when missing."""
    extractor = LogisticsFeatureExtractor().fit(pd.DataFrame())
    df = pd.DataFrame([
        {"route": "ASIA-EUR-01", "distance_km": np.nan},
        {"route": "UNKNOWN_ROUTE", "distance_km": 8500.0},
    ])
    transformed = extractor.transform(df)
    assert transformed["distance_km"].iloc[0] == pytest.approx(19500.0, rel=1e-3)
    assert transformed["distance_km"].iloc[1] == pytest.approx(8500.0, rel=1e-3)


def test_weather_x_distance_interaction():
    """Formula: weather_x_distance = weather_severity * distance_km."""
    extractor = LogisticsFeatureExtractor().fit(pd.DataFrame())
    df = pd.DataFrame([{
        "weather_severity": 3.5,
        "distance_km": 10000.0,
    }])
    transformed = extractor.transform(df)
    # 3.5 * 10000.0 = 35000.0
    assert transformed["weather_x_distance"].iloc[0] == pytest.approx(35000.0, rel=1e-3)


# ==============================================================================
# 5. Full Pipeline Integration & Consistency
# ==============================================================================
def test_full_pipeline_transform_columns():
    """Verifies that create_feature_pipeline transforms all engineered columns consistently."""
    pipeline = create_feature_pipeline()
    sample_df = pd.DataFrame([{
        "shipment_id": "TEST-01",
        "route": "ASIA-EUR-01",
        "origin": "CNSHA",
        "destination": "NLRTM",
        "distance_km": 19500.0,
        "carrier_rating": 0.85,
        "scheduled_departure": "2026-03-01T08:00:00",
        "actual_departure": "2026-03-01T12:00:00",
        "departure_delay_hours": 4.0,
        "scheduled_port_arrival": "2026-03-24T12:00:00",
        "actual_port_arrival": "2026-03-24T18:00:00",
        "port_arrival_delay_hours": 6.0,
        "port_congestion": 0.45,
        "weather_severity": 2.5,
        "port_processing_hours": 28.0,
        "customs_processing_hours": 18.0,
        "scheduled_delivery": "2026-03-28T12:00:00",
    }])
    
    transformed_matrix = pipeline.fit_transform(sample_df)
    assert transformed_matrix.shape[0] == 1
    # Check that output matrix is numeric and contains no NaNs
    assert not np.isnan(transformed_matrix).any()
