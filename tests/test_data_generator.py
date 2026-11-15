"""
Unit tests for Synthetic Shipment Dataset Generator.
"""

import os
import pandas as pd
import pytest

from src.data_generator import generate_shipment_dataset, save_dataset


def test_generate_dataset_shape_and_columns():
    df = generate_shipment_dataset(n_samples=50, random_seed=42)
    assert len(df) == 50

    expected_cols = [
        "shipment_id", "origin", "destination", "route", "distance_km",
        "carrier_rating", "scheduled_departure", "actual_departure",
        "departure_delay_hours", "scheduled_port_arrival", "actual_port_arrival",
        "port_arrival_delay_hours", "port_congestion", "weather_severity",
        "port_processing_hours", "customs_processing_hours", "scheduled_delivery",
        "actual_delivery", "final_delay_hours", "is_delayed_24h",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in dataset"


def test_reproducibility():
    df1 = generate_shipment_dataset(n_samples=20, random_seed=99)
    df2 = generate_shipment_dataset(n_samples=20, random_seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_missing_telemetry_rate():
    rate = 0.10
    df = generate_shipment_dataset(n_samples=100, random_seed=42, missing_telemetry_rate=rate)
    missing_count = df["actual_port_arrival"].isna().sum()
    # Should be close to 10%
    assert 4 <= missing_count <= 18


def test_positive_distances_and_realistic_ranges():
    df = generate_shipment_dataset(n_samples=50, random_seed=42)
    assert (df["distance_km"] > 0).all()
    assert (df["weather_severity"] >= 1.0).all()
    assert (df["weather_severity"] <= 5.0).all()
    assert (df["port_congestion"] >= 0.0).all()
    assert (df["port_congestion"] <= 1.0).all()
