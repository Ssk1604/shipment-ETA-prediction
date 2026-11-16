"""
Synthetic Shipment Dataset Generator for Aiqon ETA & Delay Propagation.

Supports both:
1. Standard evaluation dataset (~250 records).
2. Large-scale high-throughput dataset generation (10,000 - 500,000+ records)
   using vectorized NumPy array operations and columnar Apache Parquet storage.
3. Automatic sync with SQLite indexed database.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.database import bulk_insert_shipments


GLOBAL_ROUTES = [
    {"route": "ASIA-EUR-01", "origin": "CNSHA", "destination": "NLRTM", "distance_km": 19500},
    {"route": "ASIA-EUR-02", "origin": "SGSIN", "destination": "DEHAM", "distance_km": 15600},
    {"route": "TPAC-01",     "origin": "CNNBO", "destination": "USLAX", "distance_km": 11200},
    {"route": "TPAC-02",     "origin": "KRPUS", "destination": "USLGB", "distance_km": 10900},
    {"route": "ASIA-EUR-03", "origin": "JPYOK", "destination": "GBFXT", "distance_km": 20400},
    {"route": "ME-EUR-01",   "origin": "AEJEA", "destination": "BEANR", "distance_km": 12800},
    {"route": "IND-EUR-01",  "origin": "INNSA", "destination": "NLRTM", "distance_km": 11800},
    {"route": "TPAC-03",     "origin": "VNSGN", "destination": "USOAK", "distance_km": 12500},
    {"route": "TRANSATL-01", "origin": "DEHAM", "destination": "USNYC", "distance_km": 7200},
    {"route": "ASIA-MED-01", "origin": "CNSHA", "destination": "ITGOA", "distance_km": 16400},
]


def generate_shipment_dataset(
    n_samples: int = 250,
    random_seed: int = 42,
    missing_telemetry_rate: float = 0.08,
    start_date: datetime = datetime(2026, 1, 10, 8, 0, 0),
) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset of international maritime shipments.
    Fully vectorized across arrays for high performance scalability.

    Args:
        n_samples: Total number of shipment records.
        random_seed: Seed for NumPy RNG reproducibility.
        missing_telemetry_rate: Fraction of intermediate port arrival timestamps to omit.
        start_date: Beginning baseline timestamp for scheduled departures.

    Returns:
        pd.DataFrame containing complete operational and target columns.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Route Assignment Vectorized
    route_indices = rng.choice(len(GLOBAL_ROUTES), size=n_samples)
    routes = [GLOBAL_ROUTES[idx]["route"] for idx in route_indices]
    origins = [GLOBAL_ROUTES[idx]["origin"] for idx in route_indices]
    destinations = [GLOBAL_ROUTES[idx]["destination"] for idx in route_indices]
    distances = np.array([GLOBAL_ROUTES[idx]["distance_km"] for idx in route_indices], dtype=np.float64)

    # 2. Timeline Offsets
    dep_day_offsets = rng.uniform(0, 90, size=n_samples)
    dep_hour_offsets = rng.choice([8.0, 14.0, 20.0], size=n_samples)
    total_sec_offsets = (dep_day_offsets * 86400) + (dep_hour_offsets * 3600)
    scheduled_departures = [start_date + timedelta(seconds=float(s)) for s in total_sec_offsets]

    # 3. Origin Departure Disruption
    carrier_ratings = np.round(rng.beta(7, 3, size=n_samples), 2)
    is_origin_delayed = rng.random(size=n_samples) > (carrier_ratings * 0.9)
    exp_delays = np.round(rng.exponential(scale=18.0, size=n_samples), 2)
    norm_delays = np.maximum(-3.0, np.round(rng.normal(loc=1.0, scale=2.5, size=n_samples), 2))
    departure_delay_hours = np.where(is_origin_delayed, exp_delays, norm_delays)

    actual_departures = [
        sched + timedelta(hours=float(d))
        for sched, d in zip(scheduled_departures, departure_delay_hours)
    ]

    # 4. Ocean Transit
    nominal_speeds = rng.normal(loc=35.0, scale=1.2, size=n_samples)
    nominal_transit_hours = distances / nominal_speeds
    transit_buffer_hours = rng.uniform(16.0, 28.0, size=n_samples)
    scheduled_transit_hours = nominal_transit_hours + transit_buffer_hours

    scheduled_port_arrivals = [
        sched + timedelta(hours=float(h))
        for sched, h in zip(scheduled_departures, scheduled_transit_hours)
    ]

    # Weather & Ocean Disruption
    weather_severities = np.round(np.clip(rng.gamma(shape=2.5, scale=0.8, size=n_samples), 1.0, 5.0), 2)
    weather_scales = np.maximum(1.0, (weather_severities - 1.5) * 9.0)
    weather_transit_disruptions = np.round(rng.exponential(scale=weather_scales), 2)

    actual_transit_hours = nominal_transit_hours + weather_transit_disruptions
    actual_port_arrivals = [
        act_dep + timedelta(hours=float(h))
        for act_dep, h in zip(actual_departures, actual_transit_hours)
    ]

    port_arrival_delay_hours = np.array([
        round((act - sched).total_seconds() / 3600.0, 2)
        for act, sched in zip(actual_port_arrivals, scheduled_port_arrivals)
    ])

    # 5. Port Congestion & Handling
    port_congestions = np.round(rng.beta(2.5, 3.5, size=n_samples), 2)
    nominal_port_processing = rng.uniform(20.0, 30.0, size=n_samples)
    congestion_wait_hours = np.round(port_congestions * rng.uniform(24.0, 72.0, size=n_samples), 2)
    port_processing_hours = np.round(nominal_port_processing + congestion_wait_hours, 2)

    # 6. Customs Processing
    has_customs_holds = rng.random(size=n_samples) < 0.22
    customs_hold_hours = np.where(has_customs_holds, rng.uniform(20.0, 56.0, size=n_samples), rng.uniform(0.0, 4.0, size=n_samples))
    nominal_customs = rng.uniform(12.0, 18.0, size=n_samples)
    customs_processing_hours = np.round(nominal_customs + customs_hold_hours, 2)

    # 7. Hinterland Drayage & Final Delivery
    nominal_hinterlands = rng.uniform(20.0, 32.0, size=n_samples)
    scheduled_port_handling = 24.0
    scheduled_customs_clearance = 16.0
    scheduled_hinterland = 28.0

    scheduled_deliveries = [
        sched_port + timedelta(hours=(scheduled_port_handling + scheduled_customs_clearance + scheduled_hinterland))
        for sched_port in scheduled_port_arrivals
    ]

    actual_deliveries = [
        act_port + timedelta(hours=float(p + c + h))
        for act_port, p, c, h in zip(actual_port_arrivals, port_processing_hours, customs_processing_hours, nominal_hinterlands)
    ]

    final_delay_hours = np.array([
        round((act - sched).total_seconds() / 3600.0, 2)
        for act, sched in zip(actual_deliveries, scheduled_deliveries)
    ])
    is_delayed_24h = (final_delay_hours >= 24.0).astype(int)

    # Telemetry Dropout Simulation
    has_missing_telemetry = rng.random(size=n_samples) < missing_telemetry_rate

    shipment_ids = [f"SHP-2026-{i:04d}" for i in range(1, n_samples + 1)]

    reported_port_arrivals = [
        None if missing else act.isoformat()
        for missing, act in zip(has_missing_telemetry, actual_port_arrivals)
    ]
    reported_port_delays = [
        None if missing else float(delay)
        for missing, delay in zip(has_missing_telemetry, port_arrival_delay_hours)
    ]

    df = pd.DataFrame({
        "shipment_id": shipment_ids,
        "origin": origins,
        "destination": destinations,
        "route": routes,
        "distance_km": distances,
        "carrier_rating": carrier_ratings,
        "scheduled_departure": [s.isoformat() for s in scheduled_departures],
        "actual_departure": [a.isoformat() for a in actual_departures],
        "departure_delay_hours": departure_delay_hours,
        "scheduled_port_arrival": [s.isoformat() for s in scheduled_port_arrivals],
        "actual_port_arrival": reported_port_arrivals,
        "port_arrival_delay_hours": reported_port_delays,
        "port_congestion": port_congestions,
        "weather_severity": weather_severities,
        "port_processing_hours": port_processing_hours,
        "customs_processing_hours": customs_processing_hours,
        "scheduled_delivery": [s.isoformat() for s in scheduled_deliveries],
        "actual_delivery": [a.isoformat() for a in actual_deliveries],
        "final_delay_hours": final_delay_hours,
        "is_delayed_24h": is_delayed_24h,
    })

    df = df.sort_values(by="scheduled_departure").reset_index(drop=True)
    return df


def save_dataset(
    df: pd.DataFrame,
    output_csv: str = "dataset/shipments.csv",
    output_parquet: str = "dataset/shipments.parquet",
    sync_db: bool = True,
) -> Dict[str, str]:
    """
    Saves dataset to CSV, high-performance compressed Parquet, and syncs with SQLite database.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 1. Save CSV
    df.to_csv(output_csv, index=False)
    
    # 2. Save Columnar Parquet (Snappy compression for 10x faster reads)
    try:
        df.to_parquet(output_parquet, index=False, compression="snappy")
    except Exception as e:
        print(f"Parquet save warning: {e}")

    # 3. Sync to Indexed SQLite Database
    if sync_db:
        bulk_insert_shipments(df)

    return {
        "csv": output_csv,
        "parquet": output_parquet,
    }


if __name__ == "__main__":
    import time
    start_t = time.time()
    df = generate_shipment_dataset(n_samples=250, random_seed=42)
    saved = save_dataset(df)
    elapsed = time.time() - start_t
    print(f"Generated {len(df)} shipment records in {elapsed:.3f} seconds.")
    print(f"Saved to:\n  - CSV: {saved['csv']}\n  - Parquet: {saved['parquet']}")
    print(f"Mean Final Delay: {df['final_delay_hours'].mean():.2f}h | Delayed >=24h: {df['is_delayed_24h'].sum()} ({df['is_delayed_24h'].mean():.1%})")
