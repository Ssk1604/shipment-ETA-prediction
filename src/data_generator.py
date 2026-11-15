"""
Synthetic Shipment Dataset Generator for Aiqon ETA & Delay Propagation.

Generates realistic international maritime shipment records adhering to:
- Physical transit constraints (distance, nominal vessel speed ~35 km/h).
- Realistic weather impact (severity 1.0 - 5.0).
- Port congestion queues (anchorage dwell time).
- Customs clearance variances and random inspections.
- Commercial schedule slack (buffers that absorb modest delays).
- Missing telemetry timestamps (~8% missing port arrivals) to test imputation.
- Calibrated delay distribution (approx 25-35% delayed >= 24h).
- Reproducibility via fixed random seed (default: 42).
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


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

    Args:
        n_samples: Total number of shipment records (~250).
        random_seed: Seed for NumPy RNG reproducibility.
        missing_telemetry_rate: Fraction of intermediate port arrival timestamps to omit.
        start_date: Beginning baseline timestamp for scheduled departures.

    Returns:
        pd.DataFrame containing complete operational and target columns.
    """
    rng = np.random.default_rng(random_seed)
    records: List[Dict] = []

    for i in range(1, n_samples + 1):
        shipment_id = f"SHP-2026-{i:04d}"
        route_meta = GLOBAL_ROUTES[rng.choice(len(GLOBAL_ROUTES))]
        route = route_meta["route"]
        origin = route_meta["origin"]
        destination = route_meta["destination"]
        distance_km = float(route_meta["distance_km"])

        # Scheduled Departure staggered across a 90-day window
        dep_day_offset = float(rng.uniform(0, 90))
        dep_hour_offset = float(rng.choice([8.0, 14.0, 20.0]))
        scheduled_departure = start_date + timedelta(days=dep_day_offset, hours=dep_hour_offset)

        # 1. Origin Terminal & Departure Disruption
        # Carrier reliability rating (0.50 to 0.98)
        carrier_rating = round(float(rng.beta(7, 3)), 2)
        is_origin_delayed = rng.random() > (carrier_rating * 0.9)
        if is_origin_delayed:
            departure_delay_hours = round(float(rng.exponential(scale=18.0)), 2)
        else:
            departure_delay_hours = round(float(rng.normal(loc=1.0, scale=2.5)), 2)
            departure_delay_hours = max(-3.0, departure_delay_hours)  # Cap early departure at 3h

        actual_departure = scheduled_departure + timedelta(hours=departure_delay_hours)

        # 2. Ocean Transit
        # Vessel nominal cruising speed: ~35 km/h
        nominal_speed = float(rng.normal(loc=35.0, scale=1.2))
        nominal_transit_hours = distance_km / nominal_speed
        
        # Planned commercial buffer for transit: realistic 16 to 28 hours
        transit_buffer_hours = float(rng.uniform(16.0, 28.0))
        scheduled_transit_hours = nominal_transit_hours + transit_buffer_hours
        scheduled_port_arrival = scheduled_departure + timedelta(hours=scheduled_transit_hours)

        # Weather Severity: 1.0 (calm) to 5.0 (storm/typhoon)
        weather_severity = round(float(np.clip(rng.gamma(shape=2.5, scale=0.8), 1.0, 5.0)), 2)
        
        # Weather disruption adds sailing delay
        weather_transit_disruption = round(
            float(rng.exponential(scale=max(1.0, (weather_severity - 1.5) * 9.0))), 2
        )

        actual_transit_hours = nominal_transit_hours + weather_transit_disruption
        actual_port_arrival = actual_departure + timedelta(hours=actual_transit_hours)

        port_arrival_delay_hours = round(
            (actual_port_arrival - scheduled_port_arrival).total_seconds() / 3600.0, 2
        )

        # 3. Port Congestion & Offloading
        # Port congestion index (0.0 to 1.0)
        port_congestion = round(float(rng.beta(2.5, 3.5)), 2)
        nominal_port_processing = float(rng.uniform(20.0, 30.0))
        
        # Congestion delay: waiting at anchorage outside the port before berth opens
        congestion_wait_hours = round(port_congestion * float(rng.uniform(24.0, 72.0)), 2)
        port_processing_hours = round(nominal_port_processing + congestion_wait_hours, 2)

        # 4. Customs Processing
        # Standard customs takes 12-20 hours; regulatory inspection/audit can add 20-60 hours
        has_customs_hold = rng.random() < 0.22
        customs_hold_hours = float(rng.uniform(20.0, 56.0)) if has_customs_hold else float(rng.uniform(0.0, 4.0))
        nominal_customs_hours = float(rng.uniform(12.0, 18.0))
        customs_processing_hours = round(nominal_customs_hours + customs_hold_hours, 2)

        # 5. Hinterland Drayage & Final Delivery
        nominal_hinterland_hours = float(rng.uniform(20.0, 32.0))
        
        # Scheduled port, customs, and inland allocations in contract SLA
        scheduled_port_handling = 24.0
        scheduled_customs_clearance = 16.0
        scheduled_hinterland = 28.0
        
        scheduled_delivery = scheduled_port_arrival + timedelta(
            hours=(scheduled_port_handling + scheduled_customs_clearance + scheduled_hinterland)
        )

        # Actual Delivery
        actual_delivery = actual_port_arrival + timedelta(
            hours=(port_processing_hours + customs_processing_hours + nominal_hinterland_hours)
        )

        final_delay_hours = round(
            (actual_delivery - scheduled_delivery).total_seconds() / 3600.0, 2
        )
        is_delayed_24h = int(final_delay_hours >= 24.0)

        # Telemetry missingness: mask actual_port_arrival in ~8% of records
        has_missing_telemetry = rng.random() < missing_telemetry_rate
        reported_port_arrival: Optional[str] = (
            None if has_missing_telemetry else actual_port_arrival.isoformat()
        )
        reported_port_delay: Optional[float] = (
            None if has_missing_telemetry else port_arrival_delay_hours
        )

        records.append({
            "shipment_id": shipment_id,
            "origin": origin,
            "destination": destination,
            "route": route,
            "distance_km": distance_km,
            "carrier_rating": carrier_rating,
            "scheduled_departure": scheduled_departure.isoformat(),
            "actual_departure": actual_departure.isoformat(),
            "departure_delay_hours": departure_delay_hours,
            "scheduled_port_arrival": scheduled_port_arrival.isoformat(),
            "actual_port_arrival": reported_port_arrival,
            "port_arrival_delay_hours": reported_port_delay,
            "port_congestion": port_congestion,
            "weather_severity": weather_severity,
            "port_processing_hours": port_processing_hours,
            "customs_processing_hours": customs_processing_hours,
            "scheduled_delivery": scheduled_delivery.isoformat(),
            "actual_delivery": actual_delivery.isoformat(),
            "final_delay_hours": final_delay_hours,
            "is_delayed_24h": is_delayed_24h,
        })

    df = pd.DataFrame(records)
    # Sort chronologically by scheduled departure for realistic time-series / temporal validation
    df = df.sort_values(by="scheduled_departure").reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, output_path: str = "dataset/shipments.csv") -> str:
    """Saves generated shipment records to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    df = generate_shipment_dataset(n_samples=250, random_seed=42)
    saved_file = save_dataset(df)
    print(f"Generated {len(df)} shipment records successfully at: {saved_file}")
    print("\nDataset Summary:")
    print(f"Total Records: {len(df)}")
    print(f"Missing actual_port_arrival: {df['actual_port_arrival'].isna().sum()} ({df['actual_port_arrival'].isna().mean():.1%})")
    print(f"Mean Final Delay (hours): {df['final_delay_hours'].mean():.2f}")
    print(f"Median Final Delay (hours): {df['final_delay_hours'].median():.2f}")
    print(f"Delayed >= 24h: {df['is_delayed_24h'].sum()} ({df['is_delayed_24h'].mean():.1%})")
    print(f"Min Delay: {df['final_delay_hours'].min():.2f}h, Max Delay: {df['final_delay_hours'].max():.2f}h")
