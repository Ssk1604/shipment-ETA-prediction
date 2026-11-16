"""
Interactive Feature Engineering Inspector.
Demonstrates the real-time mathematical calculations of all 4 advanced features.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.feature_engineering import LogisticsFeatureExtractor


def run_feature_inspection():
    print("=" * 80)
    print("🚢 AIQON FEATURE ENGINEERING: LIVE MATHEMATICAL INSPECTION")
    print("=" * 80)

    # Two contrasting test shipments
    test_shipments = pd.DataFrame([
        {
            "shipment_id": "SHP-ALPHA-01 (On-Time)",
            "route": "ASIA-EUR-01",
            "distance_km": 19500.0,
            "carrier_rating": 0.92,
            "scheduled_departure": "2026-03-01T08:00:00",
            "actual_departure": "2026-03-01T09:30:00",
            "departure_delay_hours": 1.5,
            "scheduled_port_arrival": "2026-03-24T12:00:00",
            "actual_port_arrival": "2026-03-24T14:00:00",
            "port_arrival_delay_hours": 2.0,   # <= 4h (No Missed Slot Penalty)
            "port_congestion": 0.50,
            "weather_severity": 2.2,
            "port_processing_hours": 26.0,
            "customs_processing_hours": 14.0,
        },
        {
            "shipment_id": "SHP-BETA-02 (Severe Delay)",
            "route": "TPAC-01",
            "distance_km": 11200.0,
            "carrier_rating": 0.65,
            "scheduled_departure": "2026-03-01T08:00:00",
            "actual_departure": "2026-03-01T16:00:00",
            "departure_delay_hours": 8.0,
            "scheduled_port_arrival": "2026-03-14T12:00:00",
            "actual_port_arrival": None,       # MISSING TELEMETRY!
            "port_arrival_delay_hours": None,  # MISSING!
            "port_congestion": 0.70,           # High congestion
            "weather_severity": 4.5,           # Severe storm
            "port_processing_hours": 48.0,
            "customs_processing_hours": 36.0,
        },
    ])

    extractor = LogisticsFeatureExtractor(nominal_vessel_speed_kmh=35.0, default_buffer_hours=20.0)
    extractor.fit(test_shipments)
    transformed = extractor.transform(test_shipments)

    for i in range(len(transformed)):
        row = transformed.iloc[i]
        orig = test_shipments.iloc[i]
        print(f"\n📦 SCENARIO {i+1}: {row['shipment_id']}")
        print("-" * 70)
        print(f"  • Route: {orig['route']} | Distance: {row['distance_km']:,.0f} km")
        print(f"  • Weather Severity: {row['weather_severity']} / 5.0 | Port Congestion: {row['port_congestion']}")
        print(f"  • Departure Delay: {row['departure_delay_hours']} hrs | Port Delay: {orig['port_arrival_delay_hours']} hrs")
        
        print("\n  🔍 ADVANCED ENGINEERED FEATURES:")
        print(f"  1. [Feature 1] congestion_x_missed_slot: {row['congestion_x_missed_slot']:.3f}")
        penalty = 2.0 if row['port_arrival_delay_hours'] > 4.0 else 1.0
        print(f"     └─ Math: congestion ({row['port_congestion']}) × penalty ({penalty:.1f})")

        print(f"  2. [Feature 2] Schedule Slack Accounting:")
        print(f"     ├─ remaining_buffer_hours : {row['remaining_buffer_hours']:.1f} hrs (Delay absorbed by timetable)")
        print(f"     └─ unabsorbed_delay_hours : {row['unabsorbed_delay_hours']:.1f} hrs (Cascades to downstream delivery)")

        print(f"  3. [Feature 3] Missing Data Telemetry:")
        print(f"     ├─ is_port_arrival_missing: {int(row['is_port_arrival_missing'])} ({'FLAGGED AS IMPUTED' if row['is_port_arrival_missing'] else 'CONFIRMED SENSOR'})")
        print(f"     └─ effective port delay   : {row['port_arrival_delay_hours']:.1f} hrs (Imputed via training prior)")

        print(f"  4. [Feature 4] Route & Weather Interactions:")
        print(f"     ├─ route_encoded          : {row['route_encoded']:.2f} hrs (Historical smoothed route prior)")
        print(f"     └─ weather_x_distance     : {row['weather_x_distance']:,.1f} (Severity × Distance interaction)")

    print("\n" + "=" * 80)
    print("✅ OBSERVATION SUMMARY:")
    print("  • Notice in Scenario 1: Port delay was 2h (<= 4h), so penalty was 1.0 -> congestion factor stayed at 0.50.")
    print("  • Notice in Scenario 2: Telemetry was None -> flag is_port_arrival_missing became 1.0, imputed to 8.0h.")
    print("  • Since 8.0h > 4h window, penalty doubled to 2.0 -> congestion factor escalated to 1.40!")
    print("  • The schedule buffer absorbed all 2h in Scenario 1 (unabsorbed=0), but exhausted in Scenario 2 (unabsorbed=8h).")
    print("=" * 80)


if __name__ == "__main__":
    run_feature_inspection()
