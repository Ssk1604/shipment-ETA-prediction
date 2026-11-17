"""
Unit tests for Mathematical Delay Propagation and Slack Absorption.
"""

from datetime import datetime
import pytest

from src.delay_propagation import (
    calculate_delay_hours,
    compute_stage_propagation,
    simulate_delay_impact,
    format_hours_to_hhmm,
    get_operational_status,
    format_buffer_duration,
    calculate_schedule_buffer_analysis,
)


def test_calculate_delay_hours():
    t_sched = datetime(2026, 2, 1, 10, 0, 0)
    t_act_late = datetime(2026, 2, 1, 14, 30, 0)
    t_act_early = datetime(2026, 2, 1, 8, 0, 0)

    assert calculate_delay_hours(t_act_late, t_sched) == 4.5
    assert calculate_delay_hours(t_act_early, t_sched) == -2.0


def test_compute_stage_propagation_slack_absorption():
    # Inherited 10 hours delay, nominal time 40h, scheduled timetable allocated 50h
    # Scheduled slack = 50 - 40 = 10h.
    # Buffer should absorb all 10h of inherited delay!
    result = compute_stage_propagation(
        inherited_delay_hours=10.0,
        nominal_duration_hours=40.0,
        scheduled_duration_hours=50.0,
        stage_disruption_hours=0.0,
    )
    assert result["available_slack_hours"] == 10.0
    assert result["slack_absorbed_hours"] == 10.0
    assert result["net_delay_hours"] == 0.0


def test_compute_stage_propagation_disruption_exceeding_slack():
    # Inherited 12 hours delay, nominal 40h, scheduled 48h (slack = 8h)
    # Stage disruption added = 5h
    # Absorbed = min(12, 8) = 8h.
    # Net delay = 12 - 8 + 5 = 9h.
    result = compute_stage_propagation(
        inherited_delay_hours=12.0,
        nominal_duration_hours=40.0,
        scheduled_duration_hours=48.0,
        stage_disruption_hours=5.0,
    )
    assert result["slack_absorbed_hours"] == 8.0
    assert result["net_delay_hours"] == 9.0


def test_simulate_delay_impact():
    res = simulate_delay_impact(
        current_final_delay_hours=15.0,
        stage="port",
        additional_disruption_hours=20.0,
        remaining_slack_hours=8.0,
    )
    # 8h absorbed, 12h net added
    assert res["absorbed_by_slack"] == 8.0
    assert res["delta_impact_hours"] == 12.0
    assert res["new_final_delay_hours"] == 27.0


def test_format_hours_to_hhmm():
    assert format_hours_to_hhmm(10.94) == "+10h 56m"
    assert format_hours_to_hhmm(-5.61) == "-5h 37m (Early)"
    assert format_hours_to_hhmm(-5.61, show_early=False) == "-5h 37m"
    assert format_hours_to_hhmm(0.57) == "+0h 34m"
    assert format_hours_to_hhmm(1.76) == "+1h 46m"
    assert format_hours_to_hhmm(0.0) == "On Schedule (0m)"
    assert format_hours_to_hhmm(None) == "N/A"


def test_get_operational_status():
    assert get_operational_status(-5.61) == "🟢 Early"
    assert get_operational_status(0.0) == "🟢 On-Time"
    assert get_operational_status(8.5) == "🟢 On-Time"
    assert get_operational_status(18.2) == "🟡 Moderate Delay"
    assert get_operational_status(28.0) == "🔴 Critical Delay (≥24h)"
    assert get_operational_status(None) == "⚪ Unknown"


def test_format_buffer_duration():
    assert format_buffer_duration(18.0) == "18h"
    assert format_buffer_duration(13.8333) == "13h 50m"
    assert format_buffer_duration(4.1667) == "4h 10m"
    assert format_buffer_duration(0.0) == "0h 00m"
    assert format_buffer_duration(None) == "N/A"


def test_calculate_schedule_buffer_analysis_illustrative():
    # Test case matching the exact illustrative example:
    # 18h Initial buffer, 13h 50m consumed buffer, 4h 10m remaining buffer, 76.9% utilization
    record = {
        "departure_delay_hours": 0.57,
        "weather_severity": 2.5,  # est_ocean = (2.5 - 2.0) * 8.0 = 4.0
        "port_congestion": 0.5,
        "port_processing_hours": 30.0,  # est_port = 0.5 * 10.0 = 5.0
        "customs_processing_hours": 22.0,  # est_cust = 22.0 - 16.0 = 6.0
        # gross = 0.57 + 4.0 + 5.0 + 6.0 = 15.57h
        "final_delay_hours": 1.74,  # final delay = 1.74h -> absorbed = 15.57 - 1.74 = 13.83h (13h 50m)
    }
    result = calculate_schedule_buffer_analysis(record, initial_buffer_override=18.0)

    assert result["initial_slack_hours"] == 18.0
    assert result["consumed_slack_hours"] == 13.83
    assert result["remaining_slack_hours"] == 4.17
    assert result["buffer_utilization_pct"] == 76.9
    assert result["initial_slack_display"] == "18h"
    assert result["consumed_slack_display"] == "13h 50m"
    assert result["remaining_slack_display"] == "4h 10m"
    assert result["utilization_display"] == "76.9%"
    assert result["status_label"] == "Caution"


def test_calculate_schedule_buffer_analysis_exhausted():
    # Massive disruption where gross disruption exceeds initial buffer
    record = {
        "departure_delay_hours": 12.0,
        "weather_severity": 4.0,  # (4-2)*8 = 16.0
        "port_congestion": 0.8,
        "port_processing_hours": 40.0,  # 0.8 * 20 = 16.0
        "customs_processing_hours": 20.0,  # 4.0
        # gross = 12 + 16 + 16 + 4 = 48h
        "final_delay_hours": 30.0,  # absorbed = 18h
    }
    result = calculate_schedule_buffer_analysis(record, initial_buffer_override=18.0)

    assert result["initial_slack_hours"] == 18.0
    assert result["consumed_slack_hours"] == 18.0
    assert result["remaining_slack_hours"] == 0.0
    assert result["buffer_utilization_pct"] == 100.0
    assert result["status_label"] == "Exhausted"
    assert "🔴 Exhausted" in result["status_badge"]



