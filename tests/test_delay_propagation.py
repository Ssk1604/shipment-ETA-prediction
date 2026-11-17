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

