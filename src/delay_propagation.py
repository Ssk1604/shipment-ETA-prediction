"""
Delay Propagation Engine for Multi-Stage Logistics.

This module models realistic international logistics delay propagation:
1. Stage Delays: (Actual Stage Timestamp - Scheduled Stage Timestamp) in hours.
2. Scheduled Slack: Buffer planned into the schedule to absorb unexpected delays.
3. Slack Absorption: Portions of inherited upstream delays absorbed when operations
   proceed faster than the scheduled buffer.
4. Downstream Compounding: Disruption added at any stage (e.g. port congestion,
   customs inspection, storm rerouting) that propagates forward to subsequent stages.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class StagePropagationResult:
    """Represents the delay accounting for a single shipment transition."""
    stage_name: str
    scheduled_arrival: datetime
    actual_arrival: datetime
    inherited_delay_hours: float
    stage_disruption_hours: float
    slack_absorbed_hours: float
    net_delay_hours: float


@dataclass
class ShipmentDelayBreakdown:
    """Comprehensive delay waterfall breakdown for an entire shipment."""
    shipment_id: str
    departure_delay_hours: float
    port_arrival_delay_hours: float
    port_congestion_delay_hours: float
    customs_delay_hours: float
    hinterland_delay_hours: float
    total_slack_absorbed_hours: float
    final_delay_hours: float
    predicted_eta: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if self.predicted_eta:
            res["predicted_eta"] = self.predicted_eta.isoformat()
        return res


def calculate_delay_hours(actual: datetime, scheduled: datetime) -> float:
    """
    Computes delay in decimal hours between an actual and scheduled timestamp.
    Positive values indicate late arrival; negative values indicate early arrival.
    """
    delta = actual - scheduled
    return round(delta.total_seconds() / 3600.0, 2)


def compute_stage_propagation(
    inherited_delay_hours: float,
    nominal_duration_hours: float,
    scheduled_duration_hours: float,
    stage_disruption_hours: float,
) -> Dict[str, float]:
    """
    Models the delay propagation across a specific leg or stage.

    Formula:
      - scheduled_slack = max(0.0, scheduled_duration - nominal_duration)
      - If inherited_delay > 0, the buffer absorbs:
          absorbed_delay = min(inherited_delay, scheduled_slack)
      - The net delay reaching the end of the stage is:
          net_delay = inherited_delay - absorbed_delay + stage_disruption

    Args:
        inherited_delay_hours: Delay carried over from the prior stage.
        nominal_duration_hours: Expected time under standard non-disrupted conditions.
        scheduled_duration_hours: Time allocated in the commercial timetable.
        stage_disruption_hours: Delay added strictly within this stage.

    Returns:
        Dictionary with:
          - net_delay_hours
          - slack_absorbed_hours
          - stage_disruption_hours
          - available_slack_hours
    """
    available_slack = max(0.0, scheduled_duration_hours - nominal_duration_hours)
    
    # Slack can only absorb positive (late) delays
    if inherited_delay_hours > 0:
        absorbed_delay = min(inherited_delay_hours, available_slack)
    else:
        absorbed_delay = 0.0

    net_delay = inherited_delay_hours - absorbed_delay + stage_disruption_hours

    return {
        "net_delay_hours": round(net_delay, 2),
        "slack_absorbed_hours": round(absorbed_delay, 2),
        "stage_disruption_hours": round(stage_disruption_hours, 2),
        "available_slack_hours": round(available_slack, 2),
    }


def simulate_delay_impact(
    current_final_delay_hours: float,
    stage: str,
    additional_disruption_hours: float,
    remaining_slack_hours: float = 0.0,
) -> Dict[str, float]:
    """
    Simulates how an incremental disruption at an intermediate stage (e.g. extra port congestion)
    impacts the downstream final delivery delay, taking into account any downstream schedule slack.
    
    Args:
        current_final_delay_hours: Currently projected final delay.
        stage: The stage experiencing disruption ('port', 'customs', 'transit').
        additional_disruption_hours: Additional hours added.
        remaining_slack_hours: Total uncommitted schedule buffer downstream of this stage.

    Returns:
        Dictionary with new_final_delay_hours, absorbed_by_slack, and delta_impact_hours.
    """
    absorbed = min(additional_disruption_hours, max(0.0, remaining_slack_hours))
    effective_increase = max(0.0, additional_disruption_hours - absorbed)
    new_delay = current_final_delay_hours + effective_increase

    return {
        "stage": stage,
        "additional_disruption_hours": round(additional_disruption_hours, 2),
        "absorbed_by_slack": round(absorbed, 2),
        "delta_impact_hours": round(effective_increase, 2),
        "new_final_delay_hours": round(new_delay, 2),
    }


def format_hours_to_hhmm(hours: Optional[float], show_early: bool = True) -> str:
    """
    Converts continuous decimal hours into human-readable hours and minutes notation.

    Examples:
        10.94 -> "+10h 56m"
        -5.61 -> "-5h 37m (Early)" (or "-5h 37m" if show_early=False)
        0.0   -> "On Schedule (0m)"
        0.57  -> "+0h 34m"
        None  -> "N/A"
    """
    if hours is None:
        return "N/A"
    try:
        val = float(hours)
    except (ValueError, TypeError):
        return "N/A"

    if abs(val) < 1e-4:
        return "On Schedule (0m)" if show_early else "0h 00m"

    is_negative = val < 0
    total_minutes = int(round(abs(val) * 60))
    h = total_minutes // 60
    m = total_minutes % 60

    if is_negative:
        suffix = " (Early)" if show_early else ""
        return f"-{h}h {m:02d}m{suffix}"
    else:
        return f"+{h}h {m:02d}m"


def get_operational_status(delay_hours: Optional[float], with_badge: bool = True) -> str:
    """
    Provides consistent operational status classification for logistics operators:
      - Early: delay < -0.1 hours
      - On-Time: -0.1 <= delay < 12.0 hours
      - Moderate Delay: 12.0 <= delay < 24.0 hours
      - Critical Delay (≥24h SLA breach): delay >= 24.0 hours
    """
    if delay_hours is None:
        return "⚪ Unknown" if with_badge else "Unknown"
    try:
        val = float(delay_hours)
    except (ValueError, TypeError):
        return "⚪ Unknown" if with_badge else "Unknown"

    if val < -0.1:
        return "🟢 Early" if with_badge else "Early"
    elif val < 12.0:
        return "🟢 On-Time" if with_badge else "On-Time"
    elif val < 24.0:
        return "🟡 Moderate Delay" if with_badge else "Moderate Delay"
    else:
        return "🔴 Critical Delay (≥24h)" if with_badge else "Critical Delay (≥24h)"


def format_buffer_duration(hours: Optional[float]) -> str:
    """
    Formats a buffer duration into clean human-readable hours and minutes notation without signs.

    Examples:
        18.0   -> "18h"
        13.833 -> "13h 50m"
        4.167  -> "4h 10m"
        0.0    -> "0h 00m"
        None   -> "N/A"
    """
    if hours is None:
        return "N/A"
    try:
        val = float(hours)
    except (ValueError, TypeError):
        return "N/A"

    val = max(0.0, val)
    total_minutes = int(round(val * 60))
    if total_minutes == 0:
        return "0h 00m"
    h = total_minutes // 60
    m = total_minutes % 60
    if m == 0:
        return f"{h}h"
    return f"{h}h {m:02d}m"


def calculate_schedule_buffer_analysis(
    record: Dict[str, Any],
    initial_buffer_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates the complete schedule buffer dynamics for a shipment:
      1. Initial Schedule Slack: Planned buffer built into timetables (ocean transit + terminal + customs).
      2. Slack Consumed: Buffer already used to absorb upstream delays & disruptions.
      3. Remaining Buffer: Buffer still available for downstream operations.
      4. Buffer Utilization: Percentage of available buffer consumed (e.g. 76.9%).
      5. Status Tier: 'Healthy', 'Caution', or 'Exhausted'.

    Args:
        record: Dictionary or Series with shipment milestones and operational metrics.
        initial_buffer_override: Optional explicit initial buffer in hours.

    Returns:
        Dictionary containing both numeric values and executive display strings.
    """
    # 1. Determine Initial Available Schedule Slack
    if initial_buffer_override is not None and initial_buffer_override > 0:
        initial_slack = round(float(initial_buffer_override), 2)
    else:
        # Calculate planned timetable buffer from milestone deltas
        sched_dep = record.get("scheduled_departure")
        sched_port = record.get("scheduled_port_arrival")
        sched_del = record.get("scheduled_delivery")
        dist_km = float(record.get("distance_km", 0.0) or 0.0)

        ocean_slack = 0.0
        downstream_slack = 0.0

        if sched_dep and sched_port:
            try:
                t_dep = pd.to_datetime(sched_dep)
                t_port = pd.to_datetime(sched_port)
                sched_transit_h = (t_port - t_dep).total_seconds() / 3600.0
                nominal_transit_h = dist_km / 35.0 if dist_km > 0 else (sched_transit_h - 20.0)
                ocean_slack = max(0.0, sched_transit_h - nominal_transit_h)
            except Exception:
                ocean_slack = 18.0

        if sched_port and sched_del:
            try:
                t_port = pd.to_datetime(sched_port)
                t_del = pd.to_datetime(sched_del)
                sched_dwell_h = (t_del - t_port).total_seconds() / 3600.0
                # Standard nominal dwell: 24h port + 14h customs + 24h hinterland = 62h
                nominal_dwell_h = 62.0
                downstream_slack = max(0.0, sched_dwell_h - nominal_dwell_h)
            except Exception:
                downstream_slack = 6.0

        calculated_initial = round(ocean_slack + downstream_slack, 2)
        initial_slack = calculated_initial if calculated_initial >= 4.0 else 18.0

    # 2. Determine Gross Disruptions & Delay Accounting
    dep_delay = float(record.get("departure_delay_hours", 0.0) or 0.0)
    final_delay = float(record.get("final_delay_hours", 0.0) or 0.0)
    raw_port = record.get("port_arrival_delay_hours")
    port_delay = float(raw_port) if (raw_port is not None and pd.notnull(raw_port)) else None

    # Checkpoint milestone deviation changes
    origin_disruption = max(0.0, dep_delay)

    if port_delay is not None:
        transit_change = port_delay - dep_delay
        transit_disruption = max(0.0, transit_change)
        dwell_change = final_delay - port_delay
        dwell_disruption = max(0.0, dwell_change)
    else:
        # Impute if port arrival telemetry is missing
        weather_sev = float(record.get("weather_severity", 2.0) or 2.0)
        port_cong = float(record.get("port_congestion", 0.3) or 0.3)
        port_proc = float(record.get("port_processing_hours", 24.0) or 24.0)
        cust_proc = float(record.get("customs_processing_hours", 16.0) or 16.0)

        transit_disruption = max(0.0, (weather_sev - 2.0) * 8.0)
        dwell_disruption = (port_cong * max(0.0, port_proc - 20.0)) + max(0.0, cust_proc - 16.0)

    gross_disruption = origin_disruption + transit_disruption + dwell_disruption

    # 3. Calculate Consumed Slack
    if final_delay <= 0:
        # Early or on-time: buffer absorbed all gross upstream disruptions
        consumed = gross_disruption
    else:
        # Arrived late: buffer absorbed the difference between gross disruptions and final delay
        absorbed = max(0.0, gross_disruption - final_delay)
        consumed = absorbed

    # Cap consumed slack at available initial buffer
    consumed_slack = min(initial_slack, round(consumed, 2))

    # 4. Remaining Buffer
    remaining_slack = max(0.0, round(initial_slack - consumed_slack, 2))

    # 5. Buffer Utilization Percentage (minute-aligned for consistency with HH:MM display)
    if initial_slack > 0:
        consumed_min = int(round(min(initial_slack, consumed) * 60))
        initial_min = int(round(initial_slack * 60))
        utilization_pct = min(100.0, round((consumed_min / initial_min) * 100.0, 1)) if initial_min > 0 else 0.0
    else:
        utilization_pct = 100.0 if gross_disruption > 0 else 0.0

    # 6. Status and Badging
    if utilization_pct >= 99.9:
        status_label = "Exhausted"
        status_badge = "🔴 Exhausted (100%)"
        status_color = "#dc2626"
        status_desc = "Buffer fully depleted; downstream disruptions compound 1:1 into customer SLA delay"
    elif utilization_pct >= 70.0:
        status_label = "Caution"
        status_badge = f"🟡 Caution ({utilization_pct:.1f}%)"
        status_color = "#ea580c"
        status_desc = "High buffer burn rate; remaining slack is tight for downstream dwell"
    else:
        status_label = "Healthy"
        status_badge = f"🟢 Healthy ({utilization_pct:.1f}%)"
        status_color = "#16a34a"
        status_desc = "Buffer intact; ample absorptive capacity remains for downstream ops"

    return {
        "initial_slack_hours": initial_slack,
        "consumed_slack_hours": consumed_slack,
        "remaining_slack_hours": remaining_slack,
        "buffer_utilization_pct": utilization_pct,
        "initial_slack_display": format_buffer_duration(initial_slack),
        "consumed_slack_display": format_buffer_duration(consumed_slack),
        "remaining_slack_display": format_buffer_duration(remaining_slack),
        "utilization_display": f"{utilization_pct:.1f}%",
        "status_label": status_label,
        "status_badge": status_badge,
        "status_color": status_color,
        "status_desc": status_desc,
        "gross_disruption_hours": round(gross_disruption, 2),
    }


