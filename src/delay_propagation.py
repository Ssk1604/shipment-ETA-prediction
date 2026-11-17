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


