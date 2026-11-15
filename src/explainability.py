"""
Explainability Module for Shipment ETA Predictions.

Provides dual-level explainability:
1. Global Feature Importance:
   Extracts tree-based Gini feature importance across all engineered features from the
   Random Forest models, illustrating overall statistical predictive drivers.
2. Local Stage-Level Delay Attribution (Waterfall):
   Calculates the deterministic operational delay accounting for any individual shipment,
   distinguishing:
   - Inherited delays vs. stage-specific disruptions.
   - Delay absorbed by planned commercial schedule slack.
   - Explains the distinction between statistical correlation and physical causality.
"""

import json
import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.delay_propagation import ShipmentDelayBreakdown


def extract_feature_names(pipeline: Pipeline) -> List[str]:
    """
    Extracts transformed feature names from a trained Scikit-learn Pipeline.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    if not preprocessor:
        return []

    feature_names: List[str] = []
    
    # Numerical features
    num_cols = preprocessor.transformers_[0][2]
    feature_names.extend(num_cols)

    # Categorical features
    cat_step = preprocessor.transformers_[1][1]
    encoder = cat_step.named_steps.get("encoder")
    cat_cols = preprocessor.transformers_[1][2]
    
    if encoder and hasattr(encoder, "get_feature_names_out"):
        cat_encoded_names = encoder.get_feature_names_out(cat_cols)
        feature_names.extend(cat_encoded_names.tolist())
    else:
        feature_names.extend(cat_cols)

    return feature_names


def get_global_feature_importance(
    pipeline: Pipeline,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Retrieves the top N global feature importances from a trained Random Forest pipeline.
    """
    model = pipeline.named_steps.get("model")
    if not model or not hasattr(model, "feature_importances_"):
        return []

    raw_names = extract_feature_names(pipeline)
    importances = model.feature_importances_

    # Fallback if length mismatch
    if len(raw_names) != len(importances):
        raw_names = [f"feature_{i}" for i in range(len(importances))]

    sorted_indices = np.argsort(importances)[::-1]
    results = []
    for idx in sorted_indices[:top_n]:
        results.append({
            "feature": raw_names[idx],
            "importance": round(float(importances[idx]), 4),
        })

    return results


def plot_feature_importance(
    importances: List[Dict[str, Any]],
    output_path: str = "reports/figures/feature_importance.png",
) -> str:
    """Generates a horizontal bar chart of top global feature importances."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not importances:
        return output_path

    df = pd.DataFrame(importances).sort_values(by="importance", ascending=True)
    
    plt.figure(figsize=(9, 6))
    bars = plt.barh(df["feature"], df["importance"], color="#2ca02c", alpha=0.85, edgecolor="k")
    plt.xlabel("Gini Feature Importance (Normalized)", fontsize=11, fontweight="bold")
    plt.ylabel("Engineered Feature", fontsize=11, fontweight="bold")
    plt.title("Global Feature Importance (Random Forest Regressor)", fontsize=13, fontweight="bold")
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                 va="center", ha="left", fontsize=9)

    plt.xlim(0, max(df["importance"]) * 1.18)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def explain_shipment_delay(
    shipment_row: pd.Series,
    predicted_delay_hours: float,
) -> Dict[str, Any]:
    """
    Produces a local, multi-stage operational attribution for a specific shipment.

    Distinguishes:
      - Departure Delay: Delay at vessel release from origin terminal.
      - Ocean Transit Disruption: Extra sailing hours induced by weather/speed.
      - Port Congestion Wait: Dwell time in anchorage outside berth.
      - Customs Clearance Wait: Time spent in customs clearance & inspection.
      - Absorbed Delay: Schedule slack planned into the commercial route that
        prevented upstream delay from propagating downstream.

    NOTE ON EXPLAINABILITY METHODOLOGY:
    Global feature importance reveals what the model relied on statistically across
    the training dataset (correlation and variance reduction). In contrast, this
    stage attribution represents the operational logistics accounting (causal timeline physics).
    """
    dep_delay = float(shipment_row.get("departure_delay_hours", 0.0) or 0.0)
    weather_sev = float(shipment_row.get("weather_severity", 2.0) or 2.0)
    port_cong = float(shipment_row.get("port_congestion", 0.3) or 0.3)
    port_proc = float(shipment_row.get("port_processing_hours", 24.0) or 24.0)
    customs_proc = float(shipment_row.get("customs_processing_hours", 16.0) or 16.0)
    
    # Operational attribution estimates
    est_ocean_weather_delay = round(max(0.0, (weather_sev - 2.0) * 8.0), 2)
    est_congestion_delay = round(port_cong * max(0.0, port_proc - 20.0), 2)
    est_customs_delay = round(max(0.0, customs_proc - 16.0), 2)
    
    # Slack absorbed
    gross_disruption = dep_delay + est_ocean_weather_delay + est_congestion_delay + est_customs_delay
    slack_absorbed = max(0.0, round(gross_disruption - predicted_delay_hours, 2))

    risk_category = "LOW" if predicted_delay_hours < 12.0 else ("MEDIUM" if predicted_delay_hours < 24.0 else "HIGH")

    return {
        "shipment_id": str(shipment_row.get("shipment_id", "UNKNOWN")),
        "predicted_delay_hours": round(predicted_delay_hours, 2),
        "delay_risk_category": risk_category,
        "stages": [
            {"stage": "Origin Departure", "delay_contribution_hours": round(dep_delay, 2), "description": "Terminal & gate release delay"},
            {"stage": "Ocean Transit (Weather)", "delay_contribution_hours": est_ocean_weather_delay, "description": "Sailing slowdown due to sea state"},
            {"stage": "Port Congestion", "delay_contribution_hours": est_congestion_delay, "description": "Anchorage waiting time before berthing"},
            {"stage": "Customs Inspection", "delay_contribution_hours": est_customs_delay, "description": "Regulatory clearance & documentation holds"},
            {"stage": "Schedule Slack Absorption", "delay_contribution_hours": -slack_absorbed, "description": "Commercial schedule buffer absorbing upstream delay"},
        ],
        "net_explained_delay_hours": round(predicted_delay_hours, 2),
        "methodology_note": (
            "Attributions reflect physical logistics stage accounting and schedule slack absorption. "
            "Global feature importance reflects ML model variance reduction across all shipments."
        ),
    }
