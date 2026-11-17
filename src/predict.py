"""
Prediction and Simulation Inference Engine.

Loads serialized models and provides unified functions for:
1. Single and batch shipment ETA & risk prediction.
2. Downstream delay propagation simulation under operational disruptions.
3. Local stage attribution generation.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.delay_propagation import simulate_delay_impact, format_hours_to_hhmm
from src.explainability import explain_shipment_delay


class ShipmentInferenceEngine:
    """Production inference engine for ETA prediction and delay propagation."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.reg_pipeline = None
        self.clf_pipeline = None
        self.baseline_meta = None
        self.load_models()

    def load_models(self) -> None:
        """Loads trained pipelines and metadata from disk."""
        reg_path = os.path.join(self.models_dir, "regressor_pipeline.joblib")
        clf_path = os.path.join(self.models_dir, "classifier_pipeline.joblib")
        baseline_path = os.path.join(self.models_dir, "baseline_model.joblib")

        if not os.path.exists(reg_path) or not os.path.exists(clf_path):
            raise FileNotFoundError(
                f"Trained model files not found in {self.models_dir}. "
                "Ensure `python -m src.train_model` has been executed."
            )

        self.reg_pipeline = joblib.load(reg_path)
        self.clf_pipeline = joblib.load(clf_path)
        if os.path.exists(baseline_path):
            self.baseline_meta = joblib.load(baseline_path)

    def predict_single(self, shipment_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates ETA, delay hours, and risk classification for a single shipment record.
        """
        df_input = pd.DataFrame([shipment_dict])

        # 1. Regressor Prediction
        pred_delay_hours = float(self.reg_pipeline.predict(df_input)[0])
        pred_delay_hours = round(pred_delay_hours, 2)

        # 2. Classifier Prediction & Probability
        pred_prob = float(self.clf_pipeline.predict_proba(df_input)[0, 1])
        pred_is_delayed = int(pred_prob >= 0.5)

        # 3. Dynamic ETA Calculation
        scheduled_delivery_raw = shipment_dict.get("scheduled_delivery")
        if scheduled_delivery_raw:
            try:
                sched_dt = datetime.fromisoformat(str(scheduled_delivery_raw))
                predicted_eta_dt = sched_dt + timedelta(hours=pred_delay_hours)
                predicted_eta_iso = predicted_eta_dt.isoformat()
            except Exception:
                predicted_eta_iso = None
        else:
            predicted_eta_iso = None

        # 4. Risk Tier
        if pred_prob < 0.35:
            risk_tier = "LOW"
        elif pred_prob < 0.70:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "HIGH"

        # 5. Local Explainability Breakdown
        explanation = explain_shipment_delay(pd.Series(shipment_dict), pred_delay_hours)

        return {
            "shipment_id": shipment_dict.get("shipment_id", "CUSTOM"),
            "predicted_delay_hours": pred_delay_hours,
            "predicted_delay_formatted": format_hours_to_hhmm(pred_delay_hours),
            "predicted_eta": predicted_eta_iso,
            "scheduled_delivery": scheduled_delivery_raw,
            "delay_probability": round(pred_prob, 4),
            "is_delayed_24h": pred_is_delayed,
            "risk_tier": risk_tier,
            "explanation": explanation,
        }

    def predict_batch(self, shipments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        High-throughput vectorized batch prediction for thousands of shipments simultaneously.
        Processes entire DataFrames in vectorized C-level execution.
        """
        if not shipments:
            return []

        df_input = pd.DataFrame(shipments)

        # 1. Vectorized Regressor Prediction
        delays = np.round(self.reg_pipeline.predict(df_input), 2)

        # 2. Vectorized Classifier Prediction & Probability
        probs = self.clf_pipeline.predict_proba(df_input)[:, 1]
        is_delayed_flags = (probs >= 0.5).astype(int)

        results = []
        for i, row_dict in enumerate(shipments):
            delay_h = float(delays[i])
            prob = float(probs[i])
            flag = int(is_delayed_flags[i])

            sched_delivery = row_dict.get("scheduled_delivery")
            predicted_eta_iso = None
            if sched_delivery:
                try:
                    sched_dt = datetime.fromisoformat(str(sched_delivery))
                    predicted_eta_iso = (sched_dt + timedelta(hours=delay_h)).isoformat()
                except Exception:
                    pass

            if prob < 0.35:
                tier = "LOW"
            elif prob < 0.70:
                tier = "MEDIUM"
            else:
                tier = "HIGH"

            results.append({
                "shipment_id": row_dict.get("shipment_id", f"BATCH-{i:04d}"),
                "predicted_delay_hours": delay_h,
                "predicted_delay_formatted": format_hours_to_hhmm(delay_h),
                "predicted_eta": predicted_eta_iso,
                "scheduled_delivery": sched_delivery,
                "delay_probability": round(prob, 4),
                "is_delayed_24h": flag,
                "risk_tier": tier,
            })

        return results

    def simulate_delay_impact_on_shipment(
        self,
        shipment_dict: Dict[str, Any],
        disruption_stage: str,
        additional_hours: float,
    ) -> Dict[str, Any]:
        """
        Simulates how an operational bottleneck (e.g. extra port congestion or customs delay)
        propagates downstream and updates predicted ETA.
        """
        # Baseline prediction first
        base_prediction = self.predict_single(shipment_dict)
        base_delay = base_prediction["predicted_delay_hours"]

        # Estimate remaining buffer slack based on route and nominal operations
        # Typically 6-12 hours of inland / terminal slack remains
        remaining_slack = 8.0 if disruption_stage in ["port", "customs"] else 0.0

        simulation_math = simulate_delay_impact(
            current_final_delay_hours=base_delay,
            stage=disruption_stage,
            additional_disruption_hours=additional_hours,
            remaining_slack_hours=remaining_slack,
        )

        new_delay = simulation_math["new_final_delay_hours"]
        
        # Calculate revised ETA
        sched_delivery = shipment_dict.get("scheduled_delivery")
        revised_eta = None
        if sched_delivery:
            try:
                sched_dt = datetime.fromisoformat(str(sched_delivery))
                revised_eta = (sched_dt + timedelta(hours=new_delay)).isoformat()
            except Exception:
                pass

        return {
            "shipment_id": shipment_dict.get("shipment_id", "CUSTOM"),
            "stage_disrupted": disruption_stage,
            "additional_delay_added_hours": additional_hours,
            "absorbed_by_schedule_slack_hours": simulation_math["absorbed_by_slack"],
            "net_effective_delay_increase_hours": simulation_math["delta_impact_hours"],
            "original_predicted_delay_hours": base_delay,
            "revised_predicted_delay_hours": new_delay,
            "original_delay_formatted": format_hours_to_hhmm(base_delay),
            "revised_delay_formatted": format_hours_to_hhmm(new_delay),
            "slack_absorbed_formatted": format_hours_to_hhmm(-simulation_math["absorbed_by_slack"], show_early=False),
            "original_eta": base_prediction["predicted_eta"],
            "revised_eta": revised_eta,
            "risk_tier": "HIGH" if new_delay >= 24.0 else ("MEDIUM" if new_delay >= 12.0 else "LOW"),
        }


# Global singleton instance for efficient serving without reloading models
_ENGINE_INSTANCE: Optional[ShipmentInferenceEngine] = None


def get_inference_engine(models_dir: str = "models") -> ShipmentInferenceEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ShipmentInferenceEngine(models_dir=models_dir)
    return _ENGINE_INSTANCE
