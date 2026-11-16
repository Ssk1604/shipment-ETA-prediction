"""
Application Service Layer.

Provides data retrieval, model orchestration, and simulation integration for the FastAPI routes.
Leverages indexed SQLite database for sub-millisecond querying across large datasets.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import (
    DB_PATH,
    get_shipment_by_id as db_get_shipment_by_id,
    query_shipments as db_query_shipments,
)
from src.predict import get_inference_engine


class ShipmentService:
    def __init__(self, data_path: str = "dataset/shipments.csv", models_dir: str = "models"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.df = None
        self.engine = None
        self._initialize()

    def _initialize(self) -> None:
        """Loads dataset and inference engine into memory."""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = pd.DataFrame()

        self.engine = get_inference_engine(models_dir=self.models_dir)

    def get_shipments(
        self,
        limit: int = 50,
        offset: int = 0,
        route: Optional[str] = None,
        is_delayed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns paginated and filtered shipment records.
        Uses indexed database query if available for O(1) memory overhead.
        """
        if os.path.exists(DB_PATH):
            return db_query_shipments(limit=limit, offset=offset, route=route, is_delayed=is_delayed)

        if self.df is None or self.df.empty:
            return []

        filtered = self.df.copy()
        if route:
            filtered = filtered[filtered["route"] == route]
        if is_delayed is not None:
            filtered = filtered[filtered["is_delayed_24h"] == is_delayed]

        paginated = filtered.iloc[offset : offset + limit]
        return paginated.where(pd.notnull(paginated), None).to_dict(orient="records")

    def get_shipment_by_id(self, shipment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single shipment by its unique ID via primary key index."""
        if os.path.exists(DB_PATH):
            record = db_get_shipment_by_id(shipment_id)
            if record:
                return record

        if self.df is None or self.df.empty:
            return None
        match = self.df[self.df["shipment_id"] == shipment_id]
        if match.empty:
            return None
        record = match.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in record.items()}

    def predict(self, shipment_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs single ETA and delay risk prediction."""
        return self.engine.predict_single(shipment_dict)

    def predict_batch(self, shipments_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runs high-throughput vectorized batch prediction."""
        return self.engine.predict_batch(shipments_list)

    def simulate(
        self,
        disruption_stage: str,
        additional_hours: float,
        shipment_id: Optional[str] = None,
        shipment_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulates delay propagation under additional disruption."""
        if shipment_id:
            record = self.get_shipment_by_id(shipment_id)
            if not record:
                raise ValueError(f"Shipment {shipment_id} not found.")
            data = record
        elif shipment_data:
            data = shipment_data
        else:
            raise ValueError("Must provide either shipment_id or shipment_data.")

        return self.engine.simulate_delay_impact_on_shipment(
            data,
            disruption_stage=disruption_stage,
            additional_hours=additional_hours,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Loads model evaluation metrics from metrics.json."""
        metrics_file = os.path.join(self.models_dir, "metrics.json")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError("Metrics not found. Ensure models have been trained.")
        with open(metrics_file, "r") as f:
            return json.load(f)

    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """Retrieves top global feature importance from metrics."""
        metrics = self.get_metrics()
        return metrics.get("top_features", [])


# Singleton instance
_SERVICE_INSTANCE: Optional[ShipmentService] = None


def get_shipment_service() -> ShipmentService:
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = ShipmentService()
    return _SERVICE_INSTANCE
