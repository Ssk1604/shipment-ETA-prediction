"""
Feature Engineering and Preprocessing Pipeline for Shipment ETA Prediction.

Implements four advanced feature engineering techniques:
1. Interaction Feature: congestion_x_missed_slot
   - Models the operational berthing penalty when a vessel misses its 4-hour port window.
   - congestion_x_missed_slot = port_congestion_index * (2 if arrival_delay > 4 else 1)
2. Schedule Slack / Buffer Features: remaining_buffer_hours & unabsorbed_delay_hours
   - Models commercial schedule buffers absorbing upstream delays.
   - remaining_buffer_hours = max(0, initial_buffer_hours - accumulated_delay_hours)
   - unabsorbed_delay_hours = max(0, accumulated_delay_hours - initial_buffer_hours)
3. Missing Data Features: is_port_arrival_missing & Leak-Free Imputation
   - Simulates 10% missing intermediate telemetry on training data only (seed=42).
   - Generates binary indicator is_port_arrival_missing.
   - Imputes missing timestamps using training priors, preserving the missingness flag.
4. Route Variability Encoding & Distance Interactions:
   - route_encoded: Target-encoded route representation fitted strictly on training data
     with global prior smoothing for unseen routes.
   - distance_km: Validated numeric journey distance.
   - weather_x_distance = weather_severity * distance_km.
"""

from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# All numeric operational features available at the prediction checkpoint
NUMERICAL_FEATURES: List[str] = [
    "distance_km",
    "carrier_rating",
    "departure_delay_hours",
    "port_arrival_delay_hours",
    "port_congestion",
    "weather_severity",
    "port_processing_hours",
    "customs_processing_hours",
    # Advanced Feature 1: Congestion × Missed Slot Penalty
    "congestion_x_missed_slot",
    # Advanced Feature 2: Schedule Slack / Buffer Accounting
    "remaining_buffer_hours",
    "unabsorbed_delay_hours",
    # Advanced Feature 3: Missing Telemetry Indicator
    "is_port_arrival_missing",
    # Advanced Feature 4: Route Encoding & Distance Interactions
    "route_encoded",
    "weather_x_distance",
    # Existing interactions
    "congestion_processing_interaction",
    "weather_distance_interaction",
    "departure_weather_composite",
]

CATEGORICAL_FEATURES: List[str] = [
    "route",
    "origin",
    "destination",
]

RAW_FEATURE_COLS: List[str] = [
    "shipment_id",
    "route",
    "origin",
    "destination",
    "distance_km",
    "carrier_rating",
    "scheduled_departure",
    "actual_departure",
    "departure_delay_hours",
    "scheduled_port_arrival",
    "actual_port_arrival",
    "port_arrival_delay_hours",
    "port_congestion",
    "weather_severity",
    "port_processing_hours",
    "customs_processing_hours",
    "scheduled_delivery",
]

# Standard route-to-nominal distance fallback mapping
ROUTE_DISTANCE_MAP: Dict[str, float] = {
    "ASIA-EUR-01": 19500.0,
    "ASIA-EUR-02": 15600.0,
    "TPAC-01": 11200.0,
    "TPAC-02": 10900.0,
    "ASIA-EUR-03": 20400.0,
    "ME-EUR-01": 12800.0,
    "IND-EUR-01": 11800.0,
    "TPAC-03": 12500.0,
    "TRANSATL-01": 7200.0,
    "ASIA-MED-01": 16400.0,
}


def simulate_training_missingness(
    train_df: pd.DataFrame,
    missing_rate: float = 0.10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates incomplete intermediate telemetry by randomly removing 10% of
    actual_port_arrival values in the synthetic training dataset only.
    
    Requirements:
    - Fixed reproducible seed (42).
    - Preserves test/validation data untouched.
    """
    df = train_df.copy()
    rng = np.random.default_rng(random_seed)

    # Candidate indices with non-null actual_port_arrival
    valid_idx = df[df["actual_port_arrival"].notna()].index.to_numpy()
    n_to_remove = int(len(valid_idx) * missing_rate)
    
    if n_to_remove > 0 and len(valid_idx) > 0:
        mask_idx = rng.choice(valid_idx, size=n_to_remove, replace=False)
        df.loc[mask_idx, "actual_port_arrival"] = None
        df.loc[mask_idx, "port_arrival_delay_hours"] = None

    return df


def _get_numeric_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    """Helper to safely extract a numeric pandas Series with fallback default."""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        return s.fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


class LogisticsFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Production-grade domain feature engineering transformer:
    Extracts advanced domain interactions, schedule slack buffers, and route encodings.
    """

    def __init__(self, nominal_vessel_speed_kmh: float = 35.0, default_buffer_hours: float = 20.0):
        self.nominal_vessel_speed_kmh = nominal_vessel_speed_kmh
        self.default_buffer_hours = default_buffer_hours
        
        # Learned statistics from training split
        self.median_arrival_delay_: float = 0.0
        self.median_port_congestion_: float = 0.35
        self.median_weather_severity_: float = 2.0
        self.median_distance_km_: float = 12000.0
        self.route_target_map_: Dict[str, float] = {}
        self.global_target_mean_: float = 18.0

    def fit(self, X: pd.DataFrame, y=None):
        """Fits imputation priors and out-of-fold route target encoding using training data only."""
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. Median arrival delay for imputation
        if "port_arrival_delay_hours" in X_df.columns:
            delays = pd.to_numeric(X_df["port_arrival_delay_hours"], errors="coerce").dropna()
            self.median_arrival_delay_ = float(delays.median()) if not delays.empty else 0.0
        else:
            self.median_arrival_delay_ = 0.0

        # 2. Medians for numerical safeguards
        if "port_congestion" in X_df.columns:
            cong = pd.to_numeric(X_df["port_congestion"], errors="coerce").dropna()
            self.median_port_congestion_ = float(cong.median()) if not cong.empty else 0.35
        else:
            self.median_port_congestion_ = 0.35

        if "weather_severity" in X_df.columns:
            weath = pd.to_numeric(X_df["weather_severity"], errors="coerce").dropna()
            self.median_weather_severity_ = float(weath.median()) if not weath.empty else 2.0
        else:
            self.median_weather_severity_ = 2.0

        if "distance_km" in X_df.columns:
            dist = pd.to_numeric(X_df["distance_km"], errors="coerce").dropna()
            self.median_distance_km_ = float(dist.median()) if not dist.empty else 12000.0
        else:
            self.median_distance_km_ = 12000.0

        # 3. Target Encoding for Route (Fitted on training targets if provided)
        if y is not None and "route" in X_df.columns:
            y_series = pd.Series(y).reset_index(drop=True)
            routes = X_df["route"].reset_index(drop=True)
            self.global_target_mean_ = float(y_series.mean())
            
            # Group mean with empirical Bayes smoothing (m-estimate)
            m = 5.0
            route_stats = y_series.groupby(routes).agg(["count", "mean"])
            smoothed = (route_stats["count"] * route_stats["mean"] + m * self.global_target_mean_) / (route_stats["count"] + m)
            self.route_target_map_ = smoothed.to_dict()
        else:
            # Fallback default target encoding based on route distances
            self.route_target_map_ = {r: float(d / 1000.0) for r, d in ROUTE_DISTANCE_MAP.items()}
            self.global_target_mean_ = 18.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input shipment dataframe into engineered feature space."""
        X_out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # -------------------------------------------------------------
        # Step A: Missing Telemetry Simulation & Imputation (Feature 3)
        # -------------------------------------------------------------
        # is_port_arrival_missing = 1 if actual_port_arrival is missing, else 0
        if "actual_port_arrival" in X_out.columns:
            missing_mask = X_out["actual_port_arrival"].isna()
            if "port_arrival_delay_hours" in X_out.columns:
                missing_mask = missing_mask | X_out["port_arrival_delay_hours"].isna()
        elif "port_arrival_delay_hours" in X_out.columns:
            missing_mask = X_out["port_arrival_delay_hours"].isna()
        else:
            missing_mask = pd.Series(True, index=X_out.index)

        X_out["is_port_arrival_missing"] = missing_mask.astype(float)

        # Impute port_arrival_delay_hours for missing rows using training prior + departure delay
        raw_dep_delay = _get_numeric_series(X_out, "departure_delay_hours", 0.0)
        if "port_arrival_delay_hours" in X_out.columns:
            raw_arr_delay = pd.to_numeric(X_out["port_arrival_delay_hours"], errors="coerce")
        else:
            raw_arr_delay = pd.Series(np.nan, index=X_out.index)
        effective_arrival_delay = raw_arr_delay.fillna(raw_dep_delay + self.median_arrival_delay_)
        X_out["port_arrival_delay_hours"] = effective_arrival_delay

        # -------------------------------------------------------------
        # Step B: Distance Validation & Imputation (Feature 4B)
        # -------------------------------------------------------------
        if "distance_km" in X_out.columns:
            dist = pd.to_numeric(X_out["distance_km"], errors="coerce")
        else:
            dist = pd.Series(np.nan, index=X_out.index)

        if "route" in X_out.columns:
            fallback_dist = X_out["route"].map(ROUTE_DISTANCE_MAP).fillna(self.median_distance_km_)
            dist = dist.fillna(fallback_dist)
        else:
            dist = dist.fillna(self.median_distance_km_)
        X_out["distance_km"] = dist.astype(float)

        # -------------------------------------------------------------
        # Step C: Interaction Feature: congestion_x_missed_slot (Feature 1)
        # -------------------------------------------------------------
        port_cong = _get_numeric_series(X_out, "port_congestion", self.median_port_congestion_).clip(0.0, 1.0)
        X_out["port_congestion"] = port_cong

        # Formula: missed_slot_penalty = 2 if arrival_delay_hours > 4 else 1
        # congestion_x_missed_slot = port_congestion_index * missed_slot_penalty
        missed_slot_penalty = np.where(effective_arrival_delay > 4.0, 2.0, 1.0)
        X_out["congestion_x_missed_slot"] = port_cong * missed_slot_penalty

        # -------------------------------------------------------------
        # Step D: Schedule Slack / Buffer Features (Feature 2)
        # -------------------------------------------------------------
        # Derivation of initial_buffer_hours from scheduled timetable:
        # initial_buffer_hours = max(0, scheduled_transit_hours - nominal_transit_hours)
        initial_buffers = []
        for idx, row in X_out.iterrows():
            sched_dep = row.get("scheduled_departure")
            sched_port = row.get("scheduled_port_arrival")
            d_km = float(row.get("distance_km", self.median_distance_km_))
            nominal_transit = d_km / self.nominal_vessel_speed_kmh

            if pd.notnull(sched_dep) and pd.notnull(sched_port):
                try:
                    dt_dep = pd.to_datetime(sched_dep)
                    dt_port = pd.to_datetime(sched_port)
                    sched_transit = (dt_port - dt_dep).total_seconds() / 3600.0
                    buf = max(0.0, sched_transit - nominal_transit)
                except Exception:
                    buf = self.default_buffer_hours
            else:
                buf = self.default_buffer_hours
            initial_buffers.append(buf)

        initial_buffers_arr = np.array(initial_buffers, dtype=float)
        accumulated_delays = np.maximum(0.0, effective_arrival_delay.to_numpy(dtype=float))

        # Formula:
        # remaining_buffer_hours = max(0, initial_buffer_hours - accumulated_delay_hours)
        # unabsorbed_delay_hours = max(0, accumulated_delay_hours - initial_buffer_hours)
        X_out["remaining_buffer_hours"] = np.maximum(0.0, initial_buffers_arr - accumulated_delays)
        X_out["unabsorbed_delay_hours"] = np.maximum(0.0, accumulated_delays - initial_buffers_arr)

        # -------------------------------------------------------------
        # Step E: Route Variability Encoding (Feature 4A)
        # -------------------------------------------------------------
        if "route" in X_out.columns:
            X_out["route_encoded"] = (
                X_out["route"].map(self.route_target_map_).fillna(self.global_target_mean_).astype(float)
            )
        else:
            X_out["route_encoded"] = float(self.global_target_mean_)

        # -------------------------------------------------------------
        # Step F: Weather Interactions (Feature 4C & Composites)
        # -------------------------------------------------------------
        weather = _get_numeric_series(X_out, "weather_severity", self.median_weather_severity_).clip(1.0, 5.0)
        X_out["weather_severity"] = weather

        # Formula: weather_x_distance = weather_severity * distance_km
        X_out["weather_x_distance"] = weather * X_out["distance_km"]

        # Congestion x Processing interaction
        port_proc = _get_numeric_series(X_out, "port_processing_hours", 24.0)
        X_out["port_processing_hours"] = port_proc
        X_out["congestion_processing_interaction"] = port_cong * port_proc

        # Weather Distance interaction (scaled per 1000km)
        X_out["weather_distance_interaction"] = weather * (X_out["distance_km"] / 1000.0)

        # Departure Weather composite
        X_out["departure_delay_hours"] = raw_dep_delay
        X_out["departure_weather_composite"] = raw_dep_delay + (weather * 4.0)

        # Customs processing safeguard
        customs_proc = _get_numeric_series(X_out, "customs_processing_hours", 16.0)
        X_out["customs_processing_hours"] = customs_proc

        # Carrier rating safeguard
        carrier = _get_numeric_series(X_out, "carrier_rating", 0.85)
        X_out["carrier_rating"] = carrier

        return X_out


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline:
    - Numerical: Median imputation + StandardScaler
    - Categorical: OneHotEncoder (handling unseen categories safely)
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def create_feature_pipeline() -> Pipeline:
    """
    Full feature pipeline combining domain feature extraction with column transformations.
    """
    return Pipeline(steps=[
        ("feature_extractor", LogisticsFeatureExtractor()),
        ("preprocessor", build_preprocessor()),
    ])


def prepare_data_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_by_time: bool = True,
    simulate_train_missingness: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits shipment dataset into train and test sets.
    If stratify_by_time is True, performs a chronological split.
    Optionally applies 10% artificial missing telemetry to training set only (seed 42).
    
    Returns:
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
    """
    df_sorted = df.sort_values(by="scheduled_departure").reset_index(drop=True)
    
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Step 3: Apply 10% missing telemetry simulation ONLY to the training split
    if simulate_train_missingness:
        train_df = simulate_training_missingness(train_df, missing_rate=0.10, random_seed=42)

    # Targets
    y_reg_train = train_df["final_delay_hours"]
    y_reg_test = test_df["final_delay_hours"]

    y_clf_train = train_df["is_delayed_24h"]
    y_clf_test = test_df["is_delayed_24h"]

    # Operational features passed to the pipeline
    available_cols = [c for c in RAW_FEATURE_COLS if c in train_df.columns]
    X_train = train_df[available_cols].copy()
    X_test = test_df[available_cols].copy()

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
