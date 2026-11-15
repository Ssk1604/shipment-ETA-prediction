"""
Feature Engineering and Preprocessing Pipeline for Shipment ETA Prediction.

Transforms raw shipment operational records into structured ML features while strictly
preventing target leakage:
- Pre-checkpoint features only (no downstream actual delivery data).
- Interaction features capturing compounding domain physics:
    * congestion_processing_interaction: port_congestion × port_processing_hours
    * weather_distance_interaction: weather_severity × (distance_km / 1000)
    * departure_weather_composite: departure_delay_hours + (weather_severity × 5)
- ColumnTransformer pipeline handling categorical encoding, missing telemetry imputation,
  and numerical scaling.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Operational feature columns available at the prediction checkpoint
NUMERICAL_FEATURES: List[str] = [
    "distance_km",
    "carrier_rating",
    "departure_delay_hours",
    "port_arrival_delay_hours",
    "port_congestion",
    "weather_severity",
    "port_processing_hours",
    "customs_processing_hours",
    "congestion_processing_interaction",
    "weather_distance_interaction",
    "departure_weather_composite",
]

CATEGORICAL_FEATURES: List[str] = [
    "route",
    "origin",
    "destination",
]


class LogisticsFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer:
    Generates interaction terms modeling logistics physics and slack propagation.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Fill missing port_arrival_delay_hours temporarily if needed before imputer,
        # or construct surrogate features
        # 1. Congestion × Port Processing interaction
        # High congestion amplifies terminal dwell time non-linearly
        port_cong = pd.to_numeric(X["port_congestion"], errors="coerce").fillna(0.3)
        port_proc = pd.to_numeric(X["port_processing_hours"], errors="coerce").fillna(24.0)
        X["congestion_processing_interaction"] = port_cong * port_proc

        # 2. Weather × Distance interaction
        # High sea state disrupts long voyages significantly more than short coastal routes
        weather = pd.to_numeric(X["weather_severity"], errors="coerce").fillna(2.0)
        dist = pd.to_numeric(X["distance_km"], errors="coerce").fillna(10000.0)
        X["weather_distance_interaction"] = weather * (dist / 1000.0)

        # 3. Departure Delay + Weather Composite
        # Early departure delay compounding with bad en-route weather
        dep_delay = pd.to_numeric(X["departure_delay_hours"], errors="coerce").fillna(0.0)
        X["departure_weather_composite"] = dep_delay + (weather * 4.0)

        return X


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline:
    - Numerical: Median imputation (for missing telemetry) + StandardScaler
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits shipment dataset into train and test sets.
    If stratify_by_time is True, performs a chronological split (earlier shipments train,
    later shipments test), reflecting realistic deployment conditions.
    
    Returns:
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
    """
    df_sorted = df.sort_values(by="scheduled_departure").reset_index(drop=True)
    
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Targets
    y_reg_train = train_df["final_delay_hours"]
    y_reg_test = test_df["final_delay_hours"]

    y_clf_train = train_df["is_delayed_24h"]
    y_clf_test = test_df["is_delayed_24h"]

    # Raw features passed to the pipeline
    raw_feature_cols = [
        "shipment_id", "route", "origin", "destination", "distance_km",
        "carrier_rating", "departure_delay_hours", "port_arrival_delay_hours",
        "port_congestion", "weather_severity", "port_processing_hours",
        "customs_processing_hours",
    ]
    X_train = train_df[raw_feature_cols]
    X_test = test_df[raw_feature_cols]

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
