"""
Training and Evaluation Pipeline for Shipment ETA & Delay Risk Models.

Trains:
1. Baseline Model: Historical training mean predictor.
2. Model A: RandomForestRegressor for final delay in hours.
3. Model B: RandomForestClassifier for binary delay risk (≥ 24 hours).

Evaluates models using chronological train/test splits, generates plots in reports/figures/,
and serializes models and metrics to models/.
"""

import os
import sys

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.evaluation import (
    evaluate_classification,
    evaluate_regression,
    plot_confusion_matrix,
    plot_regression_performance,
    save_metrics,
)
from src.explainability import (
    get_global_feature_importance,
    plot_feature_importance,
)
from src.feature_engineering import (
    build_preprocessor,
    LogisticsFeatureExtractor,
    prepare_data_splits,
)


def train_and_evaluate(
    data_path: str = "dataset/shipments.csv",
    output_dir: str = "models",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the end-to-end model training, evaluation, explainability extraction,
    and artifact persistence.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    print(f"Loading shipment records from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")

    # 1. Chronological Split (prevents future information leaking into the past)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data_splits(
        df, test_size=0.2, stratify_by_time=True
    )
    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # 2. Baseline Model (Mean Delay Predictor)
    baseline_mean = float(y_reg_train.mean())
    baseline_preds = np.full(shape=len(y_reg_test), fill_value=baseline_mean)
    baseline_metrics = evaluate_regression(y_reg_test.values, baseline_preds, model_name="Baseline (Historical Mean)")
    print("\n--- Baseline Model Performance ---")
    print(f"MAE: {baseline_metrics['mae_hours']} hours | RMSE: {baseline_metrics['rmse_hours']} hours | R2: {baseline_metrics['r2_score']}")

    # 3. Model A: RandomForestRegressor (Final Delay Hours)
    regressor = RandomForestRegressor(
        n_estimators=120,
        max_depth=7,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_seed,
        n_jobs=-1,
    )
    reg_pipeline = Pipeline(steps=[
        ("feature_extractor", LogisticsFeatureExtractor()),
        ("preprocessor", build_preprocessor()),
        ("model", regressor),
    ])

    print("\nTraining Model A: RandomForestRegressor...")
    reg_pipeline.fit(X_train, y_reg_train)
    reg_preds = reg_pipeline.predict(X_test)
    reg_metrics = evaluate_regression(y_reg_test.values, reg_preds, model_name="RandomForestRegressor")
    print(f"Model A MAE: {reg_metrics['mae_hours']} hours | RMSE: {reg_metrics['rmse_hours']} hours | R2: {reg_metrics['r2_score']}")

    # 4. Model B: RandomForestClassifier (Delay Risk >= 24h)
    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    clf_pipeline = Pipeline(steps=[
        ("feature_extractor", LogisticsFeatureExtractor()),
        ("preprocessor", build_preprocessor()),
        ("model", classifier),
    ])

    print("\nTraining Model B: RandomForestClassifier...")
    clf_pipeline.fit(X_train, y_clf_train)
    clf_preds = clf_pipeline.predict(X_test)
    clf_probs = clf_pipeline.predict_proba(X_test)[:, 1]
    clf_metrics = evaluate_classification(
        y_clf_test.values, clf_preds, clf_probs, model_name="RandomForestClassifier"
    )
    print(f"Model B Accuracy: {clf_metrics['accuracy']:.2%} | Precision: {clf_metrics['precision']:.2%} | Recall: {clf_metrics['recall']:.2%} | F1: {clf_metrics['f1_score']:.4f} | ROC-AUC: {clf_metrics['roc_auc']:.4f}")

    # 5. Extract Explainability & Global Feature Importance
    top_importances = get_global_feature_importance(reg_pipeline, top_n=10)
    print("\nTop 5 Global Predictive Features:")
    for item in top_importances[:5]:
        print(f"  - {item['feature']}: {item['importance']:.4f}")

    # 6. Generate Figures
    fig_reg = plot_regression_performance(y_reg_test.values, reg_preds, baseline_preds)
    fig_cm = plot_confusion_matrix(clf_metrics["confusion_matrix"])
    fig_fi = plot_feature_importance(top_importances)
    print(f"\nGenerated evaluation plots:\n  - {fig_reg}\n  - {fig_cm}\n  - {fig_fi}")

    # 7. Persist Artifacts
    joblib.dump(reg_pipeline, os.path.join(output_dir, "regressor_pipeline.joblib"))
    joblib.dump(clf_pipeline, os.path.join(output_dir, "classifier_pipeline.joblib"))
    joblib.dump({"baseline_mean_hours": baseline_mean}, os.path.join(output_dir, "baseline_model.joblib"))

    all_metrics = {
        "baseline_regression": baseline_metrics,
        "model_a_regression": reg_metrics,
        "model_b_classification": clf_metrics,
        "top_features": top_importances,
        "split_info": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "split_type": "chronological",
        },
    }
    save_metrics(all_metrics, os.path.join(output_dir, "metrics.json"))
    print(f"Artifacts and metrics serialized to: {output_dir}/")

    return all_metrics


if __name__ == "__main__":
    train_and_evaluate()
