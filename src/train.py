"""
Model Training and Comparison Pipeline.

Trains and benchmarks Logistic Regression, Random Forest, and XGBoost classifiers.
Performs 5-fold stratified cross-validation, evaluates on held-out test data,
and persists the champion pipeline alongside rich evaluation metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_loader import load_raw_data, validate_raw_data
from src.preprocessing import (
    create_preprocessor,
    split_and_save_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.evaluate import evaluate_model_performance
from src.explainability import compute_permutation_importance, extract_model_feature_importance

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_and_compare_models(
    raw_data_path: str = "data/raw/Telco-Customer-Churn.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes full reproducible training workflow:
    1. Loads raw dataset and validates schema
    2. Performs stratified train-test split (80/20)
    3. Trains Logistic Regression, Random Forest, and XGBoost
    4. Computes 5-fold CV and test evaluation metrics
    5. Saves champion pipeline to joblib and benchmark results to JSON

    Parameters
    ----------
    raw_data_path : str
        Path to raw CSV file.
    models_dir : str
        Target directory to save serialized models.
    reports_dir : str
        Target directory for evaluation reports.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Complete benchmark summary dictionary.
    """
    logger.info("--- Starting Churn Prediction Training Pipeline ---")

    # Step 1: Ingestion & Validation
    df_raw = load_raw_data(raw_data_path)
    val_info = validate_raw_data(df_raw)
    if not val_info["is_valid"]:
        raise ValueError(f"Raw data validation failed: {val_info['missing_expected_columns']}")

    # Step 2: Stratified Split & Clean Partitions
    X_train, X_test, y_train, y_test = split_and_save_data(
        df_raw, test_size=0.20, random_state=random_state
    )

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = round(neg_count / max(1, pos_count), 2)
    logger.info(
        f"Training set: {len(y_train)} samples ({pos_count} churned, {neg_count} retained). "
        f"Imbalance ratio: {scale_pos_weight:.2f}"
    )

    # Step 3: Define Candidate Models
    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    benchmark_results: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, clf in candidates.items():
        logger.info(f"Training pipeline for: {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", clf),
            ]
        )

        # 5-Fold Cross Validation on Training partition
        cv_scores_auc = cross_val_score(
            pipeline, X_train, y_train, cv=cv_strategy, scoring="roc_auc", n_jobs=-1
        )
        cv_scores_f1 = cross_val_score(
            pipeline, X_train, y_train, cv=cv_strategy, scoring="f1", n_jobs=-1
        )

        # Fit full pipeline on entire training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on held-out test set
        eval_metrics = evaluate_model_performance(name, pipeline, X_test, y_test)
        eval_metrics["cv_roc_auc_mean"] = round(float(np.mean(cv_scores_auc)), 4)
        eval_metrics["cv_roc_auc_std"] = round(float(np.std(cv_scores_auc)), 4)
        eval_metrics["cv_f1_mean"] = round(float(np.mean(cv_scores_f1)), 4)
        eval_metrics["cv_f1_std"] = round(float(np.std(cv_scores_f1)), 4)

        benchmark_results.append(eval_metrics)

    # Step 4: Champion Model Selection (prioritizing ROC-AUC and F1)
    # Composite score: 0.6 * ROC-AUC + 0.4 * F1
    benchmark_results.sort(
        key=lambda m: (0.6 * m["roc_auc"] + 0.4 * m["f1_score"]), reverse=True
    )
    champion_metric = benchmark_results[0]
    champion_name = champion_metric["model_name"]
    champion_pipeline = fitted_pipelines[champion_name]

    logger.info(
        f"★ Selected Champion Model: {champion_name} "
        f"(ROC-AUC: {champion_metric['roc_auc']:.4f}, F1: {champion_metric['f1_score']:.4f})"
    )

    # Step 5: Permutation Feature Importance & Model Coefficients
    perm_importance = compute_permutation_importance(
        champion_pipeline, X_test, y_test, n_repeats=5, random_state=random_state
    )
    native_importance = extract_model_feature_importance(champion_pipeline)

    # Step 6: Persist Outputs
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    champion_model_file = models_path / "best_churn_pipeline.joblib"
    joblib.dump(champion_pipeline, champion_model_file)
    logger.info(f"Persisted champion pipeline to {champion_model_file.resolve()}")

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    metrics_file = reports_path / "model_metrics.json"

    final_report = {
        "metadata": {
            "selected_model": champion_name,
            "evaluation_date": "2026-10-14",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "historical_churn_rate": val_info["historical_churn_rate"],
            "features_numerical": NUMERICAL_FEATURES,
            "features_categorical": CATEGORICAL_FEATURES,
        },
        "model_comparison": benchmark_results,
        "champion_metrics": champion_metric,
        "permutation_importance": perm_importance,
        "native_feature_importance": native_importance,
    }

    with open(metrics_file, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Saved comprehensive evaluation report to {metrics_file.resolve()}")

    return final_report


if __name__ == "__main__":
    train_and_compare_models()
