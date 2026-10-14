"""
Model Explainability and Feature Importance Module.

Provides global feature attribution (permutation importance and tree/linear importances)
as well as local explanation heuristics for individual customer churn risk scoring.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.inspection import permutation_importance
from src.preprocessing import (
    get_feature_names_from_preprocessor,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_permutation_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes model-agnostic permutation feature importance on raw test features.

    Parameters
    ----------
    pipeline : Any
        Fitted Scikit-Learn Pipeline.
    X_test : pd.DataFrame
        Held-out test features (raw DataFrame).
    y_test : pd.Series
        Held-out ground-truth target.
    n_repeats : int
        Number of shuffling iterations per feature.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping feature names to mean importance score and standard deviation.
    """
    logger.info("Computing permutation importance across raw features on test set...")
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )

    feature_names = list(X_test.columns)
    importances = []
    for name, mean_score, std in zip(
        feature_names, result.importances_mean, result.importances_std
    ):
        importances.append({
            "feature": name,
            "importance_mean": round(float(mean_score), 5),
            "importance_std": round(float(std), 5),
        })

    # Sort descending by mean importance
    importances.sort(key=lambda x: x["importance_mean"], reverse=True)
    return {"ranking": importances}


def extract_model_feature_importance(pipeline: Any) -> Dict[str, Any]:
    """
    Extracts native feature importances (for trees/ensembles) or coefficients (for linear models).

    Parameters
    ----------
    pipeline : Any
        Fitted Scikit-Learn Pipeline.

    Returns
    -------
    Dict[str, Any]
        Dictionary with encoded feature names and their corresponding weights/importances.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    results: List[Dict[str, Any]] = []

    if hasattr(classifier, "feature_importances_"):
        raw_vals = classifier.feature_importances_
        metric_name = "gini_importance"
    elif hasattr(classifier, "coef_"):
        raw_vals = classifier.coef_[0]
        metric_name = "coefficient"
    else:
        logger.warning("Classifier has neither feature_importances_ nor coef_ attribute.")
        return {"features": []}

    for name, val in zip(feature_names, raw_vals):
        results.append({
            "feature": name,
            "value": round(float(val), 5),
            "abs_value": round(float(abs(val)), 5),
            "metric": metric_name,
        })

    results.sort(key=lambda x: x["abs_value"], reverse=True)
    return {"features": results}


def explain_individual_prediction(
    pipeline: Any,
    customer_record: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Generates local feature explanations indicating key drivers of churn probability
    for an individual customer record.

    Parameters
    ----------
    pipeline : Any
        Fitted Scikit-Learn Pipeline.
    customer_record : pd.DataFrame
        Single-row customer DataFrame.
    top_k : int
        Number of top risk and protective factors to return.

    Returns
    -------
    Dict[str, Any]
        Risk probability, key churn drivers, and protective retention factors.
    """
    proba = float(pipeline.predict_proba(customer_record)[0, 1])
    row = customer_record.iloc[0]

    risk_factors: List[str] = []
    protective_factors: List[str] = []

    # Business rule heuristics grounded in Telco empirical feature relationships:
    contract = str(row.get("Contract", ""))
    tenure = float(row.get("tenure", 0))
    monthly = float(row.get("MonthlyCharges", 0))
    internet = str(row.get("InternetService", ""))
    payment = str(row.get("PaymentMethod", ""))
    tech_support = str(row.get("TechSupport", ""))
    online_sec = str(row.get("OnlineSecurity", ""))

    # Contract analysis
    if "month" in contract.lower():
        risk_factors.append("Month-to-month contract (3x higher churn propensity vs long-term)")
    elif "two year" in contract.lower():
        protective_factors.append("Two-year contract commitment strongly inhibits churn")
    elif "one year" in contract.lower():
        protective_factors.append("One-year contract provides stability")

    # Tenure analysis
    if tenure <= 6:
        risk_factors.append(f"Early customer lifecycle ({int(tenure)} months tenure): highest risk onboarding window")
    elif tenure >= 48:
        protective_factors.append(f"High brand loyalty ({int(tenure)} months tenure)")

    # Monthly charges
    if monthly >= 80:
        risk_factors.append(f"High monthly charges (${monthly:.2f}/mo) increases price sensitivity")
    elif monthly <= 30:
        protective_factors.append(f"Low monthly commitment (${monthly:.2f}/mo)")

    # Internet service & tech support
    if "fiber" in internet.lower():
        risk_factors.append("Fiber optic internet subscription (statistically higher churn rate in Telco)")
    if tech_support.lower() == "no":
        risk_factors.append("No Tech Support subscription (elevates support friction churn)")
    elif tech_support.lower() == "yes":
        protective_factors.append("Active Tech Support subscriber (reduces technical dissatisfaction)")

    if online_sec.lower() == "no":
        risk_factors.append("No Online Security add-on (lower account stickiness)")
    elif online_sec.lower() == "yes":
        protective_factors.append("Online Security enabled (increases product stickiness)")

    # Payment method
    if "electronic check" in payment.lower():
        risk_factors.append("Electronic check payment (highest churn rate among payment types)")
    elif "automatic" in payment.lower():
        protective_factors.append("Automatic payment method (hands-off billing retention)")

    return {
        "churn_probability": round(proba, 4),
        "primary_risk_drivers": risk_factors[:top_k],
        "primary_protective_factors": protective_factors[:top_k],
        "disclaimer": (
            "Model predictions represent estimated statistical probabilities based on historical "
            "cohort patterns, not definitive guarantees. Use as an assistive decision-support tool."
        ),
    }
