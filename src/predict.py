"""
Prediction and Inference Engine for Customer Churn.

Provides production scoring for single customer profiles and batch DataFrames,
assigning configurable risk tiers, actionable retention recommendations,
and local feature attributions.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    clean_telecom_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.explainability import explain_individual_prediction

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL_PATH = "models/best_churn_pipeline.joblib"

# Configurable risk thresholds
DEFAULT_LOW_THRESHOLD = 0.30
DEFAULT_HIGH_THRESHOLD = 0.60

REQUIRED_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_production_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads serialized Scikit-Learn production pipeline from disk.

    Parameters
    ----------
    model_path : str
        Filepath to the serialized .joblib pipeline.

    Returns
    -------
    Pipeline
        Fitted Scikit-Learn pipeline.

    Raises
    ------
    FileNotFoundError
        If the model artifact has not yet been trained or does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Production model artifact not found at '{path.resolve()}'. "
            f"Please run 'python src/train.py' first to train and serialize the model."
        )
    pipeline = joblib.load(path)
    logger.info(f"Loaded production pipeline from {path.resolve()}")
    return pipeline


def assign_risk_category(
    probability: float,
    low_thresh: float = DEFAULT_LOW_THRESHOLD,
    high_thresh: float = DEFAULT_HIGH_THRESHOLD,
) -> Tuple[str, str, str]:
    """
    Maps churn probability to risk tier, UI badge hex color, and recommended business action.

    Parameters
    ----------
    probability : float
        Estimated churn probability [0.0, 1.0].
    low_thresh : float
        Upper bound for Low Risk category (default 0.30).
    high_thresh : float
        Lower bound for High Risk category (default 0.60).

    Returns
    -------
    Tuple[str, str, str]
        (risk_category, hex_color, action_recommendation)
    """
    if probability >= high_thresh:
        return (
            "High Risk",
            "#E71D36",
            "Urgent Intervention: Dedicated account manager outreach, contract renewal bonus, priority tech support.",
        )
    elif probability >= low_thresh:
        return (
            "Medium Risk",
            "#FF9F1C",
            "Targeted Nurture: Proactive satisfaction check-in, targeted service enhancement, usage advisory.",
        )
    else:
        return (
            "Low Risk",
            "#2EC4B6",
            "Standard Retention: Standard lifecycle engagement, loyalty recognition, up-sell/cross-sell evaluation.",
        )


def validate_batch_input(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that an uploaded DataFrame contains the necessary feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame uploaded by user.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of missing column names)
    """
    missing = [col for col in REQUIRED_INPUT_FEATURES if col not in df.columns]
    return (len(missing) == 0, missing)


def predict_single_customer(
    customer_data: Dict[str, Any],
    pipeline: Optional[Any] = None,
    low_thresh: float = DEFAULT_LOW_THRESHOLD,
    high_thresh: float = DEFAULT_HIGH_THRESHOLD,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Generates churn probability, risk tier, and explainability factors for a single customer.

    Parameters
    ----------
    customer_data : Dict[str, Any]
        Dictionary of customer attributes.
    pipeline : Optional[Any]
        Pre-loaded pipeline, or None to load from model_path.
    low_thresh : float
        Threshold below which customer is Low Risk.
    high_thresh : float
        Threshold above which customer is High Risk.
    model_path : str
        Path to pipeline artifact if not provided.

    Returns
    -------
    Dict[str, Any]
        Comprehensive scoring result.
    """
    if pipeline is None:
        pipeline = load_production_pipeline(model_path)

    df_single = pd.DataFrame([customer_data])
    df_cleaned = clean_telecom_data(df_single)

    proba = float(pipeline.predict_proba(df_cleaned)[0, 1])
    risk_cat, color_code, recommendation = assign_risk_category(
        proba, low_thresh=low_thresh, high_thresh=high_thresh
    )

    explanation = explain_individual_prediction(pipeline, df_cleaned)

    return {
        "churn_probability": round(proba, 4),
        "predicted_label": int(proba >= 0.50),
        "risk_category": risk_cat,
        "risk_color": color_code,
        "recommendation": recommendation,
        "primary_risk_drivers": explanation["primary_risk_drivers"],
        "primary_protective_factors": explanation["primary_protective_factors"],
        "disclaimer": explanation["disclaimer"],
    }


def predict_batch(
    df: pd.DataFrame,
    pipeline: Optional[Any] = None,
    low_thresh: float = DEFAULT_LOW_THRESHOLD,
    high_thresh: float = DEFAULT_HIGH_THRESHOLD,
    model_path: str = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """
    Generates batch churn predictions and risk categories for a DataFrame of customer records.
    Never alters the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input customer records.
    pipeline : Optional[Any]
        Pre-loaded pipeline or None.
    low_thresh : float
        Threshold for Low Risk.
    high_thresh : float
        Threshold for High Risk.
    model_path : str
        Model filepath.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame augmented with prediction columns.
    """
    is_valid, missing = validate_batch_input(df)
    if not is_valid:
        raise ValueError(
            f"Batch dataset is missing required features: {missing}. "
            f"Required columns are: {REQUIRED_INPUT_FEATURES}"
        )

    if pipeline is None:
        pipeline = load_production_pipeline(model_path)

    df_cleaned = clean_telecom_data(df)
    probas = pipeline.predict_proba(df_cleaned)[:, 1]

    result_df = df.copy()
    result_df["churn_probability"] = [round(float(p), 4) for p in probas]
    result_df["predicted_churn"] = (result_df["churn_probability"] >= 0.50).astype(int)

    risk_cats = []
    actions = []
    for p in result_df["churn_probability"]:
        cat, _, action = assign_risk_category(p, low_thresh=low_thresh, high_thresh=high_thresh)
        risk_cats.append(cat)
        actions.append(action.split(":")[0])

    result_df["risk_category"] = risk_cats
    result_df["retention_priority"] = actions

    logger.info(
        f"Batch prediction complete for {len(result_df)} records. "
        f"High Risk: {(result_df['risk_category'] == 'High Risk').sum()}, "
        f"Medium Risk: {(result_df['risk_category'] == 'Medium Risk').sum()}, "
        f"Low Risk: {(result_df['risk_category'] == 'Low Risk').sum()}."
    )

    return result_df
