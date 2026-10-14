"""
Data Preprocessing and Feature Engineering Module.

Builds reusable scikit-learn transformers and pipelines to ensure zero data leakage
between training and inference phases.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Explicit feature definitions
NUMERICAL_FEATURES: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"


def clean_telecom_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw telecom customer dataframe:
    - Strips whitespace
    - Coerces 'TotalCharges' to float, handling blank strings (' ')
    - Maps 0 tenure customers with NaN TotalCharges to 0.0
    - Converts SeniorCitizen to object/string for consistent categorical handling

    Parameters
    ----------
    df : pd.DataFrame
        Raw or batch input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the dataframe.
    """
    cleaned = df.copy()

    # Clean TotalCharges: convert whitespace to NaN, then numeric
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )
        # If customer has 0 months tenure, TotalCharges is legitimately 0.0
        if "tenure" in cleaned.columns:
            cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"] = cleaned.loc[
                cleaned["tenure"] == 0, "TotalCharges"
            ].fillna(0.0)

    # Standardize SeniorCitizen as categorical string representation
    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(str)

    return cleaned


def encode_target(series: pd.Series) -> pd.Series:
    """
    Converts binary churn target ('Yes'/'No' or 1/0) into integer binary 1/0.

    Parameters
    ----------
    series : pd.Series
        Target column series.

    Returns
    -------
    pd.Series
        Binary integer series (1 = Churned, 0 = Retained).
    """
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype in [np.int64, np.int32, int]:
        return series

    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    encoded = series.map(mapping)
    if encoded.isnull().any():
        unknown_vals = series[encoded.isnull()].unique()
        raise ValueError(f"Unrecognized target values in Churn column: {unknown_vals}")
    return encoded.astype(int)


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a reusable ColumnTransformer for numerical scaling and categorical encoding.

    Numerical Pipeline:
        1. SimpleImputer with median strategy
        2. StandardScaler

    Categorical Pipeline:
        1. SimpleImputer with most_frequent strategy
        2. OneHotEncoder (drop='first', handle_unknown='ignore', sparse_output=False)

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer instance.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extracts features matrix X and binary target y from cleaned dataframe.
    Drops customerID and handles missing columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series]]
        Features DataFrame X, and target Series y (None if target not in df).
    """
    cleaned = clean_telecom_data(df)

    cols_to_use = [col for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if col in cleaned.columns]
    X = cleaned[cols_to_use].copy()

    y = None
    if TARGET_COLUMN in cleaned.columns:
        y = encode_target(cleaned[TARGET_COLUMN])

    return X, y


def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train-test split, preserving target class distribution.
    Saves clean train and test partitions to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned raw dataframe.
    test_size : float
        Proportion of dataset to allocate to the held-out test set (default 0.20).
    random_state : int
        Seed for reproducibility.
    output_dir : str
        Directory to store processed partitions.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    from pathlib import Path

    X, y = prepare_features_and_target(df)
    if y is None:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required for training split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    train_df.to_csv(out_path / "train.csv", index=False)

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(out_path / "test.csv", index=False)

    logger.info(
        f"Data partitioned into Train ({len(X_train)} rows) and Test ({len(X_test)} rows). "
        f"Saved to {out_path.resolve()}"
    )

    return X_train, X_test, y_train, y_test


def get_feature_names_from_preprocessor(
    fitted_preprocessor: ColumnTransformer,
) -> List[str]:
    """
    Extracts human-readable feature column names from a fitted ColumnTransformer.

    Parameters
    ----------
    fitted_preprocessor : ColumnTransformer
        A fitted ColumnTransformer object.

    Returns
    -------
    List[str]
        List of encoded feature names.
    """
    feature_names: List[str] = []

    # Numerical feature names
    feature_names.extend(NUMERICAL_FEATURES)

    # Categorical one-hot encoded feature names
    cat_pipeline = fitted_preprocessor.named_transformers_["cat"]
    onehot_encoder = cat_pipeline.named_steps["onehot"]
    cat_feature_names = onehot_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names.extend(cat_feature_names)

    return feature_names
