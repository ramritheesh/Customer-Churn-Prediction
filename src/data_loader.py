"""
Data Ingestion and Validation Module for Customer Churn Prediction.

Source Dataset Attribution:
- Dataset: IBM Telco Customer Churn
- Source: IBM Analytics / Community / Kaggle Public Dataset
- License: Apache 2.0 / CC0 Public Domain
- Description: Customer account, demographic, and subscription attributes
  from a fictional telecommunications company in California in Q3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List
import urllib.request
import pandas as pd
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Official IBM Telco Churn repository raw URL
OFFICIAL_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

# Expected raw column schema
EXPECTED_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

NUMERICAL_COLUMNS: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS: List[str] = [
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
TARGET_COLUMN = "Churn"


def fetch_dataset_if_missing(
    destination_path: str = "data/raw/Telco-Customer-Churn.csv",
    source_url: str = OFFICIAL_DATASET_URL,
) -> Path:
    """
    Ensures the raw dataset exists locally; downloads it from the official IBM URL if missing.

    Parameters
    ----------
    destination_path : str
        Local file path where the raw CSV should reside.
    source_url : str
        URL to download the dataset from if not found.

    Returns
    -------
    Path
        Path object pointing to the confirmed raw dataset file.
    """
    dest = Path(destination_path)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Dataset already exists at {dest.resolve()} ({dest.stat().st_size} bytes)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading IBM Telco dataset from {source_url} to {dest}...")
    try:
        urllib.request.urlretrieve(source_url, dest)
        logger.info(f"Successfully downloaded dataset to {dest.resolve()}")
    except Exception as exc:
        logger.error(f"Failed to download dataset from {source_url}: {exc}")
        raise RuntimeError(
            f"Could not download dataset from {source_url}. Ensure network connectivity "
            f"or manually place 'Telco-Customer-Churn.csv' in '{dest.parent}'."
        ) from exc
    return dest


def load_raw_data(data_path: str = "data/raw/Telco-Customer-Churn.csv") -> pd.DataFrame:
    """
    Loads the raw Telco customer churn dataset into a pandas DataFrame.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from disk.
    """
    path = Path(data_path)
    if not path.exists():
        path = fetch_dataset_if_missing(destination_path=data_path)

    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects and validates raw dataset structure, column types, duplicates, and missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    Dict[str, Any]
        Dictionary detailing schema validity, anomalies, and summary statistics.
    """
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    has_target = TARGET_COLUMN in df.columns

    duplicates = int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else 0
    row_count = len(df)

    # Check for empty strings in TotalCharges (known issue in IBM dataset for tenure=0)
    blank_total_charges = 0
    if "TotalCharges" in df.columns:
        blank_total_charges = int((df["TotalCharges"].astype(str).str.strip() == "").sum())

    # Churn target distribution
    churn_dist: Dict[str, int] = {}
    churn_rate: float = 0.0
    if has_target:
        churn_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        churn_dist = {str(k): int(v) for k, v in churn_counts.items()}
        yes_count = churn_dist.get("Yes", 0)
        churn_rate = round(float(yes_count / row_count), 4) if row_count > 0 else 0.0

    validation_result: Dict[str, Any] = {
        "is_valid": len(missing_cols) == 0 and has_target,
        "row_count": row_count,
        "column_count": len(df.columns),
        "missing_expected_columns": missing_cols,
        "customer_id_duplicates": duplicates,
        "blank_total_charges_count": blank_total_charges,
        "target_distribution": churn_dist,
        "historical_churn_rate": churn_rate,
        "null_counts_by_col": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
    }

    if not validation_result["is_valid"]:
        logger.warning(f"Data validation failed: missing columns {missing_cols}")
    else:
        logger.info(
            f"Data validation passed: {row_count} rows, churn rate: {churn_rate:.2%}, "
            f"blank TotalCharges: {blank_total_charges}"
        )

    return validation_result


def generate_synthetic_fallback(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic mock dataset that strictly mirrors the IBM Telco schema
    for isolated unit testing and CI pipelines when no external data file is provided.
    Explicitly labeled as synthetic in logging.

    Parameters
    ----------
    n_samples : int
        Number of synthetic customer records to produce.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic Telco customer DataFrame.
    """
    logger.warning("Generating SYNTHETIC mock fallback dataset for testing/CI purposes.")
    rng = np.random.default_rng(seed)

    contracts = ["Month-to-month", "One year", "Two year"]
    internet_services = ["DSL", "Fiber optic", "No"]
    payment_methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    records: List[Dict[str, Any]] = []
    for i in range(n_samples):
        tenure = int(rng.integers(0, 73))
        monthly = round(float(rng.uniform(18.25, 118.75)), 2)
        total = round(float(tenure * monthly), 2) if tenure > 0 else " "
        has_churned = rng.choice(["Yes", "No"], p=[0.27, 0.73])

        rec = {
            "customerID": f"SYN-{i:05d}",
            "gender": rng.choice(["Male", "Female"]),
            "SeniorCitizen": int(rng.choice([0, 1], p=[0.84, 0.16])),
            "Partner": rng.choice(["Yes", "No"]),
            "Dependents": rng.choice(["Yes", "No"]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["No phone service", "No", "Yes"]),
            "InternetService": rng.choice(internet_services),
            "OnlineSecurity": rng.choice(["No internet service", "No", "Yes"]),
            "OnlineBackup": rng.choice(["No internet service", "No", "Yes"]),
            "DeviceProtection": rng.choice(["No internet service", "No", "Yes"]),
            "TechSupport": rng.choice(["No internet service", "No", "Yes"]),
            "StreamingTV": rng.choice(["No internet service", "No", "Yes"]),
            "StreamingMovies": rng.choice(["No internet service", "No", "Yes"]),
            "Contract": rng.choice(contracts),
            "PaperlessBilling": rng.choice(["Yes", "No"]),
            "PaymentMethod": rng.choice(payment_methods),
            "MonthlyCharges": monthly,
            "TotalCharges": str(total),
            "Churn": has_churned,
        }
        records.append(rec)

    return pd.DataFrame(records)
