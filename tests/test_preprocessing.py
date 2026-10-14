"""
Unit tests for data preprocessing and leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import (
    clean_telecom_data,
    encode_target,
    create_preprocessor,
    split_and_save_data,
    get_feature_names_from_preprocessor,
    NUMERICAL_FEATURES,
)


def test_clean_telecom_data_handles_blank_charges():
    """Verify clean_telecom_data converts whitespace TotalCharges to 0.0 when tenure is 0."""
    df_raw = pd.DataFrame({
        "tenure": [0, 5],
        "MonthlyCharges": [20.0, 50.0],
        "TotalCharges": [" ", " 250.0 "],
        "SeniorCitizen": [0, 1],
    })
    cleaned = clean_telecom_data(df_raw)
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 250.0
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_encode_target_valid_and_invalid():
    """Verify target binary encoding and exception raising for unknown tokens."""
    valid_series = pd.Series(["Yes", "No", "No", "Yes"])
    encoded = encode_target(valid_series)
    assert list(encoded) == [1, 0, 0, 1]

    invalid_series = pd.Series(["Yes", "Unknown", "No"])
    with pytest.raises(ValueError, match="Unrecognized target values"):
        encode_target(invalid_series)


def test_preprocessor_transformation_and_leakage_safety():
    """Verify ColumnTransformer fits on train data, handles unknown categories, and outputs valid array."""
    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.20, random_state=42)

    preprocessor = create_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Check output shapes and absence of NaNs
    assert isinstance(X_train_trans, np.ndarray)
    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()

    # Verify feature names extraction
    feature_names = get_feature_names_from_preprocessor(preprocessor)
    assert len(feature_names) == X_train_trans.shape[1]
    assert feature_names[:3] == NUMERICAL_FEATURES


def test_stratification_ratio_preserved():
    """Verify target class ratio is preserved between train and test partitions."""
    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.20, random_state=42)

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    # Stratification ensures train and test rates match within 0.5%
    assert abs(train_churn_rate - test_churn_rate) < 0.005
