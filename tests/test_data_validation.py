"""
Unit tests for data loading and raw data schema validation.
"""

import pandas as pd
from src.data_loader import (
    load_raw_data,
    validate_raw_data,
    generate_synthetic_fallback,
    EXPECTED_COLUMNS,
)


def test_load_raw_data_success():
    """Verify loading raw dataset returns correct structure and column headers."""
    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043
    assert all(col in df.columns for col in EXPECTED_COLUMNS)


def test_validate_raw_data_valid_dataset():
    """Verify validate_raw_data passes for clean IBM Telco dataset."""
    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    result = validate_raw_data(df)
    assert result["is_valid"] is True
    assert result["row_count"] == 7043
    assert result["column_count"] == 21
    assert result["customer_id_duplicates"] == 0
    assert result["blank_total_charges_count"] == 11
    assert 0.20 < result["historical_churn_rate"] < 0.35


def test_validate_raw_data_missing_column():
    """Verify validation flags missing required columns."""
    df = generate_synthetic_fallback(n_samples=50)
    df_missing = df.drop(columns=["MonthlyCharges", "Churn"])
    result = validate_raw_data(df_missing)
    assert result["is_valid"] is False
    assert "MonthlyCharges" in result["missing_expected_columns"]
    assert "Churn" in result["missing_expected_columns"]


def test_generate_synthetic_fallback_schema():
    """Verify synthetic fallback generator satisfies Telco schema exactly."""
    df_synth = generate_synthetic_fallback(n_samples=75, seed=123)
    assert len(df_synth) == 75
    assert set(df_synth.columns) == set(EXPECTED_COLUMNS)
    val = validate_raw_data(df_synth)
    assert val["is_valid"] is True
