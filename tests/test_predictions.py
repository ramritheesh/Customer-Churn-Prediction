"""
Unit tests for the prediction engine, risk categorization, and batch inference.
"""

import pytest
import pandas as pd
from src.predict import (
    load_production_pipeline,
    assign_risk_category,
    predict_single_customer,
    predict_batch,
    DEFAULT_MODEL_PATH,
)
from src.data_loader import generate_synthetic_fallback


@pytest.fixture(scope="module")
def sample_customer():
    """Returns a realistic single customer profile."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00,
    }


def test_load_production_pipeline():
    """Verify loading production pipeline succeeds and raises error on missing file."""
    pipeline = load_production_pipeline(DEFAULT_MODEL_PATH)
    assert pipeline is not None
    assert hasattr(pipeline, "predict_proba")

    with pytest.raises(FileNotFoundError):
        load_production_pipeline("models/non_existent_model.joblib")


def test_assign_risk_category():
    """Verify risk categorization thresholds and action recommendations."""
    cat_low, col_low, act_low = assign_risk_category(0.15)
    assert cat_low == "Low Risk"
    assert col_low == "#2EC4B6"
    assert "Standard Retention" in act_low

    cat_med, col_med, act_med = assign_risk_category(0.45)
    assert cat_med == "Medium Risk"
    assert col_med == "#FF9F1C"
    assert "Targeted Nurture" in act_med

    cat_high, col_high, act_high = assign_risk_category(0.85)
    assert cat_high == "High Risk"
    assert col_high == "#E71D36"
    assert "Urgent Intervention" in act_high


def test_predict_single_customer(sample_customer):
    """Verify single customer prediction output structure, ranges, and explanation."""
    result = predict_single_customer(sample_customer)
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["predicted_label"] in [0, 1]
    assert result["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert isinstance(result["primary_risk_drivers"], list)
    assert len(result["primary_risk_drivers"]) > 0
    assert "disclaimer" in result


def test_predict_batch_success():
    """Verify batch prediction augments DataFrame and keeps original intact."""
    synth_df = generate_synthetic_fallback(n_samples=25, seed=99)
    synth_copy = synth_df.copy()

    enriched_df = predict_batch(synth_df)

    # Verify input DataFrame was not mutated
    pd.testing.assert_frame_equal(synth_df, synth_copy)

    # Verify new prediction columns
    assert "churn_probability" in enriched_df.columns
    assert "predicted_churn" in enriched_df.columns
    assert "risk_category" in enriched_df.columns
    assert "retention_priority" in enriched_df.columns
    assert len(enriched_df) == 25
    assert enriched_df["churn_probability"].between(0.0, 1.0).all()


def test_predict_batch_missing_features():
    """Verify batch prediction raises informative ValueError on missing required columns."""
    df_invalid = pd.DataFrame({"MonthlyCharges": [50.0], "tenure": [12]})
    with pytest.raises(ValueError, match="Batch dataset is missing required features"):
        predict_batch(df_invalid)


def test_handling_unseen_categorical_levels(sample_customer):
    """Verify pipeline handles completely unseen categorical levels without crashing."""
    corrupted_customer = sample_customer.copy()
    corrupted_customer["Contract"] = "Decade-to-Decade"  # Unseen category
    corrupted_customer["PaymentMethod"] = "Cryptocurrency"  # Unseen category

    result = predict_single_customer(corrupted_customer)
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
