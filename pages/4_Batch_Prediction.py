"""
Batch Prediction Page.

Enables bulk CSV ingestion, schema validation, multi-customer scoring,
risk tier segmentation, and enriched CSV export.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_utils import (
    apply_custom_css,
    render_metric_card,
    format_plotly_figure,
    get_cached_model_pipeline,
)
from src.predict import (
    predict_batch,
    validate_batch_input,
    REQUIRED_INPUT_FEATURES,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
)
from src.data_loader import generate_synthetic_fallback

st.set_page_config(page_title="Batch Prediction - ChurnIntel", page_icon="📂", layout="wide")
apply_custom_css()

st.title("📂 Batch Churn Prediction")
st.markdown(
    "Upload a CSV batch of customer profiles to generate churn probability scores, "
    "assign retention urgency tiers, and export enriched action lists."
)

# Load pipeline
try:
    pipeline = get_cached_model_pipeline()
except Exception as e:
    st.error(f"⚠️ Production pipeline not found: {e}. Please run `python src/train.py`.")
    st.stop()

# Threshold controls in sidebar
st.sidebar.header("⚙️ Batch Risk Cutoffs")
low_thresh = st.sidebar.slider("Low Risk Cap", 0.10, 0.45, DEFAULT_LOW_THRESHOLD, 0.05)
high_thresh = st.sidebar.slider("High Risk Floor", 0.50, 0.85, DEFAULT_HIGH_THRESHOLD, 0.05)

st.markdown('<div class="section-header">DATA INGESTION & TEMPLATE</div>', unsafe_allow_html=True)

# Provide sample template download
col_info, col_download = st.columns([2, 1])
with col_info:
    st.markdown(
        "Upload a `.csv` file matching the Telco schema. You can also download a verified sample "
        "batch file to test the scoring engine instantly."
    )
with col_download:
    sample_df = generate_synthetic_fallback(n_samples=25, seed=101)
    sample_csv = sample_df.drop(columns=["Churn"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Sample Batch CSV",
        data=sample_csv,
        file_name="sample_batch_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )

uploaded_file = st.file_uploader(
    "Upload Customer CSV File (Max 50MB):",
    type=["csv"],
    help="CSV must include required demographic, contract, and service feature columns.",
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully parsed **{len(input_df):,}** customer records from `{uploaded_file.name}`.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Validate Schema
    is_valid, missing_cols = validate_batch_input(input_df)
    if not is_valid:
        st.error(
            f"❌ **Schema Validation Failed!** The uploaded file is missing {len(missing_cols)} required feature column(s):"
        )
        st.code(f"Missing columns: {missing_cols}")
        st.info(f"Required columns ({len(REQUIRED_INPUT_FEATURES)}): {', '.join(REQUIRED_INPUT_FEATURES)}")
        st.stop()

    # Run Prediction
    with st.spinner("Scoring customer cohort through production pipeline..."):
        try:
            scored_df = predict_batch(
                input_df,
                pipeline=pipeline,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
            )
        except Exception as e:
            st.error(f"Inference error during batch execution: {e}")
            st.stop()

    # Risk Summary Metrics
    st.markdown('<div class="section-header">BATCH RISK STRATIFICATION</div>', unsafe_allow_html=True)

    high_risk_count = int((scored_df["risk_category"] == "High Risk").sum())
    med_risk_count = int((scored_df["risk_category"] == "Medium Risk").sum())
    low_risk_count = int((scored_df["risk_category"] == "Low Risk").sum())
    total_scored = len(scored_df)
    avg_churn_risk = float(scored_df["churn_probability"].mean())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            label="Total Customers Scored",
            value=f"{total_scored:,}",
            subtext="Complete Batch Processed",
        )
    with m2:
        render_metric_card(
            label="Average Churn Risk",
            value=f"{avg_churn_risk:.1%}",
            subtext="Cohort Mean Probability",
        )
    with m3:
        render_metric_card(
            label="High Risk (Urgent)",
            value=f"{high_risk_count:,}",
            subtext=f"{(high_risk_count / total_scored):.1%} of batch",
        )
    with m4:
        render_metric_card(
            label="Low Risk (Stable)",
            value=f"{low_risk_count:,}",
            subtext=f"{(low_risk_count / total_scored):.1%} of batch",
        )

    # Risk Distribution Donut & Retention Priority Breakdown
    c_chart1, c_chart2 = st.columns([1, 1])

    with c_chart1:
        risk_summary = scored_df["risk_category"].value_counts().reset_index()
        risk_summary.columns = ["Risk Tier", "Customers"]
        fig_risk = px.pie(
            risk_summary,
            names="Risk Tier",
            values="Customers",
            hole=0.55,
            color="Risk Tier",
            color_discrete_map={
                "High Risk": "#E71D36",
                "Medium Risk": "#FF9F1C",
                "Low Risk": "#2EC4B6",
            },
        )
        format_plotly_figure(fig_risk, title="Risk Tier Segmentation")
        st.plotly_chart(fig_risk, use_container_width=True)

    with c_chart2:
        fig_prob_dist = px.histogram(
            scored_df,
            x="churn_probability",
            nbins=20,
            color_discrete_sequence=["#00B4D8"],
            labels={"churn_probability": "Predicted Churn Probability", "count": "Customers"},
        )
        format_plotly_figure(fig_prob_dist, title="Predicted Probability Distribution")
        st.plotly_chart(fig_prob_dist, use_container_width=True)

    # Filterable Enriched Results Table
    st.markdown('<div class="section-header">ENRICHED RETENTION ACTION LIST</div>', unsafe_allow_html=True)

    filter_risk_tier = st.multiselect(
        "Filter by Risk Tier:",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"],
    )

    filtered_scored_df = scored_df[scored_df["risk_category"].isin(filter_risk_tier)]

    # Style display
    st.dataframe(
        filtered_scored_df,
        use_container_width=True,
        height=380,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Probability",
                help="Predicted probability of churning",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "risk_category": st.column_config.TextColumn("Risk Tier"),
            "retention_priority": st.column_config.TextColumn("Action Protocol"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
        },
    )

    # Export Enriched CSV
    enriched_csv = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Scored Batch Results as CSV ({len(scored_df):,} rows)",
        data=enriched_csv,
        file_name="churn_scored_customer_batch.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Please upload a customer CSV file or download the sample template above to begin batch scoring.")
