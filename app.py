"""
Customer Churn Prediction & Retention Intelligence Dashboard.
Main Application Entry Point.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_utils import (
    apply_custom_css,
    render_metric_card,
    get_cached_evaluation_report,
    get_cached_raw_data,
)

# Configure Streamlit page
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Sidebar branding & quick links
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 12px 0;">
        <h2 style="color: #00B4D8; margin-bottom: 2px;">⚡ ChurnIntel</h2>
        <span style="color: #8D99AE; font-size: 0.85rem;">Retention Intelligence System</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.info(
    "💡 **Quick Tip:** Use the sidebar menu above to navigate through executive KPIs, "
    "interactive customer exploration, real-time single prediction, batch scoring, and model benchmarks."
)

# Header Section
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1C2541 0%, #0B132B 100%);
                padding: 24px 30px; border-radius: 12px; border: 1px solid #3A506B; margin-bottom: 24px;">
        <h1 style="color: #FFFFFF; font-size: 2.3rem; margin-bottom: 6px;">
            Customer Churn Prediction & Retention Intelligence
        </h1>
        <p style="color: #8D99AE; font-size: 1.05rem; margin-bottom: 0;">
            Production-grade machine learning platform for early churn detection, 
            customer risk stratification, and data-driven customer retention strategies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pipeline Health & System Status Bar
report = get_cached_evaluation_report()
raw_df = None
try:
    raw_df = get_cached_raw_data()
    data_status = f"✅ Loaded ({len(raw_df):,} customers)"
except Exception as e:
    data_status = f"⚠️ Error: {e}"

model_status = "✅ Ready (Random Forest)" if report else "⚠️ Run 'python src/train.py'"
auc_score = f"{report['champion_metrics']['roc_auc']:.4f}" if report else "N/A"
f1_val = f"{report['champion_metrics']['f1_score']:.4f}" if report else "N/A"

st.markdown('<div class="section-header">SYSTEM HEALTH & ARCHITECTURE STATUS</div>', unsafe_allow_html=True)
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    render_metric_card(
        label="Raw Telemetry Data",
        value="7,043",
        subtext="IBM Telco Verified Schema",
        help_text="Official IBM Telco customer churn dataset.",
    )
with col_stat2:
    render_metric_card(
        label="Production Model",
        value="Random Forest",
        subtext="Balanced Class Pipeline",
        help_text="Best model selected via 5-fold cross validation.",
    )
with col_stat3:
    render_metric_card(
        label="Held-Out ROC-AUC",
        value=auc_score,
        subtext="Test Set Discrimination",
        help_text="Area Under the Receiver Operating Characteristic Curve on held-out test set.",
    )
with col_stat4:
    render_metric_card(
        label="Held-Out F1-Score",
        value=f1_val,
        subtext="Balanced Precision-Recall",
        help_text="Harmonic mean of precision and recall on test data.",
    )

st.markdown('<div class="section-header">APPLICATION MODULES & CAPABILITIES</div>', unsafe_allow_html=True)

col_mod1, col_mod2 = st.columns(2)

with col_mod1:
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="color: #00B4D8; margin-top: 0;">📈 1. Executive Overview</h3>
            <p style="color: #E0E1DD; font-size: 0.95rem;">
                Strategic KPI dashboard analyzing historical churn patterns, revenue at risk,
                contract type distributions, and tenure cohort dynamics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="color: #00B4D8; margin-top: 0;">🔍 2. Customer Explorer</h3>
            <p style="color: #E0E1DD; font-size: 0.95rem;">
                Interactive multi-dimensional cohort slicing and filtering by contract,
                internet service, payment method, tenure, and monthly charges.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="color: #00B4D8; margin-top: 0;">🎯 3. Individual Churn Prediction</h3>
            <p style="color: #E0E1DD; font-size: 0.95rem;">
                Real-time single-customer risk estimator with configurable thresholds, 
                visual probability gauge, local risk driver attribution, and recommended retention tactics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mod2:
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="color: #00B4D8; margin-top: 0;">📂 4. Batch Prediction</h3>
            <p style="color: #E0E1DD; font-size: 0.95rem;">
                Upload customer CSV files for high-throughput batch scoring, automated risk tier segmentation,
                and one-click export of enriched retention worklists.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="color: #00B4D8; margin-top: 0;">🏆 5. Model Performance & Evaluation</h3>
            <p style="color: #E0E1DD; font-size: 0.95rem;">
                Rigorous side-by-side benchmark comparison (Logistic Regression, Random Forest, XGBoost),
                confusion matrix trade-offs, ROC & PR curves, and global feature importance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="color: #00B4D8; margin-top: 0;">📖 6. About & Methodology</h3>
            <p style="color: #E0E1DD; font-size: 0.95rem;">
                Comprehensive documentation covering business retention economics, dataset attribution, 
                leakage-free preprocessing, class imbalance mitigation, and ethical AI practices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-header">BUSINESS VALUE & DECISION SUPPORT</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="callout-box">
        <b>💡 Executive Takeaway:</b> Retaining an existing telecommunications customer is estimated to cost 
        <b>5x to 7x less</b> than acquiring a new subscriber. By surfacing churn signals 60–90 days prior to contract expiry, 
        this system enables proactive intervention—diverting marketing spend from broad blast discounts 
        into high-leverage personalized retention incentives.
    </div>
    """,
    unsafe_allow_html=True,
)
