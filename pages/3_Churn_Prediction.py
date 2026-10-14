"""
Individual Customer Churn Prediction Page.

Allows interactive input of customer attributes, renders a real-time risk gauge,
categorizes churn probability, explains key drivers, and suggests retention actions.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_utils import (
    apply_custom_css,
    get_cached_model_pipeline,
)
from src.predict import (
    predict_single_customer,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
)

st.set_page_config(page_title="Single Prediction - ChurnIntel", page_icon="🎯", layout="wide")
apply_custom_css()

st.title("🎯 Individual Churn Prediction")
st.markdown(
    "Score any customer profile in real-time, inspect risk factors, and evaluate recommended retention interventions."
)

# Load production pipeline
try:
    pipeline = get_cached_model_pipeline()
except Exception as e:
    st.error(f"⚠️ Production model could not be loaded: {e}. Please ensure `python src/train.py` has run.")
    st.stop()

# Configurable threshold sidebar
st.sidebar.header("⚙️ Risk Thresholds")
low_thresh = st.sidebar.slider(
    "Low Risk Threshold",
    min_value=0.10,
    max_value=0.45,
    value=DEFAULT_LOW_THRESHOLD,
    step=0.05,
    help="Customers below this probability are marked Low Risk.",
)
high_thresh = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.50,
    max_value=0.85,
    value=DEFAULT_HIGH_THRESHOLD,
    step=0.05,
    help="Customers above this probability are marked High Risk (Urgent).",
)

# Preset Profiles for Quick Testing
st.markdown('<div class="section-header">QUICK PRESET PROFILES</div>', unsafe_allow_html=True)
preset_col1, preset_col2, preset_col3 = st.columns(3)

preset_mode = None
with preset_col1:
    if st.button("🚨 Load High-Risk Persona", use_container_width=True):
        preset_mode = "high_risk"
with preset_col2:
    if st.button("🛡️ Load Low-Risk Persona", use_container_width=True):
        preset_mode = "low_risk"
with preset_col3:
    if st.button("🔄 Reset Defaults", use_container_width=True):
        preset_mode = "default"

# Initialize form defaults based on preset
if preset_mode == "high_risk":
    def_tenure = 2
    def_contract = "Month-to-month"
    def_internet = "Fiber optic"
    def_payment = "Electronic check"
    def_monthly = 95.0
    def_paperless = "Yes"
    def_tech = "No"
    def_sec = "No"
    def_backup = "No"
    def_device = "No"
elif preset_mode == "low_risk":
    def_tenure = 60
    def_contract = "Two year"
    def_internet = "DSL"
    def_payment = "Bank transfer (automatic)"
    def_monthly = 45.0
    def_paperless = "No"
    def_tech = "Yes"
    def_sec = "Yes"
    def_backup = "Yes"
    def_device = "Yes"
else:
    def_tenure = 12
    def_contract = "Month-to-month"
    def_internet = "Fiber optic"
    def_payment = "Electronic check"
    def_monthly = 70.0
    def_paperless = "Yes"
    def_tech = "No"
    def_sec = "No"
    def_backup = "No"
    def_device = "No"

col_form, col_result = st.columns([1.1, 0.9])

with col_form:
    st.markdown('<div class="section-header">CUSTOMER PROFILE ATTRIBUTES</div>', unsafe_allow_html=True)
    with st.form("customer_input_form"):
        st.subheader("1. Account & Contract")
        acct_c1, acct_c2 = st.columns(2)
        with acct_c1:
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(def_contract),
            )
            tenure = st.number_input(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=def_tenure,
                step=1,
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(def_payment),
            )
        with acct_c2:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=15.0,
                max_value=130.0,
                value=float(def_monthly),
                step=1.0,
            )
            calc_total = round(float(tenure * monthly_charges), 2)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(calc_total),
                step=10.0,
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=["Yes", "No"].index(def_paperless),
            )

        st.subheader("2. Demographics")
        demo_c1, demo_c2 = st.columns(2)
        with demo_c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (Senior)" if x == 1 else "No")
        with demo_c2:
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        st.subheader("3. Subscribed Services")
        serv_c1, serv_c2 = st.columns(2)
        with serv_c1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
            internet_service = st.selectbox(
                "Internet Service Provider",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(def_internet),
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(def_tech),
            )
        with serv_c2:
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(def_sec),
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(def_backup),
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(def_device),
            )
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submit_btn = st.form_submit_button("⚡ Predict Churn Probability", use_container_width=True)

with col_result:
    st.markdown('<div class="section-header">PREDICTION & RISK DIAGNOSTICS</div>', unsafe_allow_html=True)

    customer_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Execute prediction
    prediction = predict_single_customer(
        customer_dict,
        pipeline=pipeline,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )

    proba = prediction["churn_probability"]
    risk_cat = prediction["risk_category"]
    color = prediction["risk_color"]
    recommendation = prediction["recommendation"]

    # Gauge Chart
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 38}},
            title={"text": "<b>Predicted Churn Probability</b>", "font": {"color": "#8D99AE", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#3A506B"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#1C2541",
                "borderwidth": 1,
                "bordercolor": "#3A506B",
                "steps": [
                    {"range": [0, low_thresh * 100], "color": "rgba(46, 196, 182, 0.2)"},
                    {"range": [low_thresh * 100, high_thresh * 100], "color": "rgba(255, 159, 28, 0.2)"},
                    {"range": [high_thresh * 100, 100], "color": "rgba(231, 29, 54, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.8,
                    "value": proba * 100,
                },
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="#1C2541",
        plot_bgcolor="#1C2541",
        height=260,
        margin=dict(l=25, r=25, t=35, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Risk Tier Badge Card
    badge_class = "badge-high" if risk_cat == "High Risk" else ("badge-medium" if risk_cat == "Medium Risk" else "badge-low")
    st.markdown(
        f"""
        <div style="background-color: #1C2541; border: 1px solid #3A506B; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #8D99AE; font-size: 0.9rem;">RISK CLASSIFICATION:</span>
                <span class="badge {badge_class}">{risk_cat.upper()}</span>
            </div>
            <div style="margin-top: 12px; font-size: 0.95rem; color: #E0E1DD;">
                <b>Action Protocol:</b> {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary Influential Factors
    st.subheader("Key Influencing Factors")
    if prediction["primary_risk_drivers"]:
        st.markdown("**⚠️ Risk Escalation Drivers:**")
        for driver in prediction["primary_risk_drivers"]:
            st.markdown(f"- 🔴 <span style='color: #FF5A6E;'>{driver}</span>", unsafe_allow_html=True)

    if prediction["primary_protective_factors"]:
        st.markdown("**🛡️ Protective Retention Drivers:**")
        for factor in prediction["primary_protective_factors"]:
            st.markdown(f"- 🟢 <span style='color: #2EC4B6;'>{factor}</span>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="callout-box" style="margin-top: 18px; font-size: 0.82rem; color: #8D99AE;">
            <b>Decision-Support Disclaimer:</b> {prediction['disclaimer']}
        </div>
        """,
        unsafe_allow_html=True,
    )
