"""
Customer Explorer Page.

Interactive cohort slicing, filtering, and customer telemetry inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_utils import (
    apply_custom_css,
    render_metric_card,
    format_plotly_figure,
    get_cached_cleaned_data,
)

st.set_page_config(page_title="Customer Explorer - ChurnIntel", page_icon="🔍", layout="wide")
apply_custom_css()

st.title("🔍 Customer Explorer")
st.markdown("Filter and inspect individual customer cohorts across contract types, payment channels, and service subscriptions.")

df = get_cached_cleaned_data()

# Sidebar Filters
st.sidebar.header("🎯 Filter Cohorts")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=sorted(df["Contract"].unique().tolist()),
    default=sorted(df["Contract"].unique().tolist()),
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=sorted(df["InternetService"].unique().tolist()),
    default=sorted(df["InternetService"].unique().tolist()),
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    options=sorted(df["PaymentMethod"].unique().tolist()),
    default=sorted(df["PaymentMethod"].unique().tolist()),
)

min_tenure, max_tenure = int(df["tenure"].min()), int(df["tenure"].max())
tenure_range = st.sidebar.slider(
    "Tenure Range (Months)",
    min_value=min_tenure,
    max_value=max_tenure,
    value=(min_tenure, max_tenure),
)

min_monthly, max_monthly = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
monthly_range = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=float(min_monthly),
    max_value=float(max_monthly),
    value=(float(min_monthly), float(max_monthly)),
)

churn_filter = st.sidebar.selectbox(
    "Churn Status",
    options=["All Customers", "Churned Only (Yes)", "Retained Only (No)"],
    index=0,
)

# Apply filters
filtered_df = df.copy()

if contract_filter:
    filtered_df = filtered_df[filtered_df["Contract"].isin(contract_filter)]

if internet_filter:
    filtered_df = filtered_df[filtered_df["InternetService"].isin(internet_filter)]

if payment_filter:
    filtered_df = filtered_df[filtered_df["PaymentMethod"].isin(payment_filter)]

filtered_df = filtered_df[
    (filtered_df["tenure"] >= tenure_range[0]) & (filtered_df["tenure"] <= tenure_range[1])
]

filtered_df = filtered_df[
    (filtered_df["MonthlyCharges"] >= monthly_range[0])
    & (filtered_df["MonthlyCharges"] <= monthly_range[1])
]

if churn_filter == "Churned Only (Yes)":
    filtered_df = filtered_df[filtered_df["Churn"] == "Yes"]
elif churn_filter == "Retained Only (No)":
    filtered_df = filtered_df[filtered_df["Churn"] == "No"]

# Cohort Summary KPIs
col1, col2, col3, col4 = st.columns(4)

filtered_count = len(filtered_df)
total_count = len(df)
pct_of_total = (filtered_count / total_count) * 100 if total_count > 0 else 0

churn_rate_cohort = (
    (filtered_df["Churn"] == "Yes").mean() * 100 if filtered_count > 0 else 0.0
)
avg_monthly = filtered_df["MonthlyCharges"].mean() if filtered_count > 0 else 0.0
avg_tenure = filtered_df["tenure"].mean() if filtered_count > 0 else 0.0

with col1:
    render_metric_card(
        label="Matched Customers",
        value=f"{filtered_count:,}",
        subtext=f"{pct_of_total:.1f}% of total base",
    )
with col2:
    render_metric_card(
        label="Cohort Churn Rate",
        value=f"{churn_rate_cohort:.1f}%",
        subtext=f"{(filtered_df['Churn'] == 'Yes').sum():,} Churned",
    )
with col3:
    render_metric_card(
        label="Avg Monthly Charges",
        value=f"${avg_monthly:.2f}",
        subtext=f"Total: ${filtered_df['MonthlyCharges'].sum():,.0f}",
    )
with col4:
    render_metric_card(
        label="Avg Account Tenure",
        value=f"{avg_tenure:.1f} mo",
        subtext="Customer lifecycle depth",
    )

# Summary visual for filtered cohort
st.markdown('<div class="section-header">COHORT BEHAVIOR BREAKDOWN</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    if filtered_count > 0:
        fig_sub = px.box(
            filtered_df,
            x="Contract",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"No": "#2EC4B6", "Yes": "#E71D36"},
            labels={"MonthlyCharges": "Monthly Charges ($)"},
        )
        format_plotly_figure(fig_sub, title="Charges Distribution by Contract & Churn")
        st.plotly_chart(fig_sub, use_container_width=True)
    else:
        st.info("No records match the current filter selection.")

with c2:
    if filtered_count > 0:
        payment_churn = (
            filtered_df.groupby("PaymentMethod")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        payment_churn.columns = ["Payment Method", "Churn Rate (%)"]
        fig_pay = px.bar(
            payment_churn,
            x="Churn Rate (%)",
            y="Payment Method",
            orientation="h",
            color="Churn Rate (%)",
            color_continuous_scale=["#2EC4B6", "#FF9F1C", "#E71D36"],
            text=payment_churn["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_pay.update_traces(textposition="outside")
        fig_pay.update_layout(coloraxis_showscale=False, xaxis=dict(range=[0, 65]))
        format_plotly_figure(fig_pay, title="Churn Propensity by Payment Channel")
        st.plotly_chart(fig_pay, use_container_width=True)
    else:
        st.info("No records match the current filter selection.")

# Interactive Searchable Table
st.markdown('<div class="section-header">FILTERED SUBSCRIBER RECORDS</div>', unsafe_allow_html=True)

search_id = st.text_input("🔍 Search by Customer ID (optional):", "").strip()
display_df = filtered_df
if search_id:
    display_df = display_df[display_df["customerID"].str.contains(search_id, case=False, na=False)]

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
    column_config={
        "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
        "TotalCharges": st.column_config.NumberColumn("Total Charges", format="$%.2f"),
        "tenure": st.column_config.NumberColumn("Tenure (Months)"),
        "Churn": st.column_config.TextColumn("Churn Status"),
    },
)

# Export button
csv_data = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"📥 Download Filtered Cohort as CSV ({len(display_df):,} rows)",
    data=csv_data,
    file_name="telecom_customer_cohort.csv",
    mime="text/csv",
)
