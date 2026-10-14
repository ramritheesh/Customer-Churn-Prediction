"""
Shared UI Utilities and Styling for Streamlit Dashboard.

Provides custom CSS, dark navy theme formatting, reusable KPI card components,
and cached data/model loaders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_raw_data
from src.preprocessing import clean_telecom_data
from src.predict import load_production_pipeline, DEFAULT_MODEL_PATH

DARK_THEME_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #0B132B;
        color: #E0E1DD;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1C2541;
        border: 1px solid #3A506B;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00B4D8;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8D99AE;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.82rem;
        color: #48CAE4;
        margin-top: 6px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-high {
        background-color: rgba(231, 29, 54, 0.2);
        color: #FF5A6E;
        border: 1px solid #E71D36;
    }
    .badge-medium {
        background-color: rgba(255, 159, 28, 0.2);
        color: #FFB347;
        border: 1px solid #FF9F1C;
    }
    .badge-low {
        background-color: rgba(46, 196, 182, 0.2);
        color: #2EC4B6;
        border: 1px solid #2EC4B6;
    }

    /* Section divider */
    .section-header {
        border-bottom: 2px solid #3A506B;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 18px;
        color: #48CAE4;
        font-weight: 600;
    }
    
    /* Alert / Callout box */
    .callout-box {
        background-color: #1C2541;
        border-left: 4px solid #00B4D8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 18px;
        color: #E0E1DD;
    }
</style>
"""


def apply_custom_css():
    """Injects custom dark navy styling into the Streamlit session."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    help_text: Optional[str] = None,
):
    """
    Renders a custom HTML metric card matching the dashboard aesthetic.
    """
    help_attr = f'title="{help_text}"' if help_text else ""
    html = f"""
    <div class="metric-card" {help_attr}>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_plotly_layout_defaults() -> Dict[str, Any]:
    """Returns standardized layout configuration for Plotly charts."""
    return dict(
        paper_bgcolor="#1C2541",
        plot_bgcolor="#1C2541",
        font=dict(color="#E0E1DD", family="sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor="#3A506B",
            zerolinecolor="#3A506B",
            linecolor="#3A506B",
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="#3A506B",
            zerolinecolor="#3A506B",
            linecolor="#3A506B",
            showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(28, 37, 65, 0.8)",
            bordercolor="#3A506B",
            borderwidth=1,
        ),
    )


def format_plotly_figure(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies the dark navy theme to a Plotly figure."""
    layout_defaults = get_plotly_layout_defaults()
    if title:
        layout_defaults["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#FFFFFF"),
            x=0.02,
            y=0.96,
        )
    fig.update_layout(**layout_defaults)
    return fig


@st.cache_data(show_spinner=False)
def get_cached_raw_data(data_path: str = "data/raw/Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Loads and caches the raw dataset in memory."""
    return load_raw_data(data_path)


@st.cache_data(show_spinner=False)
def get_cached_cleaned_data(data_path: str = "data/raw/Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Loads, cleans, and caches the customer dataset."""
    df_raw = load_raw_data(data_path)
    return clean_telecom_data(df_raw)


@st.cache_resource(show_spinner=False)
def get_cached_model_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """Loads and caches the production ML pipeline."""
    return load_production_pipeline(model_path)


@st.cache_data(show_spinner=False)
def get_cached_evaluation_report(
    reports_path: str = "reports/model_metrics.json",
) -> Optional[Dict[str, Any]]:
    """Loads and caches the JSON evaluation benchmark report."""
    path = Path(reports_path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)
