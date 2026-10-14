"""
About and Methodology Page.

Comprehensive documentation of dataset provenance, preprocessing architecture,
class imbalance handling, precision-recall trade-offs, and ethical AI considerations.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_utils import apply_custom_css

st.set_page_config(page_title="About & Methodology - ChurnIntel", page_icon="📖", layout="wide")
apply_custom_css()

st.title("📖 About & Technical Methodology")
st.markdown("Engineering documentation, statistical decisions, pipeline architecture, and ethical safeguards.")

# Section 1: Business Problem Statement
st.markdown('<div class="section-header">1. BUSINESS PROBLEM & RETENTION ECONOMICS</div>', unsafe_allow_html=True)
st.markdown(
    """
    Customer attrition (churn) directly impacts the financial sustainability of subscription-based businesses. 
    In telecommunications:
    - **Customer Acquisition Cost (CAC):** Acquiring a new telecom subscriber is 5x to 7x more expensive than retaining an existing account.
    - **Customer Lifetime Value (LTV):** High early churn depresses lifetime value and delays capital expenditure payback for fiber optic infrastructure.
    - **Asymmetric Cost of Errors:**
      - A **False Negative (FN)** allows an at-risk customer to churn silently, incurring hundreds of dollars in lost future recurring revenue and marketing replacement costs.
      - A **False Positive (FP)** merely triggers an outreach campaign or a nominal loyalty incentive for a satisfied customer.
      - Consequently, optimization must prioritize **ROC-AUC and Recall** over raw overall accuracy.
    """
)

# Section 2: Dataset Provenance & Attribution
st.markdown('<div class="section-header">2. DATASET ATTRIBUTION & SCHEMA</div>', unsafe_allow_html=True)
st.markdown(
    """
    - **Dataset Name:** IBM Telco Customer Churn
    - **Original Publisher:** IBM Analytics Community & Kaggle Open Datasets
    - **License:** Apache 2.0 / CC0 Public Domain
    - **Sample Size:** 7,043 customer accounts, 21 attributes (19 predictive features + `customerID` + `Churn` target)
    - **Target Distribution:** 1,869 Churned (26.54%) vs. 5,174 Retained (73.46%). Moderate class imbalance (~2.77:1).
    - **Known Data Anomaly Handled:** 11 customer records where `tenure == 0` had blank whitespace strings (`" "`) in `TotalCharges`. 
      These were coerced to `0.0` to reflect zero cumulative billing for new accounts, preventing silent dropouts.
    """
)

# Section 3: Feature Engineering & Preprocessing Architecture
st.markdown('<div class="section-header">3. PREPROCESSING & LEAKAGE PREVENTION</div>', unsafe_allow_html=True)
st.markdown(
    """
    To ensure production reliability and prevent **Data Leakage**:
    1. **Strict Partitioning:** A stratified 80/20 train-test split (`random_state=42`) is executed before any statistical estimators are computed.
    2. **Encapsulated `ColumnTransformer`:**
       - **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`. 
         Fitted strictly on `X_train` and applied to `X_test` and inference payloads.
       - **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`. 
         The `handle_unknown='ignore'` setting ensures the production app gracefully handles previously unseen categorical values at inference time.
    3. **Identifier Isolation:** `customerID` is excluded from feature sets to prevent artificial memorization and overfitting.
    """
)

# Section 4: Model Training & Benchmark Procedure
st.markdown('<div class="section-header">4. MODEL BENCHMARKING & SELECTION</div>', unsafe_allow_html=True)
st.markdown(
    """
    Three distinct algorithmic paradigms were benchmarked under identical stratified validation protocols:
    1. **Logistic Regression (Linear Baseline):** `class_weight='balanced'`, L2 regularization, `max_iter=1000`. Fast, interpretable benchmark.
    2. **Random Forest Classifier (Ensemble Bagging):** 100 estimators, `max_depth=8`, `min_samples_split=10`, `class_weight='balanced'`. 
       Captures non-linear feature interactions without prone overfitting.
    3. **XGBoost Classifier (Gradient Boosting):** 100 estimators, `max_depth=4`, `learning_rate=0.08`, with `scale_pos_weight=2.77` to offset minority class imbalance.
    
    **Validation Protocol:** 5-Fold Stratified Cross-Validation on training data + Final evaluation on untouched held-out test data (N=1,409). 
    The champion pipeline is saved via `joblib` as a single self-contained artifact for reproducible serving.
    """
)

# Section 5: Model Explainability
st.markdown('<div class="section-header">5. EXPLAINABILITY & TRANSPARENCY</div>', unsafe_allow_html=True)
st.markdown(
    """
    - **Global Explainability:** Evaluated via **Permutation Importance** on the held-out test set, measuring the exact drop in ROC-AUC when each feature is permuted. 
      This avoids tree-depth bias associated with raw impurity importance.
    - **Local Explainability:** Single customer predictions evaluate key risk and protective drivers based on domain-grounded attribution heuristics, 
      explaining to business users why a specific customer was categorized as High Risk.
    - **Association vs. Causation:** The system explicitly states that identified factors represent statistical associations rather than direct causal drivers.
    """
)

# Section 6: Limitations & Ethical Considerations
st.markdown('<div class="section-header">6. LIMITATIONS & RESPONSIBLE AI</div>', unsafe_allow_html=True)
st.markdown(
    """
    - **Decision-Support Boundary:** Model outputs are probability estimates designed to inform human account managers—not autonomous triggers to cancel accounts or penalize customers.
    - **Fairness & Demographic Variables:** Demographic attributes such as `gender` and `SeniorCitizen` are included in the dataset schema. 
      Retention policies should avoid discriminatory pricing or differential service degradation based on demographic factors.
    - **Concept Drift:** Customer behavior shifts over time as competitive pricing, economic conditions, and service plans evolve. 
      The model should be retrained periodically with newly observed telemetry data.
    """
)
