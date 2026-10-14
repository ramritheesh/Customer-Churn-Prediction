"""
Model Performance & Benchmarking Page.

Presents rigorous side-by-side model comparison, confusion matrix business trade-offs,
ROC and Precision-Recall curves, and global feature attribution.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_utils import (
    apply_custom_css,
    format_plotly_figure,
    get_cached_evaluation_report,
)

st.set_page_config(page_title="Model Performance - ChurnIntel", page_icon="🏆", layout="wide")
apply_custom_css()

st.title("🏆 Model Performance & Evaluation")
st.markdown(
    "Transparent evaluation of model discrimination, error trade-offs, "
    "and benchmark metrics computed on strictly held-out test partitions (N=1,409)."
)

report = get_cached_evaluation_report()
if not report:
    st.error("⚠️ Evaluation benchmark report not found at `reports/model_metrics.json`. Please execute `python src/train.py`.")
    st.stop()

metadata = report["metadata"]
comparison_list = report["model_comparison"]
champion = report["champion_metrics"]
perm_importance = report.get("permutation_importance", {}).get("ranking", [])

# Model Comparison Table
st.markdown('<div class="section-header">BENCHMARK COMPARISON ACROSS ALGORITHMS</div>', unsafe_allow_html=True)

comp_rows = []
for m in comparison_list:
    comp_rows.append({
        "Model": m["model_name"],
        "ROC-AUC": f"{m['roc_auc']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "F1-Score": f"{m['f1_score']:.4f}",
        "Accuracy": f"{m['accuracy']:.4f}",
        "5-Fold CV AUC": f"{m.get('cv_roc_auc_mean', 0.0):.4f} (±{m.get('cv_roc_auc_std', 0.0):.3f})",
        "5-Fold CV F1": f"{m.get('cv_f1_mean', 0.0):.4f} (±{m.get('cv_f1_std', 0.0):.3f})",
    })

comp_df = pd.DataFrame(comp_rows)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown(
    f"""
    <div class="callout-box">
        <b>★ Champion Selection:</b> <b>{champion['model_name']}</b> was selected as the production model 
        due to its superior composite discrimination (<b>ROC-AUC: {champion['roc_auc']:.4f}</b>) and high recall 
        (<b>{champion['recall']:.1%}</b>), capturing <b>{champion['confusion_matrix']['true_positives']} out of 374</b> 
        churners in the test cohort while maintaining balanced cross-validation stability.
    </div>
    """,
    unsafe_allow_html=True,
)

# Confusion Matrix & Business Cost Trade-offs
st.markdown('<div class="section-header">CONFUSION MATRIX & BUSINESS IMPACT ANALYSIS</div>', unsafe_allow_html=True)
col_cm_plot, col_cm_text = st.columns([1, 1])

cm = champion["confusion_matrix"]
matrix = cm["matrix"]
tn, fp, fn, tp = cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]

with col_cm_plot:
    cm_z = [[tn, fp], [fn, tp]]
    cm_text = [
        [f"True Neg (TN)<br><b>{tn:,}</b>", f"False Pos (FP)<br><b>{fp:,}</b>"],
        [f"False Neg (FN)<br><b>{fn:,}</b>", f"True Pos (TP)<br><b>{tp:,}</b>"],
    ]

    fig_cm = px.imshow(
        cm_z,
        text_auto=False,
        x=["Predicted: Retained", "Predicted: Churn"],
        y=["Actual: Retained", "Actual: Churn"],
        color_continuous_scale=[[0, "#1C2541"], [0.5, "#00B4D8"], [1, "#2EC4B6"]],
        labels=dict(x="Model Prediction", y="Actual Customer Status", color="Customers"),
    )
    fig_cm.update_traces(text=cm_text, texttemplate="%{text}", textfont={"size": 15, "color": "#FFFFFF"})
    fig_cm.update_layout(coloraxis_showscale=False)
    format_plotly_figure(fig_cm, title=f"Confusion Matrix ({champion['model_name']})")
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cm_text:
    st.markdown("#### The Business Economics of Churn Classification Errors:")
    st.markdown(
        rf"""
        - <b>False Negatives (FN = {fn:,}):</b> Churning subscribers that the model failed to flag.
          <br><span style="color: #FF5A6E;"><b>Cost:</b> Direct loss of customer lifetime value (LTV). In Telco, replacing a lost subscriber averages \$300–\$500 in acquisition marketing.</span>
        - <b>False Positives (FP = {fp:,}):</b> Loyal subscribers flagged as at-risk.
          <br><span style="color: #FFB347;"><b>Cost:</b> Cost of proactive retention incentives (e.g., \$10/mo contract discount or bonus gigabytes). Loyal customers rarely churn from being offered a perk, making FP much less detrimental than FN.</span>
        - <b>True Positives (TP = {tp:,}):</b> Correctly identified at-risk customers ({champion['recall']:.1%} recall) who can now be proactively saved.
        """
    , unsafe_allow_html=True)

# Curves: ROC & Precision-Recall
st.markdown('<div class="section-header">DISCRIMINATION CURVES (HELD-OUT TEST SET)</div>', unsafe_allow_html=True)
col_roc, col_pr = st.columns(2)

with col_roc:
    fig_roc = go.Figure()
    # Baseline diagonal
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#8D99AE", width=1.5),
            name="Random Guess (AUC=0.50)",
        )
    )

    colors = ["#2EC4B6", "#FF9F1C", "#00B4D8"]
    for idx, model_data in enumerate(comparison_list):
        roc = model_data["roc_curve"]
        fig_roc.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=2.5),
                name=f"{model_data['model_name']} (AUC={model_data['roc_auc']:.3f})",
            )
        )

    fig_roc.update_layout(
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1]),
        yaxis=dict(title="True Positive Rate (Recall)", range=[0, 1.05]),
    )
    format_plotly_figure(fig_roc, title="Receiver Operating Characteristic (ROC) Curves")
    st.plotly_chart(fig_roc, use_container_width=True)

with col_pr:
    fig_pr = go.Figure()
    # Baseline churn rate
    base_rate = metadata["historical_churn_rate"]
    fig_pr.add_trace(
        go.Scatter(
            x=[0, 1], y=[base_rate, base_rate],
            mode="lines",
            line=dict(dash="dash", color="#8D99AE", width=1.5),
            name=f"Baseline Churn Rate ({base_rate:.1%})",
        )
    )

    for idx, model_data in enumerate(comparison_list):
        pr = model_data["pr_curve"]
        fig_pr.add_trace(
            go.Scatter(
                x=pr["recall"],
                y=pr["precision"],
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=2.5),
                name=f"{model_data['model_name']}",
            )
        )

    fig_pr.update_layout(
        xaxis=dict(title="Recall", range=[0, 1]),
        yaxis=dict(title="Precision", range=[0, 1.05]),
    )
    format_plotly_figure(fig_pr, title="Precision-Recall Curves")
    st.plotly_chart(fig_pr, use_container_width=True)

# Global Feature Importance
st.markdown('<div class="section-header">GLOBAL FEATURE IMPORTANCE (PERMUTATION IMPORTANCE)</div>', unsafe_allow_html=True)

if perm_importance:
    feat_df = pd.DataFrame(perm_importance)
    feat_df = feat_df.sort_values(by="importance_mean", ascending=True)

    fig_feat = px.bar(
        feat_df,
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        color="importance_mean",
        color_continuous_scale=["#1C2541", "#00B4D8", "#2EC4B6"],
        labels={"importance_mean": "Drop in ROC-AUC when Shuffled", "feature": "Feature"},
    )
    fig_feat.update_layout(coloraxis_showscale=False, height=520)
    format_plotly_figure(fig_feat, title="Permutation Feature Importance on Held-Out Test Set")
    st.plotly_chart(fig_feat, use_container_width=True)

    st.caption(
        "ℹ️ <b>Methodology:</b> Permutation importance measures the drop in ROC-AUC score when a feature's values are "
        "randomly shuffled on the test set. Features whose shuffling causes the largest performance drops are the most essential.",
        unsafe_allow_html=True,
    )
