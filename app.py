import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

from models import (
    load_data,
    train_supervised,
    train_all_models,
    train_semi_supervised,
    train_anomaly_models,
    anomaly_local_explanation,
    FEATURES,
    FEATURES_CLEAN,
    ALL_FEATURES_CLEAN,
    FAILURE_MODE_COLUMNS,
)
from xai_utils import (
    get_shap_explainer,
    plot_shap_summary,
    plot_shap_waterfall,
    plot_shap_bar,
    get_lime_explainer,
    plot_lime_explanation,
    plot_pdp,
)

# ── Color palette ─────────────────────────────────────────────────────────────
# Color-blind friendly Okabe-Ito colors.
# Use these consistently so "No Failure" and "Failure" are clearly distinguishable.
NO_FAILURE_COLOR = "#0072B2"   # blue
FAILURE_COLOR = "#D55E00"      # orange/vermilion
THRESHOLD_COLOR = "#CC79A7"    # purple
NO_FAILURE_LABEL = "No Failure"
FAILURE_LABEL = "Failure"
FAILURE_COLOR_MAP = {NO_FAILURE_LABEL: NO_FAILURE_COLOR, FAILURE_LABEL: FAILURE_COLOR}


def add_failure_status(df_in):
    df_out = df_in.copy()
    df_out["Failure Status"] = df_out["Machine failure"].map({0: NO_FAILURE_LABEL, 1: FAILURE_LABEL})
    return df_out


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Interactive Explainable Predictive Maintenance Dashboard")
st.caption(
    "AI4I 2020 dataset · Visual Analytics · Supervised ML · Anomaly Detection · "
    "XAI: SHAP · LIME · PDP · Reconstruction/Deviation explanations"
)

# ── Load data and models ──────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data("ai4i2020.csv")


@st.cache_resource
def get_supervised_results(df):
    return train_supervised(df)


@st.cache_resource
def get_all_model_results(df):
    return train_all_models(df)


@st.cache_resource
def get_semi_results(df, frac):
    return train_semi_supervised(df, labeled_fraction=frac)


@st.cache_resource
def get_anomaly_results(df):
    return train_anomaly_models(df)


df = get_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Global Filters")
type_filter = st.sidebar.multiselect(
    "Machine Type", options=["H", "M", "L"], default=["H", "M", "L"]
)
wear_range = st.sidebar.slider(
    "Tool Wear [min]",
    int(df["Tool wear [min]"].min()),
    int(df["Tool wear [min]"].max()),
    (0, int(df["Tool wear [min]"].max())),
)
df_filtered = df[
    df["Type"].isin(type_filter)
    & df["Tool wear [min]"].between(wear_range[0], wear_range[1])
]
df_filtered_plot = add_failure_status(df_filtered)
st.sidebar.markdown(f"**Rows after filter:** {len(df_filtered):,}")
st.sidebar.info(
    "Leakage rule: TWF/HDF/PWF/OSF/RNF are shown for explanation, but are not used as model inputs."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 Data Explorer",
    "📊 Visual Analytics",
    "🤖 Supervised ML",
    "🚨 Anomaly Detection",
    "🔀 Semi-Supervised",
    "🧠 XAI Explanations",
    "📈 Model Comparison",
    "📚 Methodology & References",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Raw Data Overview")
    st.markdown(
        "Following Shneiderman's information-seeking mantra — **overview first, "
        "zoom/filter, details on demand** — this tab gives the dataset overview before modelling."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", f"{len(df_filtered):,}")
    col2.metric("Machine Failures", f"{int(df_filtered['Machine failure'].sum()):,}")
    col3.metric("Failure Rate", f"{df_filtered['Machine failure'].mean()*100:.2f}%")
    col4.metric("Model Features", len(FEATURES) + 1)
    col5.metric("Failure Modes", len(FAILURE_MODE_COLUMNS))

    with st.expander("Why this dataset?", expanded=True):
        st.markdown(
            "The AI4I dataset is suitable for this project because it is small enough for an "
            "interactive dashboard, but still realistic for predictive maintenance. It has interpretable "
            "machine/process variables such as temperature, speed, torque and tool wear, plus a rare "
            "machine-failure target."
        )

    st.markdown("**Raw data sample**")
    st.dataframe(df_filtered.head(300), use_container_width=True)

    st.subheader("Failure Type Breakdown")
    counts = df_filtered[FAILURE_MODE_COLUMNS].sum().reset_index()
    counts.columns = ["Failure Type", "Count"]
    fig_bar = px.bar(
        counts,
        x="Failure Type",
        y="Count",
        color="Failure Type",
        title="Number of occurrences per failure mode",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Visual Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Visual Analytics")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Correlation Heatmap**")
        fig_heat, ax = plt.subplots(figsize=(6, 4))
        corr = df_filtered[FEATURES + ["Machine failure"]].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation")
        plt.tight_layout()
        st.pyplot(fig_heat)

    with col_b:
        st.markdown("**Normal vs Failure Distribution**")
        feat_sel = st.selectbox("Select feature", FEATURES, key="eda_feat")
        fig_box = px.box(
            df_filtered_plot,
            x="Failure Status",
            y=feat_sel,
            color="Failure Status",
            color_discrete_map=FAILURE_COLOR_MAP,
            category_orders={"Failure Status": [NO_FAILURE_LABEL, FAILURE_LABEL]},
            labels={"Failure Status": "Machine status"},
            title=f"{feat_sel}: normal vs failure",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("**Density Plot: normal vs failure**")
    feat_density = st.selectbox("Select density feature", FEATURES, key="density_feat")
    fig_density, axd = plt.subplots(figsize=(8, 4))
    df_filtered[df_filtered["Machine failure"] == 0][feat_density].plot(
        kind="density", ax=axd, label=NO_FAILURE_LABEL, color=NO_FAILURE_COLOR, linewidth=2.5
    )
    if df_filtered["Machine failure"].sum() > 1:
        df_filtered[df_filtered["Machine failure"] == 1][feat_density].plot(
            kind="density", ax=axd, label=FAILURE_LABEL, color=FAILURE_COLOR, linewidth=2.5
        )
    axd.set_title(f"Density plot — {feat_density}")
    axd.set_xlabel(feat_density)
    axd.legend()
    axd.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_density)

    st.markdown("**Scatter: Torque vs Rotational Speed (failure highlighted)**")
    fig_scatter = px.scatter(
        df_filtered_plot,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="Failure Status",
        color_discrete_map=FAILURE_COLOR_MAP,
        category_orders={"Failure Status": [NO_FAILURE_LABEL, FAILURE_LABEL]},
        opacity=0.65,
        title="Rotational Speed vs Torque",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("**t-SNE Projection**")
    if st.button("Run t-SNE (may take ~10 seconds)"):
        if len(df_filtered) < 5:
            st.warning("Not enough rows after filtering.")
        else:
            sample = df_filtered.sample(min(1500, len(df_filtered)), random_state=42)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, len(sample)//10)))
            emb = tsne.fit_transform(sample[FEATURES])
            tsne_df = pd.DataFrame(emb, columns=["x", "y"])
            tsne_df["Failure"] = sample["Machine failure"].values
            tsne_df["Failure"] = tsne_df["Failure"].map({0: NO_FAILURE_LABEL, 1: FAILURE_LABEL})
            fig_tsne = px.scatter(
                tsne_df,
                x="x",
                y="y",
                color="Failure",
                color_discrete_map=FAILURE_COLOR_MAP,
                category_orders={"Failure": [NO_FAILURE_LABEL, FAILURE_LABEL]},
                title="t-SNE projection",
            )
            st.plotly_chart(fig_tsne, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Supervised ML
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Supervised Learning — XGBoost")
    st.markdown(
        "XGBoost is used as a strong supervised baseline. It learns directly from the "
        "`Machine failure` label. Class imbalance is handled with `scale_pos_weight`, not SMOTE."
    )

    if st.button("Train Supervised Model"):
        with st.spinner("Training XGBoost..."):
            results = get_supervised_results(df)
        st.session_state["sup_results"] = results
        st.success("Done!")

    if "sup_results" in st.session_state:
        model, X_train, X_test, y_train, y_test, y_pred, y_prob, report, auc, cm = st.session_state["sup_results"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{auc:.4f}")
        col2.metric("Precision (failure)", f"{report['1']['precision']:.4f}")
        col3.metric("Recall (failure)", f"{report['1']['recall']:.4f}")
        col4.metric("F1 (failure)", f"{report['1']['f1-score']:.4f}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Confusion Matrix**")
            fig_cm, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm)

        with col_r:
            st.markdown("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"XGBoost (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve", height=350)
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("**Full Classification Report**")
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Anomaly Detection — trained mainly on normal machine behavior")
    st.markdown(
        "This tab aligns the dashboard more directly with the project topic: **explainable anomaly detection**. "
        "The anomaly models are trained on normal samples only and then evaluated against the failure label."
    )

    if st.button("Train Anomaly Detection Models"):
        with st.spinner("Training Isolation Forest, LOF and MLP Autoencoder..."):
            anomaly_methods, anomaly_meta = get_anomaly_results(df)
        st.session_state["anomaly_results"] = (anomaly_methods, anomaly_meta)
        st.success("Done!")

    if "anomaly_results" in st.session_state:
        anomaly_methods, anomaly_meta = st.session_state["anomaly_results"]

        summary = []
        for name, r in anomaly_methods.items():
            m = r["metrics"]
            summary.append({
                "Method": name,
                "AUC-ROC": round(m["auc"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
                "Threshold": round(r["threshold"], 5),
            })
        st.markdown("**Anomaly Detection Performance Summary**")
        st.dataframe(pd.DataFrame(summary).set_index("Method"), use_container_width=True)

        st.markdown("**Anomaly Score Distributions**")
        selected_score_method = st.selectbox("Select method for score plot", list(anomaly_methods.keys()), key="anom_score_method")
        r = anomaly_methods[selected_score_method]
        y_test = anomaly_meta["y_test"].to_numpy()
        scores = r["score"]
        fig_score, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].hist(scores[y_test == 0], bins=50, alpha=0.70, label=NO_FAILURE_LABEL, color=NO_FAILURE_COLOR)
        axs[0].hist(scores[y_test == 1], bins=50, alpha=0.75, label=FAILURE_LABEL, color=FAILURE_COLOR)
        axs[0].axvline(r["threshold"], linestyle="--", linewidth=2.5, label="Threshold", color=THRESHOLD_COLOR)
        axs[0].set_title(f"{selected_score_method}: anomaly score")
        axs[0].set_xlabel("Higher score = more anomalous")
        axs[0].set_ylabel("Frequency")
        axs[0].legend()
        sns.heatmap(r["metrics"]["cm"], annot=True, fmt="d", cmap="Oranges", ax=axs[1], xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
        axs[1].set_title("Confusion matrix")
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig_score)

        st.markdown("**Local explanation for one selected sample**")
        selected_method = st.selectbox("Select anomaly method", list(anomaly_methods.keys()), key="anom_explain_method")
        y_pred = anomaly_methods[selected_method]["y_pred"]
        candidate_idx = np.where(y_pred == 1)[0]
        default_idx = int(candidate_idx[0]) if len(candidate_idx) else int(np.argmax(anomaly_methods[selected_method]["score"]))
        sample_idx = st.number_input("Test-set sample index", 0, len(anomaly_meta["y_test"]) - 1, default_idx, step=1)
        sample_idx = int(sample_idx)

        exp_df = anomaly_local_explanation(anomaly_methods[selected_method], anomaly_meta, sample_idx)
        actual = int(anomaly_meta["y_test"].iloc[sample_idx])
        predicted = int(y_pred[sample_idx])
        score = float(anomaly_methods[selected_method]["score"][sample_idx])
        st.markdown(
            f"**Actual:** {'Failure ⚠️' if actual else 'Normal ✅'} | "
            f"**Predicted:** {'Anomaly ⚠️' if predicted else 'Normal ✅'} | "
            f"**Score:** {score:.5f}"
        )
        st.dataframe(anomaly_meta["X_test"].iloc[[sample_idx]], use_container_width=True)
        st.dataframe(exp_df.head(10), use_container_width=True)

        top = exp_df.head(8).iloc[::-1]
        top = top.copy()
        top["Direction"] = np.where(top["Explanation value"] >= 0, "Pushes toward anomaly", "Pushes toward normal")
        fig_exp = px.bar(
            top,
            x="Explanation value",
            y="Feature",
            color="Direction",
            orientation="h",
            title=f"Local explanation — {selected_method}",
            color_discrete_map={"Pushes toward anomaly": FAILURE_COLOR, "Pushes toward normal": NO_FAILURE_COLOR},
        )
        st.plotly_chart(fig_exp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Semi-Supervised
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Semi-Supervised Learning — Self-Training")
    st.markdown(
        "This tab simulates a practical maintenance situation where only a fraction of the training labels are available."
    )

    labeled_frac = st.slider("Labeled fraction of training data", 0.1, 1.0, 0.3, 0.05)

    if st.button("Train Semi-Supervised Model"):
        with st.spinner("Training Self-Training model..."):
            semi_res = get_semi_results(df, labeled_frac)
        st.session_state["semi_results"] = semi_res
        st.success("Done!")

    if "semi_results" in st.session_state:
        semi_model, X_test_s, y_test_s, y_pred_s, y_prob_s, report_s, auc_s, cm_s, frac = st.session_state["semi_results"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{auc_s:.4f}")
        col2.metric("Precision", f"{report_s['1']['precision']:.4f}")
        col3.metric("Recall", f"{report_s['1']['recall']:.4f}")
        col4.metric("F1", f"{report_s['1']['f1-score']:.4f}")

        col_l, col_r = st.columns(2)
        with col_l:
            fig_cm2, ax2 = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens", ax=ax2, xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig_cm2)
        with col_r:
            fpr2, tpr2, _ = roc_curve(y_test_s, y_prob_s)
            fig_roc2 = go.Figure()
            fig_roc2.add_trace(go.Scatter(x=fpr2, y=tpr2, name=f"Self-training {int(frac*100)}% labeled"))
            fig_roc2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig_roc2.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve — Semi-Supervised", height=350)
            st.plotly_chart(fig_roc2, use_container_width=True)

        st.dataframe(pd.DataFrame(report_s).T.round(4), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — XAI Explanations
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Explainable AI — SHAP · LIME · PDP")
    st.markdown(
        "These explanations are shown for the supervised XGBoost model. Train the supervised model first."
    )

    if "sup_results" not in st.session_state:
        st.warning("Please train the Supervised Model in Tab 3 first.")
    else:
        model, X_train, X_test, y_train, y_test, y_pred, y_prob, report, auc, cm = st.session_state["sup_results"]

        xai_method = st.radio("Choose XAI method", ["SHAP", "LIME", "Partial Dependence Plot (PDP)"], horizontal=True)

        if xai_method == "SHAP":
            st.markdown(
                "**TreeSHAP** explains how each feature pushes a tree-based model prediction above or below the baseline prediction."
            )
            if st.button("Compute SHAP values"):
                with st.spinner("Computing SHAP..."):
                    explainer, shap_values = get_shap_explainer(model, X_train)
                st.session_state["shap"] = (explainer, shap_values)

            if "shap" in st.session_state:
                _, shap_values = st.session_state["shap"]
                shap_view = st.selectbox("SHAP plot type", ["Summary (beeswarm)", "Global Bar", "Waterfall (single instance)"])
                if shap_view == "Summary (beeswarm)":
                    st.pyplot(plot_shap_summary(shap_values))
                elif shap_view == "Global Bar":
                    st.pyplot(plot_shap_bar(shap_values))
                else:
                    idx = st.number_input("Training-set instance index", 0, len(X_train) - 1, 0, step=1)
                    st.pyplot(plot_shap_waterfall(shap_values, int(idx)))

        elif xai_method == "LIME":
            st.markdown(
                "**LIME** fits a simple local surrogate model around one prediction. It is useful when we need a single-instance explanation."
            )
            lime_idx = st.number_input("Test-set instance index", 0, len(X_test) - 1, 0, step=1)
            if st.button("Explain with LIME"):
                with st.spinner("Running LIME..."):
                    lime_exp = get_lime_explainer(X_train)
                    fig_lime, exp = plot_lime_explanation(lime_exp, model, X_test.iloc[int(lime_idx)])
                actual = int(y_test.iloc[int(lime_idx)])
                predicted = int(y_pred[int(lime_idx)])
                st.markdown(
                    f"**Actual label:** {'Failure ⚠️' if actual else 'No Failure ✅'} | "
                    f"**Predicted:** {'Failure ⚠️' if predicted else 'No Failure ✅'}"
                )
                st.pyplot(fig_lime)
                lime_df = pd.DataFrame(exp.as_list(), columns=["Condition", "Weight"])
                lime_df["Direction"] = lime_df["Weight"].apply(lambda w: "→ Failure" if w > 0 else "→ No Failure")
                st.dataframe(lime_df, use_container_width=True)

        else:
            st.markdown(
                "**Partial Dependence Plots** show the average marginal effect of one feature on the predicted failure probability."
            )
            pdp_feat = st.selectbox("Select feature for PDP", FEATURES_CLEAN + ["Type_enc"])
            if st.button("Generate PDP"):
                with st.spinner("Computing PDP..."):
                    fig_pdp = plot_pdp(model, X_train, pdp_feat)
                st.pyplot(fig_pdp)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Model Comparison")
    st.markdown(
        "The supervised models are compared to select a robust prediction baseline. The anomaly models are compared in the Anomaly Detection tab."
    )

    if st.button("Run Supervised Model Comparison"):
        with st.spinner("Training XGBoost, Random Forest and Logistic Regression..."):
            all_res, X_train_c, X_test_c, y_test_c = get_all_model_results(df)
        st.session_state["all_results"] = (all_res, X_test_c, y_test_c)
        st.success("Done!")

    if "all_results" in st.session_state:
        all_res, X_test_c, y_test_c = st.session_state["all_results"]
        summary = []
        for name, r in all_res.items():
            summary.append({
                "Model": name,
                "AUC-ROC": round(r["auc"], 4),
                "Precision": round(r["report"]["1"]["precision"], 4),
                "Recall": round(r["report"]["1"]["recall"], 4),
                "F1": round(r["report"]["1"]["f1-score"], 4),
            })
        st.dataframe(pd.DataFrame(summary).set_index("Model"), use_container_width=True)

        fig_all_roc = go.Figure()
        for name, r in all_res.items():
            fpr_m, tpr_m, _ = roc_curve(y_test_c, r["y_prob"])
            fig_all_roc.add_trace(go.Scatter(x=fpr_m, y=tpr_m, name=f"{name} (AUC={r['auc']:.3f})"))
        fig_all_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
        fig_all_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve Comparison")
        st.plotly_chart(fig_all_roc, use_container_width=True)

        st.markdown("**Confusion Matrices**")
        cols = st.columns(3)
        for i, (name, r) in enumerate(all_res.items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                fig_c, ax_c = plt.subplots(figsize=(3.5, 3))
                sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax_c, xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
                ax_c.set_xlabel("Predicted")
                ax_c.set_ylabel("Actual")
                st.pyplot(fig_c)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Methodology & References
# ═══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.subheader("Methodology and Literature Justification")

    st.markdown("""
### Why AI4I?
The AI4I 2020 dataset was chosen because it is directly designed for predictive maintenance and has interpretable process variables. It also contains a rare binary failure label, which makes it suitable for both supervised failure prediction and anomaly-detection evaluation.

### Why these model types?
- **XGBoost / Random Forest / Logistic Regression:** supervised baselines for labelled failure prediction.
- **Isolation Forest / Local Outlier Factor:** classical anomaly detection models trained on normal behavior.
- **MLP Autoencoder:** reconstruction-based anomaly detection; high reconstruction error suggests an unusual machine state.

### Why these XAI methods?
- **SHAP / TreeSHAP:** strong for tree-based models and supports global and local explanations.
- **LIME:** model-agnostic local surrogate explanation; useful for explaining one selected machine case.
- **PDP:** simple global view of how one feature changes average predicted failure probability.
- **Feature-wise reconstruction/deviation explanations:** simple local explanation for anomaly-detection models.

### Important methodological decision
Failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are **not used as model inputs**. They are target-side information and would leak the answer into the model. They are used only for exploratory display and discussion.
""")

    st.markdown("""
### Reference list for report/slides

[1] UCI Machine Learning Repository, **AI4I 2020 Predictive Maintenance Dataset**. DOI: `10.24432/C5HS5C`.

[2] S. Matzka, **Explainable Artificial Intelligence for Predictive Maintenance Applications**, International Conference on Artificial Intelligence for Industries (AI4I), 2020. DOI: `10.1109/AI4I49448.2020.00023`.

[3] A. Torcianti and S. Matzka, **Explainable Artificial Intelligence for Predictive Maintenance Applications using a Local Surrogate Model**, 4th International Conference on Artificial Intelligence for Industries (AI4I), 2021. DOI: `10.1109/AI4I51902.2021.00029`.

[4] M. T. Ribeiro, S. Singh, and C. Guestrin, **“Why Should I Trust You?”: Explaining the Predictions of Any Classifier**, KDD, 2016.

[5] S. M. Lundberg et al., **From Local Explanations to Global Understanding with Explainable AI for Trees**, Nature Machine Intelligence, 2020.

[6] T. Chen and C. Guestrin, **XGBoost: A Scalable Tree Boosting System**, KDD, 2016.

[7] L. Cummins et al., **Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities**, arXiv:2401.07871, 2024.

[8] A. Maged, S. Haridy, and H. Shen, **Explainable Artificial Intelligence Techniques for Accurate Fault Detection and Diagnosis: A Review**, arXiv:2404.11597, 2024.

[9] L. C. Brito, G. A. Susto, J. N. Brito, and M. A. V. Duarte, **An Explainable Artificial Intelligence Approach for Unsupervised Fault Detection and Diagnosis in Rotating Machinery**, arXiv:2102.11848, 2021.
""")
