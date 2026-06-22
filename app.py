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
    train_semi_supervised_all,
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

# ── Color palette ────────────────────────────────────────────────────────────
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
    return train_semi_supervised_all(df, labeled_fraction=frac)


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
dashboard_tab,tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Dashboard",
    "📋 Data Explorer",
    "📊 Visual Analytics",
    "🤖 Supervised ML",
    "🚨 Anomaly Detection",
    "🔀 Semi-Supervised",
    "🧠 XAI Explanations",
    "📈 Model Comparison",
    "📚 Methodology & References",
])

with dashboard_tab:

    st.header("Executive Dashboard")

    # ======================
    # KPI CARDS
    # ======================

    total_records = len(df_filtered)
    total_failures = int(df_filtered["Machine failure"].sum())
    failure_rate = df_filtered["Machine failure"].mean() * 100

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Records", f"{total_records:,}")
    col2.metric("Failures", total_failures)
    col3.metric("Failure Rate", f"{failure_rate:.2f}%")
    col4.metric("Machine Types", df_filtered["Type"].nunique())

    # ======================
    # ROW 1
    # ======================

    col1,col2 = st.columns(2)

    with col1:

        failure_counts = (
            df_filtered["Machine failure"]
            .value_counts()
            .reset_index()
        )

        failure_counts.columns = ["Status","Count"]

        fig = px.pie(
            failure_counts,
            names="Status",
            values="Count",
            title="Failure Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        type_counts = (
            df_filtered["Type"]
            .value_counts()
            .reset_index()
        )

        type_counts.columns = ["Type","Count"]

        fig = px.bar(
            type_counts,
            x="Type",
            y="Count",
            color="Type",
            title="Machine Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        st.plotly_chart(fig,use_container_width=True)

    # ======================
    # ROW 2
    # ======================

    col1,col2 = st.columns(2)

    with col1:

        counts = (
            df_filtered[FAILURE_MODE_COLUMNS]
            .sum()
            .reset_index()
        )

        counts.columns = ["Failure Type","Count"]

        fig = px.bar(
            counts,
            x="Failure Type",
            y="Count",
            title="Failure Modes"
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        corr = df_filtered[
            FEATURES + ["Machine failure"]
        ].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Heatmap"
        )

        st.plotly_chart(fig,use_container_width=True)

    # ======================
    # ROW 3
    # ======================

    col1,col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df_filtered_plot,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color="Failure Status",
            title="Torque vs Speed"
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        st.subheader("Dataset Summary")

        st.metric(
            "Average Tool Wear",
            f"{df_filtered['Tool wear [min]'].mean():.1f}"
        )

        st.metric(
            "Average Torque",
            f"{df_filtered['Torque [Nm]'].mean():.1f}"
        )

        st.metric(
            "Average RPM",
            f"{df_filtered['Rotational speed [rpm]'].mean():.0f}"
        )

    # ======================
    # ROW 4
    # ======================

    
   
    st.markdown("---")
    st.header("🤖 AI Performance Overview")

    # ==================================================
    # SUPERVISED MODELS
    # ==================================================

    st.subheader("🤖 Supervised Models")

    if "all_results" in st.session_state:

        all_res, _, _, _ = st.session_state["all_results"]

        cols = st.columns(len(all_res))

        for i, (name, r) in enumerate(all_res.items()):
            cols[i].metric(name, f"{r['auc']:.3f}")

        supervised_df = pd.DataFrame(
            [
                {"Model": name, "AUC": r["auc"]}
                for name, r in all_res.items()
            ]
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                supervised_df.sort_values("AUC"),
                x="AUC",
                y="Model",
                orientation="h",
                color="Model",
                title="Supervised Model Ranking",
                height=250
            )

            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            best_model = supervised_df.loc[
                supervised_df["AUC"].idxmax()
            ]

            st.metric(
                "Best Model",
                best_model["Model"]
            )

            st.metric(
                "Best AUC",
                f"{best_model['AUC']:.3f}"
            )

            st.metric(
                "Models Compared",
                len(supervised_df)
            )

    # ==================================================
    # ANOMALY DETECTION
    # ==================================================

    st.subheader("🚨 Anomaly Detection")

    if "anomaly_results" in st.session_state:

        anomaly_methods, _ = st.session_state["anomaly_results"]

        cols = st.columns(len(anomaly_methods))

        for i, (name, r) in enumerate(anomaly_methods.items()):
            cols[i].metric(
                name,
                f"{r['metrics']['auc']:.3f}"
            )

        anomaly_df = pd.DataFrame(
            [
                {
                    "Method": name,
                    "AUC": r["metrics"]["auc"]
                }
                for name, r in anomaly_methods.items()
            ]
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                anomaly_df.sort_values("AUC"),
                x="AUC",
                y="Method",
                orientation="h",
                color="Method",
                title="Anomaly Model Ranking",
                height=250
            )

            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            best_method = anomaly_df.loc[
                anomaly_df["AUC"].idxmax()
            ]

            st.metric(
                "Best Method",
                best_method["Method"]
            )

            st.metric(
                "Best AUC",
                f"{best_method['AUC']:.3f}"
            )

            st.metric(
                "Methods Compared",
                len(anomaly_df)
            )

    # ==================================================
    # SEMI SUPERVISED
    # ==================================================

    st.subheader("🔀 Semi-Supervised Models")

    if "semi_results" in st.session_state:

        semi_res, _ = st.session_state["semi_results"]

        cols = st.columns(len(semi_res))

        for i, (name, r) in enumerate(semi_res.items()):
            cols[i].metric(
                name,
                f"{r['auc']:.3f}"
            )

        semi_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "AUC": r["auc"]
                }
                for name, r in semi_res.items()
            ]
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                semi_df.sort_values("AUC"),
                x="AUC",
                y="Model",
                orientation="h",
                color="Model",
                title="Semi-Supervised Ranking",
                height=250
            )

            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            best_model = semi_df.loc[
                semi_df["AUC"].idxmax()
            ]

            st.metric(
                "Best Model",
                best_model["Model"]
            )

            st.metric(
                "Best AUC",
                f"{best_model['AUC']:.3f}"
            )

            st.metric(
                "Models Compared",
                len(semi_df)
            )

    # ==================================================
    # XAI SUMMARY
    # ==================================================

    st.subheader("🧠 Explainable AI Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Top Feature",
            "Tool Wear"
        )

    with col2:
        st.metric(
            "Methods",
            "SHAP • LIME • PDP"
        )

    st.info(
        """
        Tool Wear is the strongest contributor
        to machine failure predictions.

        Detailed SHAP, LIME and PDP visualizations
        are available in the XAI Explanations tab.
        """
    )
    # ==================================================
    # RECENT RECORDS
    # ==================================================

    st.subheader("📄 Recent Machine Records")

    st.dataframe(
        df_filtered.head(10),
        use_container_width=True
    )
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

    st.subheader("Feature Relationship Explorer")
    st.markdown(
        "This section shows additional physical/process relationships, similar to the torque-speed plot. "
        "The aim is to see whether failures appear in special operating regions instead of only looking at one feature at a time."
    )

    relation_df = df_filtered.copy()
    relation_df["Temperature difference [K]"] = (
        relation_df["Process temperature [K]"] - relation_df["Air temperature [K]"]
    )
    relation_df["Mechanical power [kW]"] = (
        relation_df["Torque [Nm]"] * 2 * np.pi * relation_df["Rotational speed [rpm]"] / 60 / 1000
    )
    relation_df_plot = add_failure_status(relation_df)

    relationship_presets = {
        "Torque vs Rotational speed": ("Rotational speed [rpm]", "Torque [Nm]"),
        "Air temperature vs Process temperature": ("Air temperature [K]", "Process temperature [K]"),
        "Tool wear vs Torque": ("Tool wear [min]", "Torque [Nm]"),
        "Tool wear vs Rotational speed": ("Tool wear [min]", "Rotational speed [rpm]"),
        "Tool wear vs Mechanical power": ("Tool wear [min]", "Mechanical power [kW]"),
        "Temperature difference vs Tool wear": ("Tool wear [min]", "Temperature difference [K]"),
        "Rotational speed vs Mechanical power": ("Rotational speed [rpm]", "Mechanical power [kW]"),
    }

    preset = st.selectbox(
        "Choose a relationship preset",
        list(relationship_presets.keys()),
        key="relationship_preset",
    )
    x_col, y_col = relationship_presets[preset]

    rel_col1, rel_col2, rel_col3 = st.columns([1, 1, 1])
    with rel_col1:
        st.metric("Selected x-axis", x_col)
    with rel_col2:
        st.metric("Selected y-axis", y_col)
    with rel_col3:
        corr_val = relation_df[[x_col, y_col]].corr().iloc[0, 1]
        st.metric("Pearson correlation", f"{corr_val:.3f}")

    show_trend = st.checkbox("Show simple linear trend line", value=True, key="relation_trend")

    fig_relation = px.scatter(
        relation_df_plot,
        x=x_col,
        y=y_col,
        color="Failure Status",
        color_discrete_map=FAILURE_COLOR_MAP,
        category_orders={"Failure Status": [NO_FAILURE_LABEL, FAILURE_LABEL]},
        opacity=0.65,
        title=f"{preset} — failure highlighted",
        hover_data=["Type", "Machine failure", "Tool wear [min]", "Rotational speed [rpm]", "Torque [Nm]"],
    )

    if show_trend and len(relation_df_plot) >= 2:
        for label, color in [(NO_FAILURE_LABEL, NO_FAILURE_COLOR), (FAILURE_LABEL, FAILURE_COLOR)]:
            sub = relation_df_plot[relation_df_plot["Failure Status"] == label][[x_col, y_col]].dropna()
            if len(sub) >= 2 and sub[x_col].nunique() > 1:
                x_values = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                slope, intercept = np.polyfit(sub[x_col], sub[y_col], 1)
                y_values = slope * x_values + intercept
                fig_relation.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines",
                        name=f"{label} trend",
                        line=dict(color=color, dash="dash", width=2),
                    )
                )

    st.plotly_chart(fig_relation, use_container_width=True)

    with st.expander("Interpretation of feature relationship plots", expanded=False):
        st.markdown(
            "These relationship plots compare pairs of physical and process-related variables and color the samples by failure status. "
            "They are used to visually inspect whether machine failures occur in specific operating regions.\n\n"
            "For example, the relationship between rotational speed and torque represents the machine's load-speed behavior. "
            "High torque at lower rotational speed can indicate heavy-load operating conditions. "
            "The temperature difference shows how much the process temperature rises above the ambient air temperature, while mechanical power combines torque and rotational speed into an engineering-oriented load indicator.\n\n"
            "These plots do not prove causality, but they help identify patterns that may be useful for model interpretation and anomaly/failure detection."
        )

    st.markdown("**Derived feature distributions**")
    derived_feature = st.selectbox(
        "Select derived/relationship feature",
        ["Temperature difference [K]", "Mechanical power [kW]"],
        key="derived_feature",
    )
    fig_derived = px.box(
        relation_df_plot,
        x="Failure Status",
        y=derived_feature,
        color="Failure Status",
        color_discrete_map=FAILURE_COLOR_MAP,
        category_orders={"Failure Status": [NO_FAILURE_LABEL, FAILURE_LABEL]},
        title=f"{derived_feature}: normal vs failure",
    )
    st.plotly_chart(fig_derived, use_container_width=True)

    st.markdown("**Scatter Matrix: multiple pairwise relations**")
    matrix_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Mechanical power [kW]",
    ]
    matrix_sample = relation_df_plot.sample(min(1200, len(relation_df_plot)), random_state=42)
    fig_matrix = px.scatter_matrix(
        matrix_sample,
        dimensions=matrix_cols,
        color="Failure Status",
        color_discrete_map=FAILURE_COLOR_MAP,
        category_orders={"Failure Status": [NO_FAILURE_LABEL, FAILURE_LABEL]},
        title="Pairwise relationship matrix (sampled for readability)",
        opacity=0.55,
    )
    fig_matrix.update_traces(diagonal_visible=False, showupperhalf=False)
    fig_matrix.update_layout(height=850)
    st.plotly_chart(fig_matrix, use_container_width=True)

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
    st.subheader("Supervised Learning — XGBoost · Random Forest · Logistic Regression")
    st.markdown(
        "This tab trains **three supervised failure-prediction models** with the same train/test split. "
        "They learn directly from the `Machine failure` label. These are not pure anomaly-detection models; "
        "they are labelled prediction baselines. Class imbalance is handled with class weighting / `scale_pos_weight`, not SMOTE."
    )

    with st.expander("What is AUC-ROC?", expanded=False):
        st.markdown(
            "**AUC-ROC** means *Area Under the Receiver Operating Characteristic Curve*. "
            "It measures how well a model separates failure and no-failure samples across all possible thresholds. "
            "AUC = 0.5 means random guessing, while AUC = 1.0 means perfect separation. "
            "Because this dataset is imbalanced, AUC is useful, but we still also check precision, recall, and F1."
        )

    if st.button("Train Supervised Models"):
        with st.spinner("Training XGBoost, Random Forest and Logistic Regression..."):
            # Detailed XGBoost tuple for the XAI tab.
            xgb_results = get_supervised_results(df)
            # Full comparison dictionary for this tab and the Model Comparison tab.
            all_res, X_train_c, X_test_c, y_test_c = get_all_model_results(df)
        st.session_state["sup_results"] = xgb_results
        st.session_state["all_results"] = (all_res, X_train_c, X_test_c, y_test_c)
        st.success("Done!")

    if "all_results" in st.session_state:
        all_res, X_train_c, X_test_c, y_test_c = st.session_state["all_results"]

        summary = []
        for name, r in all_res.items():
            summary.append({
                "Model": name,
                "AUC-ROC": round(r["auc"], 4),
                "Accuracy": round(r["report"]["accuracy"], 4),
                "Precision (failure)": round(r["report"]["1"]["precision"], 4),
                "Recall (failure)": round(r["report"]["1"]["recall"], 4),
                "F1 (failure)": round(r["report"]["1"]["f1-score"], 4),
            })

        st.markdown("**Supervised model summary**")
        st.dataframe(pd.DataFrame(summary).set_index("Model"), use_container_width=True)

        selected_supervised = st.selectbox(
            "Select supervised model for detailed view",
            list(all_res.keys()),
            key="supervised_detail_model",
        )
        r = all_res[selected_supervised]
        report = r["report"]
        auc = r["auc"]
        cm = r["cm"]
        y_pred = r["y_pred"]
        y_prob = r["y_prob"]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("AUC-ROC", f"{auc:.4f}")
        col2.metric("Accuracy", f"{report['accuracy']:.4f}")
        col3.metric("Precision (failure)", f"{report['1']['precision']:.4f}")
        col4.metric("Recall (failure)", f"{report['1']['recall']:.4f}")
        col5.metric("F1 (failure)", f"{report['1']['f1-score']:.4f}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**Confusion Matrix — {selected_supervised}**")
            fig_cm, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm)

        with col_r:
            st.markdown(f"**ROC Curve — {selected_supervised}**")
            fpr, tpr, _ = roc_curve(y_test_c, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{selected_supervised} (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve", height=350)
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("**Full Classification Report**")
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        st.info(
            "The XAI tab can explain **XGBoost, Random Forest, and Logistic Regression**. "
            "SHAP is available for tree-based models; LIME and PDP can be applied to all supervised models."
        )

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

        y_true_anom = anomaly_meta["y_test"].to_numpy()
        y_pred = anomaly_methods[selected_method]["y_pred"]
        anom_scores = anomaly_methods[selected_method]["score"]

        sample_group = st.selectbox(
            "Select sample group for local explanation",
            [
                "All test samples",
                "Actual failures only",
                "Actual normal samples only",
                "Predicted anomalies only",
                "Predicted normal samples only",
                "Correct predictions only",
                "Wrong predictions only",
                "False positives: normal predicted as anomaly",
                "False negatives: failure predicted as normal",
            ],
            key="anom_sample_group",
        )

        group_masks = {
            "All test samples": np.ones_like(y_true_anom, dtype=bool),
            "Actual failures only": y_true_anom == 1,
            "Actual normal samples only": y_true_anom == 0,
            "Predicted anomalies only": y_pred == 1,
            "Predicted normal samples only": y_pred == 0,
            "Correct predictions only": y_true_anom == y_pred,
            "Wrong predictions only": y_true_anom != y_pred,
            "False positives: normal predicted as anomaly": (y_true_anom == 0) & (y_pred == 1),
            "False negatives: failure predicted as normal": (y_true_anom == 1) & (y_pred == 0),
        }
        candidate_idx = np.where(group_masks[sample_group])[0]
        st.caption(f"{len(candidate_idx)} matching test samples for: {sample_group}")

        if len(candidate_idx) == 0:
            st.warning("No samples are available for this group with the selected anomaly method.")
        else:
            sort_choice = st.selectbox(
                "Sort matching samples",
                ["Highest anomaly score first", "Lowest anomaly score first", "Original test-set order"],
                key="anom_sort_choice",
            )
            if sort_choice == "Highest anomaly score first":
                candidate_idx = candidate_idx[np.argsort(anom_scores[candidate_idx])[::-1]]
            elif sort_choice == "Lowest anomaly score first":
                candidate_idx = candidate_idx[np.argsort(anom_scores[candidate_idx])]

            def _format_anomaly_idx(i):
                actual_txt = "Failure" if int(y_true_anom[i]) == 1 else "Normal"
                pred_txt = "Anomaly" if int(y_pred[i]) == 1 else "Normal"
                return f"Index {int(i)} | Actual: {actual_txt} | Predicted: {pred_txt} | Score: {float(anom_scores[i]):.5f}"

            sample_idx = st.selectbox(
                "Select test-set sample index",
                options=candidate_idx.tolist(),
                index=0,
                format_func=_format_anomaly_idx,
                key="anom_selected_sample_idx",
            )
            sample_idx = int(sample_idx)

            with st.expander("What do these sample groups mean?", expanded=False):
                st.markdown("""
**Correct predictions** means the anomaly model prediction matches the true `Machine failure` label.  
**Wrong predictions** means the prediction does not match the true label.  
**False positive** means the machine was actually normal, but the model predicted anomaly.  
**False negative** means the machine was actually a failure, but the model predicted normal.
""")

            exp_df = anomaly_local_explanation(anomaly_methods[selected_method], anomaly_meta, sample_idx)
            actual = int(y_true_anom[sample_idx])
            predicted = int(y_pred[sample_idx])
            score = float(anom_scores[sample_idx])
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
    st.subheader("Semi-Supervised Learning — Self-Training Comparison")
    st.markdown(
        "This tab simulates a practical maintenance situation where only a fraction of the training labels are available. "
        "We apply the same semi-supervised strategy to three base models: **XGBoost**, **Random Forest**, and **Logistic Regression**."
    )

    with st.expander("What type of machine learning is this?", expanded=True):
        st.markdown(
            "This is **semi-supervised classification** using `SelfTrainingClassifier`. "
            "The model receives a small labelled subset and a larger unlabelled subset. "
            "Unlabelled samples are internally marked as `-1`. The model first learns from labelled data, "
            "then adds confident pseudo-labels for unlabelled samples and retrains. "
            "This is useful for predictive maintenance because real failure labels are often rare or expensive to collect."
        )

    labeled_frac = st.slider("Labeled fraction of training data", 0.1, 1.0, 0.3, 0.05)

    if st.button("Train Semi-Supervised Models"):
        with st.spinner("Training Self-Training XGBoost, Random Forest and Logistic Regression..."):
            semi_res, semi_meta = get_semi_results(df, labeled_frac)
        st.session_state["semi_results"] = (semi_res, semi_meta)
        st.success("Done!")

    if "semi_results" in st.session_state:
        semi_res, semi_meta = st.session_state["semi_results"]
        y_test_s = semi_meta["y_test"]

        st.info(
            f"Visible labels: {semi_meta['visible_labels']:,} | Hidden labels: {semi_meta['hidden_labels']:,} | "
            f"Visible normal samples: {semi_meta['visible_normals']:,} | Visible failure samples: {semi_meta['visible_failures']:,}"
        )

        summary = []
        for name, r in semi_res.items():
            summary.append({
                "Model": name,
                "AUC-ROC": round(r["auc"], 4),
                "Accuracy": round(r["report"]["accuracy"], 4),
                "Precision (failure)": round(r["report"]["1"]["precision"], 4),
                "Recall (failure)": round(r["report"]["1"]["recall"], 4),
                "F1 (failure)": round(r["report"]["1"]["f1-score"], 4),
            })
        st.markdown("**Semi-supervised model summary**")
        st.dataframe(pd.DataFrame(summary).set_index("Model"), use_container_width=True)

        selected_semi = st.selectbox(
            "Select semi-supervised model for detailed view",
            list(semi_res.keys()),
            key="semi_detail_model",
        )
        r = semi_res[selected_semi]
        report_s = r["report"]
        auc_s = r["auc"]
        cm_s = r["cm"]
        y_prob_s = r["y_prob"]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("AUC-ROC", f"{auc_s:.4f}")
        col2.metric("Accuracy", f"{report_s['accuracy']:.4f}")
        col3.metric("Precision", f"{report_s['1']['precision']:.4f}")
        col4.metric("Recall", f"{report_s['1']['recall']:.4f}")
        col5.metric("F1", f"{report_s['1']['f1-score']:.4f}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**Confusion Matrix — {selected_semi}**")
            fig_cm2, ax2 = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens", ax=ax2, xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm2)
        with col_r:
            st.markdown(f"**ROC Curve — {selected_semi}**")
            fpr2, tpr2, _ = roc_curve(y_test_s, y_prob_s)
            fig_roc2 = go.Figure()
            fig_roc2.add_trace(go.Scatter(x=fpr2, y=tpr2, name=f"{selected_semi} (AUC={auc_s:.3f})"))
            fig_roc2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig_roc2.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve — Semi-Supervised", height=350)
            st.plotly_chart(fig_roc2, use_container_width=True)

        st.markdown("**Full Classification Report**")
        st.dataframe(pd.DataFrame(report_s).T.round(4), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — XAI Explanations
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Explainable AI — SHAP · LIME · PDP")
    st.markdown(
        "This tab explains the **supervised failure-prediction models**. "
        "You can select XGBoost, Random Forest, or Logistic Regression and inspect how the model makes predictions."
    )

    if "all_results" not in st.session_state:
        st.warning("Please train the Supervised Models in Tab 3 first.")
    else:
        all_res, X_train_base, X_test_base, y_test = st.session_state["all_results"]

        col_model, col_method = st.columns([1.2, 1.8])
        with col_model:
            xai_model_name = st.selectbox(
                "Select model to explain",
                list(all_res.keys()),
                key="xai_model_name",
            )
        with col_method:
            xai_method = st.radio(
                "Choose XAI method",
                ["SHAP", "LIME", "Partial Dependence Plot (PDP)"],
                horizontal=True,
            )

        selected_result = all_res[xai_model_name]
        selected_model = selected_result["model"]
        selected_y_pred = np.asarray(selected_result["y_pred"])
        selected_y_prob = np.asarray(selected_result["y_prob"])
        selected_scaler = selected_result.get("scaler")

        # XGBoost and Random Forest were trained on the original clean feature table.
        # Logistic Regression was trained on standardized values, so its XAI input must
        # also be standardized. We keep the same feature names for readability.
        if selected_scaler is not None:
            X_train_xai = pd.DataFrame(selected_scaler.transform(X_train_base), columns=ALL_FEATURES_CLEAN, index=X_train_base.index)
            X_test_xai = pd.DataFrame(selected_scaler.transform(X_test_base), columns=ALL_FEATURES_CLEAN, index=X_test_base.index)
            st.info(
                "This model uses standardized/scaled input values. Therefore, the XAI plots for this model use standardized feature values, "
                "not the original physical units."
            )
        else:
            X_train_xai = X_train_base
            X_test_xai = X_test_base

        st.caption(
            f"Currently explaining: **{xai_model_name}** · "
            "Failure class = 1, No Failure class = 0"
        )

        if xai_method == "SHAP":
            st.markdown(
                "**SHAP** explains how features push model predictions toward Failure or No Failure. "
                "For this dashboard, SHAP is enabled for the tree-based models **XGBoost** and **Random Forest**. "
                "For Logistic Regression, use LIME/PDP or interpret coefficients as a simpler model-specific explanation."
            )

            if xai_model_name == "Logistic Regression":
                st.warning(
                    "SHAP is intentionally disabled here for Logistic Regression to keep the dashboard fast and simple. "
                    "Use LIME for local Logistic Regression explanations or PDP for global feature effects."
                )
            else:
                shap_key = f"shap_{xai_model_name}"
                if st.button(f"Compute SHAP values for {xai_model_name}"):
                    with st.spinner(f"Computing SHAP for {xai_model_name}..."):
                        explainer, shap_values = get_shap_explainer(selected_model, X_train_xai)
                    st.session_state[shap_key] = (explainer, shap_values)

                if shap_key in st.session_state:
                    _, shap_values = st.session_state[shap_key]
                    shap_view = st.selectbox(
                        "SHAP plot type",
                        ["Summary (beeswarm)", "Global Bar", "Waterfall (single instance)"],
                        key=f"shap_view_{xai_model_name}",
                    )
                    if shap_view == "Summary (beeswarm)":
                        st.pyplot(plot_shap_summary(shap_values))
                    elif shap_view == "Global Bar":
                        st.pyplot(plot_shap_bar(shap_values))
                    else:
                        idx = st.number_input(
                            "Training-set instance index",
                            0,
                            len(X_train_xai) - 1,
                            0,
                            step=1,
                            key=f"shap_idx_{xai_model_name}",
                        )
                        st.pyplot(plot_shap_waterfall(shap_values, int(idx)))

        elif xai_method == "LIME":
            st.markdown(
                "**LIME** is model-agnostic. This means it can explain **XGBoost, Random Forest, and Logistic Regression**. "
                "It explains one selected test instance by fitting a simple local surrogate model around that specific prediction."
            )

            if "lime_selected_idx" not in st.session_state:
                st.session_state["lime_selected_idx"] = 0

            y_test_arr = y_test.to_numpy()
            y_pred_arr = selected_y_pred
            candidate_mode = st.selectbox(
                "Random instance source",
                [
                    "All test instances",
                    "Actual failures only",
                    "Predicted failures only",
                    "Misclassified instances only",
                    "False negatives only (missed failures)",
                    "False positives only (false alarms)",
                ],
                help="This only controls the random-pick button. The manual index can still be any test-set row.",
                key=f"lime_candidate_mode_{xai_model_name}",
            )

            if candidate_mode == "Actual failures only":
                candidate_indices = np.where(y_test_arr == 1)[0]
            elif candidate_mode == "Predicted failures only":
                candidate_indices = np.where(y_pred_arr == 1)[0]
            elif candidate_mode == "Misclassified instances only":
                candidate_indices = np.where(y_test_arr != y_pred_arr)[0]
            elif candidate_mode == "False negatives only (missed failures)":
                candidate_indices = np.where((y_test_arr == 1) & (y_pred_arr == 0))[0]
            elif candidate_mode == "False positives only (false alarms)":
                candidate_indices = np.where((y_test_arr == 0) & (y_pred_arr == 1))[0]
            else:
                candidate_indices = np.arange(len(X_test_xai))

            if len(candidate_indices) == 0:
                st.warning("No instances are available for this random-selection category. Please choose another category.")
                candidate_indices = np.arange(len(X_test_xai))

            col_lime_a, col_lime_b, col_lime_c = st.columns([1.2, 1.2, 2.0])
            with col_lime_a:
                random_clicked = st.button("🎲 Pick random & explain", help="Selects a random instance from the selected source and immediately runs LIME.", key=f"lime_random_{xai_model_name}")
            with col_lime_b:
                explain_clicked = st.button("Explain selected index", key=f"lime_explain_{xai_model_name}")
            with col_lime_c:
                lime_idx_manual = st.number_input(
                    "Test-set instance index",
                    min_value=0,
                    max_value=len(X_test_xai) - 1,
                    value=int(st.session_state["lime_selected_idx"]),
                    step=1,
                    help="Manual index of the test sample that should be explained by LIME.",
                    key=f"lime_idx_{xai_model_name}",
                )

            lime_idx_to_explain = None
            if random_clicked:
                lime_idx_to_explain = int(np.random.choice(candidate_indices))
                st.session_state["lime_selected_idx"] = lime_idx_to_explain
            elif explain_clicked:
                lime_idx_to_explain = int(lime_idx_manual)
                st.session_state["lime_selected_idx"] = lime_idx_to_explain

            st.caption(
                f"Current selected test index: **{int(st.session_state['lime_selected_idx'])}** · "
                f"available random candidates in this mode: **{len(candidate_indices)}**"
            )

            if lime_idx_to_explain is not None:
                with st.spinner(f"Running LIME for {xai_model_name}..."):
                    lime_exp = get_lime_explainer(X_train_xai)
                    fig_lime, exp = plot_lime_explanation(lime_exp, selected_model, X_test_xai.iloc[int(lime_idx_to_explain)])
                actual = int(y_test.iloc[int(lime_idx_to_explain)])
                predicted = int(selected_y_pred[int(lime_idx_to_explain)])
                probability = float(selected_y_prob[int(lime_idx_to_explain)])

                st.markdown(
                    f"**Model:** `{xai_model_name}`  \n"
                    f"**Explained test index:** `{int(lime_idx_to_explain)}`  \n"
                    f"**Actual label:** {'Failure ⚠️' if actual else 'No Failure ✅'} | "
                    f"**Predicted:** {'Failure ⚠️' if predicted else 'No Failure ✅'} | "
                    f"**Predicted failure probability:** `{probability:.3f}`"
                )

                st.markdown("**Input feature values used by the selected model**")
                st.dataframe(X_test_xai.iloc[[int(lime_idx_to_explain)]], use_container_width=True)

                st.pyplot(fig_lime)
                lime_df = pd.DataFrame(exp.as_list(), columns=["Condition", "Weight"])
                lime_df["Direction"] = lime_df["Weight"].apply(lambda w: "→ Failure" if w > 0 else "→ No Failure")
                st.dataframe(lime_df, use_container_width=True)

        else:
            st.markdown(
                "**Partial Dependence Plots** show the average marginal effect of one feature on the predicted failure probability. "
                "PDP can be generated for each supervised model."
            )
            pdp_feat = st.selectbox("Select feature for PDP", FEATURES_CLEAN + ["Type_enc"], key=f"pdp_feat_{xai_model_name}")
            if st.button(f"Generate PDP for {xai_model_name}"):
                with st.spinner(f"Computing PDP for {xai_model_name}..."):
                    fig_pdp = plot_pdp(selected_model, X_train_xai, pdp_feat)
                st.pyplot(fig_pdp)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Model Comparison")
    st.markdown(
        "The supervised models are compared to select a robust prediction baseline. "
        "The anomaly models are compared in the Anomaly Detection tab. "
        "The XGBoost row uses the same configuration and train/test split as the detailed Supervised ML tab, "
        "so the XGBoost metrics should match after retraining both sections."
    )

    if st.button("Run Supervised Model Comparison"):
        with st.spinner("Training XGBoost, Random Forest and Logistic Regression..."):
            all_res, X_train_c, X_test_c, y_test_c = get_all_model_results(df)
        st.session_state["all_results"] = (all_res, X_test_c, y_test_c)
        st.success("Done!")

    if "all_results" in st.session_state:
        all_res, X_train_c, X_test_c, y_test_c = st.session_state["all_results"]
        summary = []
        for name, r in all_res.items():
            summary.append({
                "Model": name,
                "AUC-ROC": round(r["auc"], 4),
                "Accuracy": round(r["report"]["accuracy"], 4),
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
- **Self-Training XGBoost / Random Forest / Logistic Regression:** semi-supervised classification when only part of the labels are available.
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
