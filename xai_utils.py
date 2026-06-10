"""XAI helper functions for the Streamlit dashboard.

This version contains a custom PDP implementation instead of relying on
sklearn.inspection.PartialDependenceDisplay. The custom implementation is more
robust with XGBoost + recent scikit-learn versions and with pandas feature names.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
from models import ALL_FEATURES_CLEAN

# Color-blind friendly Okabe-Ito colors, kept consistent with app.py
NO_FAILURE_COLOR = "#0072B2"
FAILURE_COLOR = "#D55E00"
THRESHOLD_COLOR = "#CC79A7"


# ── SHAP ──────────────────────────────────────────────────────────────────────

def _select_failure_class_shap_values(shap_values):
    """Return SHAP values for the Failure class when a classifier returns both classes.

    Some tree classifiers, especially Random Forest, return SHAP values with shape
    (samples, features, classes). For this project we explain class 1 = Failure,
    so we select the last dimension index 1. XGBoost binary models usually already
    return a 2D Explanation, so no selection is needed.
    """
    try:
        values = shap_values.values
        if getattr(values, "ndim", 0) == 3 and values.shape[2] >= 2:
            return shap_values[:, :, 1]
    except Exception:
        pass
    return shap_values


def get_shap_explainer(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)
    shap_values = _select_failure_class_shap_values(shap_values)
    return explainer, shap_values


def _clean_matplotlib_before_shap():
    """Start SHAP plots from a clean Matplotlib state.

    SHAP uses Matplotlib internally. In Streamlit, if an older Matplotlib
    figure is still active, SHAP can accidentally draw on top of it. That was
    the reason the previous SHAP plots showed unreadable/overlapping labels and
    old plot elements in the background.
    """
    plt.close("all")


def _shap_figure_height(shap_values, min_height=6.0):
    """Choose a readable figure height based on the number of displayed features."""
    try:
        n_features = len(shap_values.feature_names)
    except Exception:
        try:
            n_features = shap_values.values.shape[1]
        except Exception:
            n_features = 8
    return max(min_height, 0.65 * min(n_features, 12) + 2.5)


def _format_shap_fig(fig, left=0.34, right=0.96, top=0.92, bottom=0.16):
    """Apply margins that leave enough room for feature labels."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=10)
        ax.set_facecolor("white")
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    return fig


def plot_shap_summary(shap_values):
    """Readable SHAP beeswarm summary plot for Streamlit."""
    _clean_matplotlib_before_shap()
    height = _shap_figure_height(shap_values, min_height=7.0)
    plt.figure(figsize=(12, height), dpi=120)
    shap.summary_plot(
        shap_values,
        max_display=12,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    fig.set_size_inches(12, height, forward=True)
    return _format_shap_fig(fig, left=0.30, right=0.92, top=0.94, bottom=0.16)


def plot_shap_waterfall(shap_values, idx=0):
    """Readable SHAP waterfall plot for one instance."""
    _clean_matplotlib_before_shap()
    height = _shap_figure_height(shap_values, min_height=6.5)
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, height, forward=True)
    return _format_shap_fig(fig, left=0.36, right=0.96, top=0.93, bottom=0.18)


def plot_shap_bar(shap_values):
    """Readable global SHAP bar plot."""
    _clean_matplotlib_before_shap()
    height = _shap_figure_height(shap_values, min_height=6.5)
    shap.plots.bar(shap_values, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, height, forward=True)
    return _format_shap_fig(fig, left=0.36, right=0.96, top=0.93, bottom=0.16)


# ── LIME ──────────────────────────────────────────────────────────────────────

def get_lime_explainer(X_train):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=ALL_FEATURES_CLEAN,
        class_names=["No Failure", "Failure"],
        mode="classification",
        random_state=42,
    )
    return explainer


def plot_lime_explanation(lime_explainer, model, X_instance):
    """Create a LIME explanation for one instance.

    LIME passes numpy arrays to the prediction function. Some models, especially
    XGBoost models trained with pandas column names, can be sensitive to feature
    names. Therefore, we wrap the numpy array back into a DataFrame before
    calling predict_proba().
    """
    def predict_fn(x_as_numpy):
        x_df = pd.DataFrame(x_as_numpy, columns=ALL_FEATURES_CLEAN)
        return model.predict_proba(x_df)

    exp = lime_explainer.explain_instance(
        X_instance.values.flatten(),
        predict_fn,
        num_features=len(ALL_FEATURES_CLEAN),
    )
    fig = exp.as_pyplot_figure()

    # LIME uses green/red by default. In this dashboard we keep the class colors
    # consistent with the rest of the project:
    #   positive contribution  -> pushes the explanation toward Failure
    #   negative contribution  -> pushes the explanation toward No Failure
    # This avoids the confusing default where Failure-supporting bars can appear green.
    for ax in fig.axes:
        for patch in ax.patches:
            try:
                width = patch.get_width()
                if width >= 0:
                    patch.set_color(FAILURE_COLOR)      # pushes toward Failure
                else:
                    patch.set_color(NO_FAILURE_COLOR)   # pushes toward No Failure
            except Exception:
                pass

        ax.set_title(
            "Local LIME explanation for class: Failure\n"
            "Orange = pushes toward Failure | Blue = pushes toward No Failure",
            fontsize=13,
        )
        ax.tick_params(axis="both", labelsize=10)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig, exp


# ── PDP ───────────────────────────────────────────────────────────────────────

def _failure_probability(model, X):
    """Return probability/score for the failure class.

    For classifiers with predict_proba, this returns P(class=1). Otherwise it
    falls back to decision_function or predict.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def plot_pdp(model, X_train, feature_name):
    """Robust custom partial-dependence plot.

    This avoids sklearn's PartialDependenceDisplay compatibility problems with
    XGBoost, pandas feature names, and newer scikit-learn versions.

    Method:
    1. Select one feature.
    2. Replace that feature by several grid values for all rows.
    3. Predict failure probability for each modified dataset.
    4. Plot the average predicted failure probability against the grid value.
    """
    if feature_name not in X_train.columns:
        raise ValueError(
            f"Feature '{feature_name}' not found in X_train. Available features: {list(X_train.columns)}"
        )

    X_base = X_train.copy()
    series = X_base[feature_name]

    # For encoded categorical features, use the actual unique values.
    # For continuous features, use a percentile-based grid to avoid extreme outliers.
    unique_values = np.sort(series.dropna().unique())
    if feature_name == "Type_enc" or len(unique_values) <= 10:
        grid = unique_values
    else:
        low, high = np.percentile(series, [1, 99])
        grid = np.linspace(low, high, 50)

    pdp_values = []
    for value in grid:
        X_temp = X_base.copy()
        X_temp[feature_name] = value
        pdp_values.append(float(np.mean(_failure_probability(model, X_temp))))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, pdp_values, marker="o" if len(grid) <= 10 else None, color=FAILURE_COLOR, linewidth=2.5)
    ax.set_title(f"Partial Dependence — {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Average predicted failure probability")
    ax.grid(alpha=0.3)

    if feature_name == "Type_enc":
        ax.set_xticks(grid)
        ax.set_xticklabels([str(int(v)) for v in grid])

    plt.tight_layout()
    return fig
