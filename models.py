"""Model utilities for the Visual Analytics project.

The project combines two modelling perspectives:
1. Supervised predictive maintenance: XGBoost / Random Forest / Logistic Regression
   learn from the Machine failure label.
2. Anomaly detection: Isolation Forest, Local Outlier Factor and an MLP-based
   reconstruction autoencoder are trained only on normal samples and then evaluated
   against the Machine failure label.

Important leakage rule:
TWF, HDF, PWF, OSF and RNF are failure-type labels. They are useful for explanation
and dashboard display, but they must not be used as input features for model training.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Raw machine/process features used for modelling.
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# XGBoost and SHAP are easier to use with clean feature names.
FEATURES_CLEAN = [
    "Air_temp_K",
    "Process_temp_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
]
RENAME_MAP = dict(zip(FEATURES, FEATURES_CLEAN))
ALL_FEATURES_CLEAN = FEATURES_CLEAN + ["Type_enc"]


def load_data(path: str = "ai4i2020.csv") -> pd.DataFrame:
    """Load dataset and add an encoded machine-type feature.

    We keep the original Type column for display and add Type_enc for modelling.
    Failure-mode labels are kept in the dataframe but are not used by get_xy().
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    le = LabelEncoder()
    df["Type_enc"] = le.fit_transform(df["Type"])
    return df


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return leakage-safe model inputs and target."""
    X = df[FEATURES + ["Type_enc"]].rename(columns=RENAME_MAP)
    y = df[TARGET].astype(int)
    return X, y


def split_xy(df: pd.DataFrame, random_state: int = 42, test_size: float = 0.2):
    X, y = get_xy(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _metrics_dict(y_true, y_pred, y_score=None) -> dict:
    """Return common metrics in a dashboard-friendly dictionary."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            out["auc"] = np.nan
    else:
        out["auc"] = np.nan
    return out


def _scale_pos_weight(y_train: pd.Series) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return max(1.0, neg / max(pos, 1))


# ---------------------------------------------------------------------------
# Supervised models
# ---------------------------------------------------------------------------

def train_supervised(df: pd.DataFrame, random_state: int = 42):
    """Train the main supervised model: XGBoost.

    Class imbalance is handled with scale_pos_weight instead of SMOTE. This avoids
    synthetic samples for Type_enc and keeps the methodology easier to defend.
    """
    X_train, X_test, y_train, y_test = split_xy(df, random_state=random_state)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        scale_pos_weight=_scale_pos_weight(y_train),
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    return model, X_train, X_test, y_train, y_test, y_pred, y_prob, report, auc, cm


def train_all_models(df: pd.DataFrame, random_state: int = 42):
    """Train supervised models for comparison."""
    X_train, X_test, y_train, y_test = split_xy(df, random_state=random_state)

    models = {
        # Same XGBoost configuration as train_supervised().
        # This keeps the XGBoost metrics consistent between the
        # "Supervised ML" tab and the "Model Comparison" tab.
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            scale_pos_weight=_scale_pos_weight(y_train),
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
            C=1.0,
        ),
    }

    results = {}
    for name, m in models.items():
        if name == "Logistic Regression":
            scaler = StandardScaler()
            Xr = scaler.fit_transform(X_train)
            Xt = scaler.transform(X_test)
        else:
            scaler = None
            Xr, Xt = X_train, X_test

        m.fit(Xr, y_train)
        y_pred = m.predict(Xt)
        y_prob = m.predict_proba(Xt)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "model": m,
            "scaler": scaler,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": report,
            "auc": auc,
            "cm": cm,
            "X_test": Xt,
        }

    return results, X_train, X_test, y_test


def _visible_label_mask(y_train: pd.Series, labeled_fraction: float, random_state: int = 42) -> np.ndarray:
    """Create a stratified mask of labels that remain visible.

    This avoids an unlucky situation where the visible labelled subset contains only
    one class. The hidden labels are later set to -1 for SelfTrainingClassifier.
    """
    rng = np.random.RandomState(random_state)
    visible = np.zeros(len(y_train), dtype=bool)
    y_arr = y_train.to_numpy()

    for cls in np.unique(y_arr):
        idx = np.where(y_arr == cls)[0]
        n_visible = max(1, int(round(len(idx) * labeled_fraction)))
        chosen = rng.choice(idx, size=n_visible, replace=False)
        visible[chosen] = True

    return visible


def train_semi_supervised_all(df: pd.DataFrame, labeled_fraction: float = 0.3, random_state: int = 42):
    """Train semi-supervised self-training versions of three supervised models.

    Self-training is a semi-supervised wrapper:
    - keep only a fraction of training labels visible;
    - mark the remaining labels as -1 (unlabelled);
    - train a base supervised model;
    - add confident pseudo-labels iteratively.

    The three base models mirror the supervised comparison tab:
    XGBoost, Random Forest, and Logistic Regression.
    """
    X_train, X_test, y_train, y_test = split_xy(df, random_state=random_state)

    visible_mask = _visible_label_mask(y_train, labeled_fraction, random_state=random_state)
    y_semi = y_train.copy().values.astype(int)
    y_semi[~visible_mask] = -1

    base_models = {
        "Self-Training XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            scale_pos_weight=_scale_pos_weight(y_train[visible_mask]),
            random_state=random_state,
        ),
        "Self-Training Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Self-Training Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
            C=1.0,
        ),
    }

    results = {}
    for name, base in base_models.items():
        if "Logistic Regression" in name:
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X_train)
            X_eval = scaler.transform(X_test)
        else:
            scaler = None
            X_fit = X_train
            X_eval = X_test

        semi_model = SelfTrainingClassifier(base, threshold=0.80, max_iter=10, verbose=False)
        semi_model.fit(X_fit, y_semi)

        y_pred = semi_model.predict(X_eval)
        y_prob = semi_model.predict_proba(X_eval)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": semi_model,
            "base_model": base.__class__.__name__,
            "scaler": scaler,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": report,
            "auc": auc,
            "cm": cm,
            "X_test": X_eval,
        }

    meta = {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "labeled_fraction": labeled_fraction,
        "visible_labels": int(visible_mask.sum()),
        "hidden_labels": int((~visible_mask).sum()),
        "visible_failures": int(((y_train.to_numpy() == 1) & visible_mask).sum()),
        "visible_normals": int(((y_train.to_numpy() == 0) & visible_mask).sum()),
    }
    return results, meta


def train_semi_supervised(df: pd.DataFrame, labeled_fraction: float = 0.3, random_state: int = 42):
    """Backward-compatible single semi-supervised model.

    Returns the Random Forest self-training result in the old tuple format.
    The dashboard now uses train_semi_supervised_all().
    """
    results, meta = train_semi_supervised_all(df, labeled_fraction, random_state)
    r = results["Self-Training Random Forest"]
    return (
        r["model"],
        meta["X_test"],
        meta["y_test"],
        r["y_pred"],
        r["y_prob"],
        r["report"],
        r["auc"],
        r["cm"],
        labeled_fraction,
    )

# ---------------------------------------------------------------------------
# Anomaly detection models
# ---------------------------------------------------------------------------

def _fit_threshold(train_normal_scores: np.ndarray, failure_rate: float) -> float:
    """Set threshold from normal training scores.

    If failure rate is around 3.4%, the threshold is around the 96.6th percentile
    of the normal training-score distribution. This is a simple unsupervised
    thresholding rule and can be adjusted in the presentation.
    """
    percentile = 100.0 * (1.0 - failure_rate)
    percentile = float(np.clip(percentile, 90.0, 99.5))
    return float(np.percentile(train_normal_scores, percentile))


def _autoencoder_scores(model: MLPRegressor, X: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(X)
    return np.mean((X - reconstructed) ** 2, axis=1)


def train_anomaly_models(df: pd.DataFrame, random_state: int = 42):
    """Train three anomaly-detection models on normal samples only.

    Returns a dictionary with Isolation Forest, LOF and a small MLP reconstruction
    autoencoder. All scores are oriented as: higher score = more anomalous.
    """
    X_train, X_test, y_train, y_test = split_xy(df, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    normal_mask = (y_train.to_numpy() == 0)
    X_train_normal = X_train_scaled[normal_mask]
    failure_rate = float(y_train.mean())
    contamination = float(np.clip(failure_rate, 0.01, 0.10))

    methods = {}

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train_normal)
    train_scores = -iso.decision_function(X_train_normal)
    test_scores = -iso.decision_function(X_test_scaled)
    threshold = _fit_threshold(train_scores, failure_rate)
    y_pred = (test_scores > threshold).astype(int)
    methods["Isolation Forest"] = {
        "model": iso,
        "score": test_scores,
        "threshold": threshold,
        "y_pred": y_pred,
        "metrics": _metrics_dict(y_test, y_pred, test_scores),
        "explanation_type": "feature deviation from normal training mean",
    }

    # Local Outlier Factor in novelty mode
    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=contamination,
        novelty=True,
    )
    lof.fit(X_train_normal)
    train_scores = -lof.score_samples(X_train_normal)
    test_scores = -lof.score_samples(X_test_scaled)
    threshold = _fit_threshold(train_scores, failure_rate)
    y_pred = (test_scores > threshold).astype(int)
    methods["Local Outlier Factor"] = {
        "model": lof,
        "score": test_scores,
        "threshold": threshold,
        "y_pred": y_pred,
        "metrics": _metrics_dict(y_test, y_pred, test_scores),
        "explanation_type": "feature deviation from normal training mean",
    }

    # MLP reconstruction autoencoder using scikit-learn to avoid heavy dashboard dependencies.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ae = MLPRegressor(
            hidden_layer_sizes=(8, 4, 8),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=180,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        )
        ae.fit(X_train_normal, X_train_normal)

    train_scores = _autoencoder_scores(ae, X_train_normal)
    test_scores = _autoencoder_scores(ae, X_test_scaled)
    threshold = _fit_threshold(train_scores, failure_rate)
    y_pred = (test_scores > threshold).astype(int)
    methods["MLP Autoencoder"] = {
        "model": ae,
        "score": test_scores,
        "threshold": threshold,
        "y_pred": y_pred,
        "metrics": _metrics_dict(y_test, y_pred, test_scores),
        "explanation_type": "feature-wise reconstruction error",
    }

    meta = {
        "X_train": X_train,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": ALL_FEATURES_CLEAN,
        "normal_mean_scaled": X_train_normal.mean(axis=0),
        "failure_rate_train": failure_rate,
    }
    return methods, meta


def anomaly_local_explanation(method_result: dict, meta: dict, sample_idx: int) -> pd.DataFrame:
    """Local explanation for anomaly-detection models.

    For the MLP autoencoder, explanation = feature-wise reconstruction error.
    For Isolation Forest and LOF, explanation = absolute deviation from the normal
    training mean in standardized feature space.
    """
    X_sample = meta["X_test_scaled"][sample_idx : sample_idx + 1]
    feature_names = meta["feature_names"]
    model = method_result["model"]

    if method_result["explanation_type"] == "feature-wise reconstruction error":
        reconstructed = model.predict(X_sample)[0]
        original = X_sample[0]
        values = (original - reconstructed) ** 2
        df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Explanation value": values,
                "Meaning": "squared reconstruction error",
            }
        )
    else:
        original = X_sample[0]
        normal_mean = meta["normal_mean_scaled"]
        values = np.abs(original - normal_mean)
        df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Explanation value": values,
                "Meaning": "absolute standardized deviation from normal mean",
            }
        )

    return df.sort_values("Explanation value", ascending=False).reset_index(drop=True)
