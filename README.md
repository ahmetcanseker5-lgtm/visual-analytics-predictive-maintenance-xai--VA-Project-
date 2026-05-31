# Interactive Explainable Predictive Maintenance Dashboard

Visual Analytics course project — Hochschule Aalen, Summer Semester 2026.

This project is an interactive end-to-end dashboard for **explainable predictive maintenance** using the **AI4I 2020 Predictive Maintenance Dataset**. It combines visual analytics, supervised machine learning, anomaly detection, and XAI methods.

## Project Goal

The project aims to answer:

> Can we detect machine failures/anomalies from process data and explain why a model predicts a failure or anomaly?

The dashboard follows the Visual Analytics idea of showing:

1. raw input data,
2. model prediction or anomaly score,
3. explanation / XAI output,
4. evaluation and comparison of models.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**  
UCI Machine Learning Repository, DOI: `10.24432/C5HS5C`

The dataset contains 10,000 machine/process records with features such as:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Product type (`L`, `M`, `H`)
- Machine failure target
- Failure-mode labels: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

## Important Leakage Rule

The failure-mode columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are **not used as model inputs** because they are target-side information. Using them as features would make the model cheat.

They are only used for dashboard display and interpretation.

## Dashboard Tabs

| Tab | Purpose |
|---|---|
| Data Explorer | Raw data, failure rate, failure-mode overview, filters |
| Visual Analytics | Correlation heatmap, box plots, density plots, scatter plots, t-SNE |
| Supervised ML | XGBoost supervised failure prediction |
| Anomaly Detection | Isolation Forest, LOF, and MLP Autoencoder trained on normal samples |
| Semi-Supervised | Self-training model with limited labelled data |
| XAI Explanations | SHAP, LIME, and Partial Dependence Plots for XGBoost |
| Model Comparison | XGBoost vs Random Forest vs Logistic Regression |
| Methodology & References | Literature-backed justification of dataset, methods, and XAI choices |

## Models Used

### Supervised Models

These models learn from the `Machine failure` label:

- **XGBoost** — strong tree-boosting model for tabular data
- **Random Forest** — ensemble baseline
- **Logistic Regression** — simpler interpretable baseline

### Anomaly Detection Models

These models are trained mainly on normal machine samples and then evaluated against the `Machine failure` label:

- **Isolation Forest** — isolates unusual samples through random tree partitions
- **Local Outlier Factor (LOF)** — detects local density deviations
- **MLP Autoencoder** — reconstructs normal samples; high reconstruction error indicates anomaly

## XAI Methods Used

| XAI method | Scope | Purpose |
|---|---|---|
| SHAP / TreeSHAP | local + global | Feature contribution explanations for tree models |
| LIME | local | Single-instance explanation using a local surrogate model |
| PDP | global | Average marginal effect of one feature on predicted failure probability |
| Reconstruction/deviation explanation | local | Explanation for anomaly-detection models |

## Setup

```bash
# recommended: Python 3.11
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at:

```text
http://localhost:8501
```

## Project Structure

```text
VA-Proj-main/
├── app.py                         # Main Streamlit dashboard
├── models.py                      # Supervised, semi-supervised and anomaly models
├── xai_utils.py                   # SHAP, LIME, PDP helper functions
├── ai4i2020.csv                   # Dataset
├── requirements.txt               # Required Python packages
├── LITERATURE_AND_METHODOLOGY.md  # Method and reference explanation
└── README.md
```

## References

[1] UCI Machine Learning Repository, *AI4I 2020 Predictive Maintenance Dataset*, 2020, doi: `10.24432/C5HS5C`.

[2] S. Matzka, “Explainable Artificial Intelligence for Predictive Maintenance Applications,” in *International Conference on Artificial Intelligence for Industries (AI4I)*, 2020, doi: `10.1109/AI4I49448.2020.00023`.

[3] A. Torcianti and S. Matzka, “Explainable Artificial Intelligence for Predictive Maintenance Applications using a Local Surrogate Model,” in *2021 4th International Conference on Artificial Intelligence for Industries (AI4I)*, 2021, doi: `10.1109/AI4I51902.2021.00029`.

[4] M. T. Ribeiro, S. Singh, and C. Guestrin, “Why Should I Trust You?: Explaining the Predictions of Any Classifier,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.

[5] S. M. Lundberg et al., “From Local Explanations to Global Understanding with Explainable AI for Trees,” *Nature Machine Intelligence*, vol. 2, pp. 56–67, 2020.

[6] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.

[7] L. Cummins et al., “Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities,” arXiv:2401.07871, 2024.

[8] A. Maged, S. Haridy, and H. Shen, “Explainable Artificial Intelligence Techniques for Accurate Fault Detection and Diagnosis: A Review,” arXiv:2404.11597, 2024.

[9] L. C. Brito, G. A. Susto, J. N. Brito, and M. A. V. Duarte, “An Explainable Artificial Intelligence Approach for Unsupervised Fault Detection and Diagnosis in Rotating Machinery,” arXiv:2102.11848, 2021.
