# Literature and Methodology Notes

This file summarizes the project choices so they can be reused in the final presentation and report.

## 1. Dataset Choice

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

Reasons for choosing it:

- It was designed for predictive maintenance.
- It has interpretable process variables: temperature, rotational speed, torque, tool wear and product type.
- It contains a rare binary machine-failure target, which is suitable for anomaly/failure detection.
- It is small enough to run interactively in Streamlit.
- It is connected to previous research on explainable AI for predictive maintenance.

Important modelling decision:

- `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are not used as input features.
- These columns describe failure modes and would create target leakage if used as model inputs.
- They are used only for dashboard display and post-analysis.

## 2. Model Choice

### 2.1 Supervised predictive maintenance models

The supervised part uses the `Machine failure` label.

- **XGBoost** is the main supervised model because tree boosting performs well on tabular datasets and is compatible with SHAP/TreeSHAP explanations.
- **Random Forest** is used as an ensemble baseline.
- **Logistic Regression** is used as a simpler baseline.

This lets the group explain why the final model was chosen, instead of randomly selecting one model.

### 2.2 Semi-supervised predictive maintenance models

The semi-supervised tab uses `SelfTrainingClassifier` with the same three base models: XGBoost, Random Forest, and Logistic Regression. Only a selected fraction of the training labels is kept visible; the remaining labels are marked as unlabelled. The model then learns from the visible labels and adds confident pseudo-labels during training. This simulates industrial situations where only part of the machine data has confirmed failure labels.

### 2.3 Anomaly detection models

The project topic is explainable anomaly detection, so supervised prediction alone is not enough. The improved dashboard adds a dedicated anomaly-detection tab.

- **Isolation Forest**: classical anomaly detection method; useful for tabular data.
- **Local Outlier Factor**: neighborhood-based anomaly detection; useful as a comparison method.
- **MLP Autoencoder**: reconstruction-based anomaly detection; trained on normal samples and detects unusual samples through high reconstruction error.

The anomaly models are trained on normal samples and evaluated using the `Machine failure` label.

## 3. XAI Method Choice

### SHAP / TreeSHAP

SHAP is used for tree-based models because it provides both global and local feature-contribution explanations. This is useful for the dashboard because users can inspect overall feature importance and individual predictions.

### LIME

LIME is used as a local, model-agnostic explanation method. It explains one selected prediction by fitting a simpler interpretable model around that local sample.

### Partial Dependence Plot

PDP is used as a global explanation method to show how changing one feature affects the average predicted failure probability.

### Reconstruction/deviation explanations

For anomaly-detection methods, a simple local explanation is used:

- for the autoencoder: feature-wise reconstruction error,
- for Isolation Forest and LOF: standardized deviation from normal training behavior.

These explanations are easy to understand for non-data-science users.

## 4. Suggested Final Project Story

The project can be presented as follows:

1. We chose AI4I because it is an interpretable predictive-maintenance dataset.
2. We first explored the raw data visually.
3. We trained supervised models to predict failure using the available labels.
4. Because the project topic is anomaly detection, we added unsupervised/reconstruction-based anomaly models trained on normal behavior.
5. We evaluated all models using confusion matrix, precision, recall, F1 and AUC.
6. We used SHAP, LIME, PDP and reconstruction/deviation explanations to make the predictions understandable.
7. We integrated everything into a Streamlit dashboard following Visual Analytics principles.
8. We added pairwise relationship plots and derived engineering-oriented features such as temperature difference and mechanical power to support visual interpretation of machine operating regions.

## 5. Reference List

[1] UCI Machine Learning Repository, *AI4I 2020 Predictive Maintenance Dataset*, 2020, doi: `10.24432/C5HS5C`.

[2] S. Matzka, “Explainable Artificial Intelligence for Predictive Maintenance Applications,” in *International Conference on Artificial Intelligence for Industries (AI4I)*, 2020, doi: `10.1109/AI4I49448.2020.00023`.

[3] A. Torcianti and S. Matzka, “Explainable Artificial Intelligence for Predictive Maintenance Applications using a Local Surrogate Model,” in *2021 4th International Conference on Artificial Intelligence for Industries (AI4I)*, 2021, doi: `10.1109/AI4I51902.2021.00029`.

[4] M. T. Ribeiro, S. Singh, and C. Guestrin, “Why Should I Trust You?: Explaining the Predictions of Any Classifier,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.

[5] S. M. Lundberg et al., “From Local Explanations to Global Understanding with Explainable AI for Trees,” *Nature Machine Intelligence*, vol. 2, pp. 56–67, 2020.

[6] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.

[7] L. Cummins et al., “Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities,” arXiv:2401.07871, 2024.

[8] A. Maged, S. Haridy, and H. Shen, “Explainable Artificial Intelligence Techniques for Accurate Fault Detection and Diagnosis: A Review,” arXiv:2404.11597, 2024.

[9] L. C. Brito, G. A. Susto, J. N. Brito, and M. A. V. Duarte, “An Explainable Artificial Intelligence Approach for Unsupervised Fault Detection and Diagnosis in Rotating Machinery,” arXiv:2102.11848, 2021.
