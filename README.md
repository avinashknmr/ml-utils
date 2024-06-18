# ml-utils
Helper functions for all binary classification modelling problems including visualisations, scoring metrics - psi, ks, gini.

## Introduction


## Installation
Clone this package and run
```bash
cd path/to/cloned/dir
python setup.py
 or
pip install .
```

## Features
Complete list of features along with documentation is available under [docs](./docs/ml_utils/index.html)
1. Data Preprocessing (preprocess)
    - Normalize / Standardize
    - Impute
    - Outlier Detection & Treatment
2. Feature Engineering
    - Feature Selection
    - Feature Extraction
    - Collinearity Detection & Removal
3. Model Development (prepare_data)
    - Train Test Validation Split
    - Cross Validation Split
    - Under Sampling
    - Over Sampling
4. Model Evaluation (measure)
    - Cross Validation
    - Metrics (metrics)
        - KS, Gini
        - Decile Analysis, IV, WoE Bins
        - Accuracy, Precision, Recall, F1 Score, Confusion Matrix
        - Gain / Lift Chart, Optimal Threshold
    - Bias & Fairness Framework (bias_fairness)
        - GSI (Group Stability Index)
    - Data Drift & Monitoring (data_drift)
        - PSI, CSI
5. Model Explainability (explain)
    - Feature Importances
6. Visualization (draw)
    - ROC AUC Curve
    - Precision Recall Curve
    - CSI, IV
    - Correlation Plot
7. Model Utilities
    - Save and Load Model
    - Model Comparison
8. Advanced Utilties
    - Custom Transformers
    - Pipeline Utilities
9. Logging (logger)

## Usage
```python
from ml_utils.measure as gini
from ml_utils.draw as roc_auc_curve
```