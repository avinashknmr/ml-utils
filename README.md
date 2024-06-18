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
1. Data Preprocessing
    - Normalize / Standardize
    - Impute
    - Outlier Detection & Treatment
2. Feature Engineering
    - Feature Selection
    - Feature Extraction
    - Collinearity Detection & Removal
3. Model Development
    - Train Test Split
    - Cross Validation Split
    - Under Sampling
    - Over Sampling
4. Model Evaluation
    - Cross Validation
    - Metrics
        - KS, Gini
        - Decile Analysis, IV, WoE Bins
        - Accuracy, Precision, Recall, F1 Score, Confusion Matrix
        - Gain / Lift Chart, Optimal Threshold
    - Bias & Fairness Framework
        - GSI (Group Stability Index)
    - Data Drift & Monitoring
        - PSI, CSI
5. Model Explainability
    - Feature Importances
6. Visualization
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
9. Logging (Logger)

## Usage
```python
from ml_utils.measure as gini
from ml_utils.draw as roc_auc_curve
```