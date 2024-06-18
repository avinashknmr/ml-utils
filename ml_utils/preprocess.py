import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger("ML UTILS")

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def tranform(self, X):
        return X[self.columns]

def preprocess(data):
    cat_cols = data.select_dtypes('object').columns
    oe_cols = [c for c in cat_cols if data[c].nunique()>3]
    ohe_cols = [c for c in cat_cols if c not in oe_cols]

    cat_imp = SimpleImputer(strategy='constant', fill_value='0.Missing')
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    oe = OrdinalEncoder()

    ohe_pipe = Pipeline(steps=[('selector', ColumnSelector(ohe_cols)), ('encoder', ohe)])
    oe_pipe = Pipeline(steps=[('selector', ColumnSelector(oe_cols)), ('encoder', oe)])

    cat_preprocess = FeatureUnion(
        [('ordinal_encoding', oe_pipe),
        ('onehot_encoding', ohe_pipe)]
    )

    num_cols = data.select_dtypes('number').columns
    num_imp = SimpleImputer(strategy='constant', fill_value=0)
    scaler = StandardScaler()

    numerical_transformer = Pipeline(steps=[('num_imp', num_imp), ('scaler', scaler)])
    categorical_transformer = Pipeline(steps=[('cat_imp', cat_imp), ('cat_preprocess', cat_preprocess if oe_cols else ohe)])

    preprocess = ColumnTransformer(
        transformers=[('numerical', numerical_transformer, num_cols),
        ('categorical', categorical_transformer, cat_cols)],
        remainder='passthrough', verbose_feature_names_out=False
    )

    processor = preprocess.fit(data)

    return processor