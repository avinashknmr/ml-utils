import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
import logging
from . import config

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

def _preprocess(data):
    # cat_cols = config['input']['features']['categorical']
    num_cols = config['input']['features']['numerical']
    oe_cols = config['input']['features']['categorical']['ordinal']
    ohe_cols = config['input']['features']['categorical']['nominal']

    impute_config = config['preprocess']['impute']
    if impute_config['enabled']:
        cat_const_imp = SimpleImputer(strategy='constant', fill_value='0.Missing')
        cat_mode_imp = SimpleImputer(strategy='most_frequent')
        num_mean_imp = SimpleImputer(strategy='mean')
        num_median_imp = SimpleImputer(strategy='median')
        num_mode_imp = SimpleImputer(strategy='most_frequent')
        num_const_imp = SimpleImputer(strategy='constant', fill_value=0)
        impute_transformer = ColumnTransformer(
            transformers=[
                ('num_mean', num_mean_imp, impute_config['mean']),
                ('num_mean', num_median_imp, impute_config['median']),
                ('num_mean', num_mode_imp, impute_config['mode']),
                ('num_mean', num_const_imp, impute_config['constant']),
                ('categorical', cat_const_imp, cat_cols)
            ],
            remainder='passthrough', verbose_feature_names_out=False
        )

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    oe = OrdinalEncoder()

    ohe_pipe = Pipeline(steps=[('selector', ColumnSelector(ohe_cols)), ('encoder', ohe)])
    oe_pipe = Pipeline(steps=[('selector', ColumnSelector(oe_cols)), ('encoder', oe)])

    cat_preprocess = FeatureUnion(
        [('ordinal_encoding', oe_pipe),
        ('onehot_encoding', ohe_pipe)]
    )

   
    num_imp = SimpleImputer(strategy='constant', fill_value=0)
    scaler = StandardScaler()

    numerical_transformer = Pipeline(steps=[('num_imp', num_imp), ('scaler', scaler)])
    categorical_transformer = Pipeline(steps=[('cat_imp', cat_const_imp), ('cat_preprocess', cat_preprocess if oe_cols else ohe)])

    preprocess = ColumnTransformer(
        transformers=[('numerical', numerical_transformer, num_cols),
        ('categorical', categorical_transformer, cat_cols)],
        remainder='passthrough', verbose_feature_names_out=False
    )

    processor = preprocess.fit(data)

    return processor