import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
_logger = logging.getLogger("ML UTILS")

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, columns=None):
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame")

        if self._columns is None:
            return X
        else:
            return X[self._columns]

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, config):
        self._config = config
        self._preprocessor()

    def _preprocessor(self):
        num_cols = self._config['input']['features']['numerical']
        oe_cols = self._config['input']['features']['categorical']['ordinal']
        ohe_cols = self._config['input']['features']['categorical']['nominal']
        cat_cols = oe_cols + ohe_cols

        impute_config = self._config['preprocess']['impute']
        if impute_config['enabled']:
            num_mean_cols = [c for c in impute_config['mean'] if c in num_cols]
            num_median_cols = [c for c in impute_config['median'] if c in num_cols]
            num_mode_cols = [c for c in impute_config['mode'] if c in num_cols]
            num_const_cols = [c for c in impute_config['constant'] if c in num_cols]

            num_mean_imp = SimpleImputer(strategy='mean')
            num_median_imp = SimpleImputer(strategy='median')
            num_mode_imp = SimpleImputer(strategy='most_frequent')
            num_const_imp = SimpleImputer(strategy='constant', fill_value=0)

            num_mean_pipe = Pipeline(steps=[('selector', ColumnSelector(num_mean_cols)), ('num_mean_imp', num_mean_imp)])
            num_median_pipe = Pipeline(steps=[('selector', ColumnSelector(num_median_cols)), ('num_median_imp', num_median_imp)])
            num_mode_pipe = Pipeline(steps=[('selector', ColumnSelector(num_mode_cols)), ('num_mode_imp', num_mode_imp)])
            num_const_pipe = Pipeline(steps=[('selector', ColumnSelector(num_const_cols)), ('num_const_imp', num_const_imp)])

            num_imp = ColumnTransformer(
                transformers=[
                    ('num_mean_imp', num_mean_pipe, num_mean_cols),
                    ('num_median_imp', num_median_pipe, num_median_cols),
                    ('num_mode_imp', num_mode_pipe, num_mode_cols),
                    ('num_const_imp', num_const_pipe, num_const_cols),
                ], remainder='passthrough'
            )

            cat_mode_cols = [c for c in impute_config['mode'] if c in ohe_cols]
            cat_const_cols = [c for c in impute_config['constant'] if c in ohe_cols]

            cat_const_imp = SimpleImputer(strategy='constant', fill_value='0.Missing')
            cat_mode_imp = SimpleImputer(strategy='most_frequent')

            cat_mode_pipe = Pipeline(steps=[('selector', ColumnSelector(cat_mode_cols)), ('cat_mode_imp', cat_mode_imp)])
            cat_const_pipe = Pipeline(steps=[('selector', ColumnSelector(cat_const_cols)), ('cat_const_imp', cat_const_imp)])

            cat_imp = ColumnTransformer(
                transformers=[
                    ('cat_mode_imp', cat_mode_pipe, cat_mode_cols),
                    ('cat_const_imp', cat_const_pipe, cat_const_cols),
                ], remainder='passthrough'
            )

            ord_mode_cols = [c for c in impute_config['mode'] if c in oe_cols]
            ord_const_cols = [c for c in impute_config['constant'] if c in oe_cols]

            ord_const_imp = SimpleImputer(strategy='constant', fill_value='0.Missing')
            ord_mode_imp = SimpleImputer(strategy='most_frequent')

            ord_mode_pipe = Pipeline(steps=[('selector', ColumnSelector(ord_mode_cols)), ('ord_mode_imp', ord_mode_imp)])
            ord_const_pipe = Pipeline(steps=[('selector', ColumnSelector(ord_const_cols)), ('ord_const_imp', ord_const_imp)])

            ord_imp = ColumnTransformer(
                transformers=[
                    ('ord_mode_imp', ord_mode_pipe, ord_mode_cols),
                    ('ord_const_imp', ord_const_pipe, ord_const_cols),
                ], remainder='passthrough'
            )

        else:
            num_imp = SimpleImputer(strategy='constant', fill_value=0)
            cat_imp = SimpleImputer(strategy='constant', fill_value='0.Missing')
            ord_imp = SimpleImputer(strategy='constant', fill_value='0.Missing')

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        oe = OrdinalEncoder()

        normalize_config = self._config['preprocess']['normalize']
        if normalize_config['enabled']:
            if normalize_config['type']=='standard':
                scaler = StandardScaler()
            elif normalize_config['type']=='minmax':
                scaler = MinMaxScaler()
            else:
                raise TypeError("Normalization type can be either 'standard' or 'minmax'.")

        numerical_pipe = Pipeline(steps=[('num_imp', num_imp), ('scaler', scaler)])
        categorical_pipe = Pipeline(steps=[('cat_imp', cat_imp), ('cat_encoder', ohe)])
        ordinal_pipe = Pipeline(steps=[('ord_imp', ord_imp), ('cat_encoder', oe)])

        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numerical_pipe, num_cols),
                ('categorical', categorical_pipe, ohe_cols),
                ('ordinal', ordinal_pipe, oe_cols),
            ],
            remainder='passthrough'
        )
        self.preprocessor = preprocessor
    
    def get_feature_names_out(self):
        # Get the feature names
        feature_names = []
        for transformer_name, transformer, features in self.preprocessor.transformers_:
            try:
                # Get feature names from the transformer
                transformer_feature_names = transformer.get_feature_names(features)
                feature_names.extend(transformer_feature_names)
            except AttributeError:
                # If the transformer doesn't have get_feature_names, use the original feature names
                feature_names.extend(features)

        return feature_names
    
    def fit(self, X, y=None):
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X, out_df=False):
        if out_df:
            return pd.DataFrame(self.preprocessor.transform(X), columns=self.get_feature_names_out())
        else:
            return self.preprocessor.transform(X)
        
    def fit_transform(self, X, y=None, out_df=False, **fit_params):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X, out_df=out_df)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X, out_df=out_df)
        
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