import logging, glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import wasserstein_distance as wd
import seaborn as sns
import shap

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger("EXPLAIN")

shap.initjs()

class ShapExplainer:
    def __init__(self, preprocessor, model, X, y, params):
        self.preprocessor = preprocessor
        self.model = model
        self.X = X
        self.y = y
        self.params = params
        self.explainer, self.shap_values = self._compute_shap()
    
    def _compute_shap(self):
        self.X_transformed, _ = self.preprocessor.transform(self.X, self.y)
        model_type = self.params['model_type']
        if model_type == 'LightGBM':
            self.model.params['objective'] = 'binary_classification'
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_transformed)
        elif model_type == 'Linear':
            explainer = shap.LinearExplainer(self.model, self.X_transformed)
            shap_values = explainer.shap_values(self.X_transformed)
        elif model_type == 'Neural Network':
            explainer = shap.KernelExplainer(self.model.predict, self.X_transformed)
            shap_values = explainer.shap_values(self.X_transformed, nsamples=100)
        else:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_transformed)

        # fix for Random Forest
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            explainer.expected_value = explainer.expected_value[1]
            shap_values = shap_values[1]

        # explainer = shap.TreeExplainer(learner.model)
        # shap_values = explainer.shap_values(self.X_transformed)
        # shap_values_df=pd.DataFrame(shap_values, columns=X_test.columns)
        return explainer, shap_values

    def get_global_explainability(self):
        """
        Returns SHAP Global Explainability bar chart (`shap.summary_plot`)
        """
        return shap.summary_plot(self.shap_values, self.X, plot_type="bar", max_display=100)

    def get_local_explainability(self, datapoint=1, features_display=20):
        """
        Returns SHAP Local Explainability (`shap.force_plot`) for given datapoint.
        """
        return shap.decision_plot(self.explainer.expected_value, self.shap_values[datapoint], features=self.X.iloc[datapoint], link='logit', highlight=0, feature_display_range=slice(None, -1*features_display-1, -1))

    def get_feature_influence(self, column):
        """
        Returns SHAP Feature Influence for a given column.
        """
        shap_values_df=pd.DataFrame(self.shap_values, columns=self.X.columns)
        return px.scatter(x=self.X_transformed[column], y=shap_values_df[column], color=self.y.astype('str'), marginal_x='box', marginal_y='box', labels={'x': 'Feature Values', 'y': 'Shap Values', 'color': self.y.name})

    def get_partial_dependence(self, column):
        """
        Returns Partial Dependence Plot for a given model for a given column.
        """
        return shap.dependence_plot(column, self.shap_values, self.X_transformed)

class FairnessAnalyzer:
    def __init__(self, data, priv_category, priv_value, target_label, priv_target_value, ignore_cols=None):
        self.data = data
        self.priv_category = priv_category
        self.priv_value = priv_value
        self.target_label = target_label
        self.priv_target_value = priv_target_value
        self._prepare_data(ignore_cols=ignore_cols)

    def _prepare_data(self, ignore_cols=None):
        """
        Prepare dataset for bias mitigation.
        
        Args:
            data (pandas dataframe): Data to fix (for fairness)
            priv_category (string): Column name that contains the privileged value (e.g. Race, Gender, etc)
            priv_value (string): Value or type in the column that denotes the privileged attribute (e.g. White, Male, etc)
            target_label (string): Column name of target variable (e.g. income, loan score, etc)
            priv_target_value (string): Value in target that favors the privileged (e.g. High income, favorable loan score, credit acceptance, etc). Must be boolean (so if target is numeric, convert to categorical by thresholding before processing.)
            ignore_cols, optional (list of string): List of columns to exclude from bias assessment and modeling.
        
        Returns:
            data_priv (standard Dataset): Dataset prepared by aif360 for processing
            encoders (dict): dictionary of encoding models
            numerical_features (list): List of numerical columns
            categorical_features (list) List of categorical columns
        """
        
        if ignore_cols:
            self.data = self.data.drop(ignore_cols, axis=1)
        else:
            pass
        
        # Get categorical features
        self.categorical_features = self.data.columns[self.data.dtypes == 'object']
        data_encoded = self.data.copy()
        
        # Store categorical names and encoders
        categorical_names = {}
        self.encoders = {}

        # Use Label Encoder for categorical columns (including target column)
        for feature in self.categorical_features:
            le = LabelEncoder()
            le.fit(data_encoded[feature])

            data_encoded[feature] = le.transform(data_encoded[feature])

            categorical_names[feature] = le.classes_
            self.encoders[feature] = le
            
        # Scale numeric columns
        self.numerical_features = [c for c in self.data.columns.values if c not in self.categorical_features]

        for feature in self.numerical_features:
            val = data_encoded[feature].values[:, np.newaxis]
            mms = MinMaxScaler().fit(val)
            data_encoded[feature] = mms.transform(val)
            self.encoders[feature] = mms

        data_encoded = data_encoded.astype(float)
        
        privileged_class = np.where(categorical_names[self.priv_category]==self.priv_value)[0]
        encoded_target_label = np.where(categorical_names[self.target_label]==self.priv_target_value)[0]
        
        self.data_priv = StandardDataset(data_encoded, 
                                label_name=self.target_label, 
                                favorable_classes=encoded_target_label, 
                                protected_attribute_names=[self.priv_category], 
                                privileged_classes=[privileged_class])

    def _get_fairness_metrics(self, pred, pred_is_dataset=False):
        """
        Measure fairness metrics.
        
        Args:
            dataset (pandas dataframe): Dataset
            pred (array): Model predictions
            pred_is_dataset, optional (bool): True if prediction is already part of the dataset, column name 'labels'.
        
        Returns:
            fair_metrics: Fairness metrics.
        """
        if pred_is_dataset:
            dataset_pred = pred
        else:
            dataset_pred = self.data.copy()
            dataset_pred.labels = pred
        
        cols = [
            'statistical_parity_difference',
            'equal_opportunity_difference',
            'generalized_entropy_index',
            'average_abs_odds_difference',
            'disparate_impact',
            'theil_index',
            'accuracy'
        ]
        
        fair_metrics = pd.DataFrame(columns=cols)
        
        for attr in dataset_pred.protected_attribute_names:
            idx = dataset_pred.protected_attribute_names.index(attr)
            privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
            unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]
            
            classified_metric = ClassificationMetric(self.data, 
                                                        dataset_pred,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)

            metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)

            row = pd.DataFrame([[classified_metric.statistical_parity_difference(),
                                    classified_metric.equal_opportunity_difference(),
                                    classified_metric.generalized_entropy_index(alpha=2),
                                    classified_metric.average_abs_odds_difference(),
                                    classified_metric.disparate_impact(),
                                    classified_metric.theil_index(),
                                    classified_metric.accuracy()
                                    ]],
                            columns  = cols,
                            index = [attr]
                            )
            fair_metrics = fair_metrics.append(row)    
        
        fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)
            
        return fair_metrics

    def show_fairness(self):
        """
        Show biases in the data.
        
        Args:
            data (pandas dataframe): Data to fix (for fairness)
            priv_category (string): Column name that contains the privileged value (e.g. Race, Gender, etc)
            priv_value (string): Value or type in the column that denotes the privileged attribute (e.g. White, Male, etc)
            target_label (string): Column name of target variable (e.g. income, loan score, etc)
            priv_target_value (string): Value in target that favors the privileged (e.g. High income, favorable loan score, credit acceptance, etc).
                                        Must be boolean (so if target is numeric, convert to categorical by thresholding before processing.)
            ignore_cols, optional (list of string): List of columns to exclude from bias assessment and modeling.
        
        Returns:
            Bias analysis chart.
        """
        # data_orig, encoders, numerical_features, categorical_features

        np.random.seed(42)
        data_priv_train, data_priv_test = self.data_priv.split([0.7], shuffle=True)
        
        # Train and save the models
        rf_orig = RandomForestClassifier().fit(data_priv_train.features, 
                            data_priv_train.labels.ravel(), 
                            sample_weight=data_priv_train.instance_weights)

        pred = rf_orig.predict(data_priv_test.features)
        fair = self._get_fair_metrics(data_priv_test, pred)

        # gsi_df = gsi(data, priv_category, priv_value, 'Prediction').gsi.sum()
        
        return fair

    def wasserstein_distance(self, prediction_col='positive_probability', logit=True):
        """
        Compute wasserstein distance for priviledged value vs others in priviledged category using probabilities.
        
        Args:
            data (pandas dataframe): Data to fix (for fairness)
            priv_category (string): Column name that contains the privileged value (e.g. Race, Gender, etc)
            priv_value (string): Value or type in the column that denotes the privileged attribute (e.g. White, Male, etc)
            prediction_col (string): Column name of prediction variable (e.g. positive_probability, prediction_1, etc)
            logit (bool): if logit score to be used for calculation else probabilities will be used
        
        Returns:
            Wasserstein distance.
        """
        df = self.data.copy()
        col = prediction_col
        if logit:
            col = 'logit'
            df[col] = df[prediction_col].apply(lambda x: np.log(x/(1-x)))
        protected = df.loc[df[self.priv_category]==self.priv_value, col].tolist()
        unprotected = df.loc[df[self.priv_category]!=self.priv_value, col].tolist()
        return wd(protected, unprotected)

    def wasserstein_distance_plot(self, prediction_col='positive_probability', logit=True):
        """
        Density comparison chart for priviledged value vs others in priviledged category using probabilities.
        
        Args:
            data (pandas dataframe): Data to fix (for fairness)
            priv_category (string): Column name that contains the privileged value (e.g. Race, Gender, etc)
            priv_value (string): Value or type in the column that denotes the privileged attribute (e.g. White, Male, etc)
            prediction_col (string): Column name of prediction variable (e.g. positive_probability, prediction_1, etc)
            logit (bool): if logit score to be used for calculation else probabilities will be used
        
        Returns:
            Density comparison chart.
        """
        df = self.data.copy()
        xvar = prediction_col
        df[self.priv_category] = df[self.priv_category].apply(lambda x: self.priv_value if x==self.priv_value else 'Rest')
        if logit:
            xvar = 'logit'
            df[xvar] = df[prediction_col].apply(lambda x: np.log(x/(1-x)))
        sns.displot(data=df, x=xvar, hue=self.priv_category, kind='kde', fill=True, aspect=3)
        plt.title('Model Score Disparity')
        plt.xlabel('Model Score')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.close()
        return plt
    
def get_feature_impact(model_path, model_name, method='mean'): # irrelevant in current context
    """
    Args:
        model_path: Path to AutoML results. Ex - AutoML_1
        model_name: Name of the model. Ex - 6_Xgboost
        method: method to be used for computing feature impact. allowed values - mean, shap
    Returns:
        Mean/Shap feature impact dataset with relative impact that can be used for plotting.
    """
    try:
        if method=='mean':
            feature_importance_files = [f for f in glob.glob(os.path.join(model_path, model_name, 'learner_*_importance.csv')) if 'shap' not in f]
        elif method=='shap':
            feature_importance_files = [f for f in glob.glob(os.path.join(model_path, model_name, 'learner_*_importance.csv')) if 'shap' in f]
        dfs = []
        for f in feature_importance_files:
            df = pd.read_csv(f)
            dfs.append(df)
        feature_importance = pd.concat(dfs)
        feature_impact = feature_importance.groupby('feature').mean()
        feature_impact['relative_impact'] = (feature_impact[method+'_importance'] - feature_impact[method+'_importance'].min()) / (feature_impact[method+'_importance'].max() - feature_impact[method+'_importance'].min())
        feature_impact = feature_impact.reset_index().sort_values(by='relative_impact')
        return feature_impact
    except Exception as e:
        logger.error(f"Error occured - {e}")
        raise