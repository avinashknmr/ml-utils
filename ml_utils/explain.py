import logging, glob, os
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger("EXPLAIN")

def get_feature_impact(model_path, model_name, method='mean'):
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