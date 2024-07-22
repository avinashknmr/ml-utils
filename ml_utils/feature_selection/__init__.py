"""
Helper functions to calculate metrics like KS, Gini, Precision, Recall, F1 Score,
Population Stability Index, Charecteristic Stability Index, Confusion Matrix,
Chi Square with IV (Informational Value)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from ..explore_data import correlated_cols

from ._woe_binning import woe_binning, woe_binning_2, woe_binning_3

from loguru import logger

def woe_bins(df, var_name, resp_name, suffix='_dev', var_cuts=None):
    """
    Returns a pandas dataframe, var_cuts after creating bins.
    Returns:
        df: dataframe has var_cuts_string, total, responders, non_responders, var_name (with _dev or _oot suffix)
        var_cuts: list of Interval items to be used on oot file.
    """ 
    df1 = df[[resp_name, var_name]]
    if (np.issubdtype(df1[var_name].dtype,np.number)):
        n = df1[var_name].nunique()
        if var_cuts is None:
            suffix='_dev'
            var_cuts = woe_binning_3(df1,resp_name,var_name,0.05,0.00001,0,50,'bad','good')
            var_cuts = list(set(var_cuts))
            var_cuts.sort()
        df1["var_binned"] = pd.cut(df1[var_name], var_cuts, right=True, labels = None,retbins=False, precision=10, include_lowest=False)
        var_min = float(df1[var_name].min())
        var_max = float(df1[var_name].max())
        summ_df = df1.groupby('var_binned')[resp_name].agg(['count','sum']).reset_index()
        summ_df['delta'] = summ_df['count'] - summ_df['sum']
        summ_df['var_name'] = var_name
        summ_df.columns = ['var_cuts','total'+suffix,'responders'+suffix,'non_responders'+suffix,'var_name']
        summ_df['var_cuts_string'+suffix] = summ_df.var_cuts.apply(lambda x: str(x.left if x.left!=-np.inf else var_min)+' To '+str(x.right if x.right!=np.inf else var_max))
    else:
        df1[var_name].fillna('Blank', inplace=True)
        summ_df = df1.groupby(var_name)[resp_name].agg(['count','sum']).reset_index()
        summ_df['delta'] = summ_df['count'] - summ_df['sum']
        summ_df['var_name'] = var_name
        summ_df.columns = ['var_cuts_string'+suffix,'total'+suffix,'responders'+suffix,'non_responders'+suffix,'var_name']
        summ_df['var_cuts'] = summ_df['var_cuts_string'+suffix]
    return summ_df[summ_df['total'+suffix]!=0], var_cuts

def iv_var(df, var_name, resp_name, var_cuts=None):
    """Returns IV dataframe and IV value of a given variable"""
    suffix = '_dev' if var_cuts is None else '_oot'
    iv_df, _ = iv(df, var_name, resp_name, var_cuts)
    return iv_df, float(iv_df['iv'+suffix].sum())

def iv(df, var_list, resp_name, var_cuts=None):
    """
    Returns a pandas dataframe with calculated fields - resp_rate, perc_dist, perc_non_resp,
    perc_resp, raw_odds, ln_odds, iv, exp_resp, exp_non_resp, chi_square
    """
    dfs = []
    cuts = {}
    for var_name in var_list:
        if var_cuts is None:
            suffix = '_dev'
            summ_df, cut = woe_bins(df, var_name, resp_name, '_dev')
        else:
            suffix = '_oot'
            summ_df, cut = woe_bins(df, var_name, resp_name, '_oot', var_cuts[var_name])
        dfs.append(summ_df)
        cuts[var_name] = cut
    idf = pd.concat(dfs, axis=0)
    idf['resp_rate'+suffix] = (idf['responders'+suffix]*100)/idf['total'+suffix]
    idf['perc_dist'+suffix] = (idf['total'+suffix]*100)/idf.groupby('var_name')['total'+suffix].transform('sum')
    idf['perc_non_resp'+suffix] = (idf['non_responders'+suffix]*100)/idf.groupby('var_name')['non_responders'+suffix].transform('sum')
    idf['perc_resp'+suffix] = (idf['responders'+suffix]*100)/idf.groupby('var_name')['responders'+suffix].transform('sum')
    idf['raw_odds'+suffix] = idf.apply(lambda r: 0 if r['perc_resp'+suffix]==0 else r['perc_non_resp'+suffix]/r['perc_resp'+suffix], axis=1)
    idf['ln_odds'+suffix] = idf['raw_odds'+suffix].apply(lambda x: 0 if abs(np.log(x))==np.inf else np.log(x))
    idf['iv'+suffix] = (idf['perc_non_resp'+suffix]-idf['perc_resp'+suffix])*idf['ln_odds'+suffix]/100
    idf['exp_resp'+suffix] = idf['total'+suffix]*idf.groupby('var_name')['responders'+suffix].transform('sum')/idf.groupby('var_name')['total'+suffix].transform('sum')
    idf['exp_non_resp'+suffix] = idf['total'+suffix]*idf.groupby('var_name')['non_responders'+suffix].transform('sum')/idf.groupby('var_name')['total'+suffix].transform('sum')
    idf['chi_square'+suffix] = (((idf['responders'+suffix]-idf['exp_resp'+suffix])**2)/idf['exp_resp'+suffix])+(((idf['non_responders'+suffix]-idf['exp_non_resp'+suffix])**2)/idf['exp_non_resp'+suffix])
    return idf, cuts


def _quick_psi(dev, val):
    """Calculates PSI from 2 arrays - dev and val"""
    try:
        return sum([(a-b)*np.log(a/b) for (a,b) in zip(dev,val)])
    except:
        return -99.0

def psi(dev, oot, target='positive_probability', n_bins=10):
    """
    Returns a pandas dataframe with **psi** column (Population Stability Index) after creating 10 deciles.
    Code includes creating score calculation using **round(500-30xlog(100x(p/(1-p))), 0)** where p is probability.
    We need to pass both dev and oot at sametime to apply same bins created on dev dataframe.
    """
    dev['score'] = dev[target].apply(lambda x: round(500-30*np.log2(100*(x/(1-x))), 0))
    oot['score'] = oot[target].apply(lambda x: round(500-30*np.log2(100*(x/(1-x))), 0))

    _, bins = pd.qcut(dev.score, n_bins, retbins=True, precision=0)
    bins = [int(i) if abs(i)!=np.inf else i for i in bins]
    dev['bins'] = pd.cut(dev.score, bins)
    oot['bins'] = pd.cut(oot.score, bins)

    dev_bins = dev.bins.value_counts(sort=False, normalize=True)
    oot_bins = oot.bins.value_counts(sort=False, normalize=True)

    psi_ = pd.concat([dev_bins, oot_bins], axis=1)
    psi_.columns=['dev', 'oot']
    psi_['psi'] = (psi_.dev-psi_.oot)*np.log(psi_.dev/psi_.oot)
    return psi_

def csi(dev_df, oot_df, var_list, resp_name):
    """
    Returns a pandas dataframe with **csi, csi_var, perc_csi** columns (Charecteristic Stability Index) calculated based on both dev and oot dfataframes.
    """
    dev, var_cuts = iv(dev_df, var_list, resp_name)

    oot, _ = iv(oot_df, var_list, resp_name, var_cuts)

    final = pd.merge(dev, oot, how='left', on=['var_name', 'var_cuts'], suffixes=['_dev', '_oot'])

    final['csi'] = ((final['perc_dist_dev']-final['perc_dist_oot'])/100)*np.log(final['perc_dist_dev']/final['perc_dist_oot'])
    final['csi_var'] = final.groupby('var_name')['csi'].transform('sum')
    final['perc_csi'] = (100*final.groupby('var_name')['csi'].transform('cumsum'))/final.groupby('var_name')['csi'].transform('sum')
    return final

def get_multivariate_feature_importances(data, labels, eval_metric, early_stopping=True, n_iterations=10):
    """
    Get feature importances by building a LightGBM model. All features are considered while building model and feature importance is feature importance of LightGBM model.
    Also computes normalized and relative importances.
    """
    if early_stopping and eval_metric is None:
        raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                            "l2" for regression.""")
            
    if labels is None:
        raise ValueError("No training labels provided.")
    
    # One hot encoding
    base_features = data.columns
    features = pd.get_dummies(data, prefix_sep='|', drop_first=True)
    one_hot_features = [column for column in features.columns if column not in base_features]

    # Add one hot encoded data to original data
    data_all = pd.concat([features[one_hot_features], data], axis = 1)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np array
    features = np.array(features)
    labels = np.array(labels).reshape((-1, ))

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Iterate through each fold
    for _ in range(n_iterations):
        
        model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)
            
        # If training using early stopping need a validation set
        if early_stopping:
            
            train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.15, random_state=1234)

            # Train the model with early stopping
            model.fit(train_features, train_labels, eval_metric = eval_metric,
                        eval_set = [(valid_features, valid_labels)],
                        early_stopping_rounds = 100, verbose = 0)
            
        else:
            model.fit(features, labels)

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / n_iterations

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Sort features according to importance
    feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

    # Normalize the feature importances to add up to one
    imp = feature_importances['importance']
    feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['relative_importance'] = (imp - imp.min()) / (imp.max() - imp.min())
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

    return feature_importances

def get_univariate_feature_importances(data, labels, eval_metric, early_stopping=True, n_iterations=10):
    """
    Get feature importances by building a LightGBM model. Models are created with one feature at a time and feature importance is eval metric used for the LightGBM model.
    This is time consuming process if there are too many features, as those many models will be developed to compute importances and this is the reason to choose LightGBM, as it is fast and accurate.
    Also computes normalized and relative importances.
    """
    if early_stopping and eval_metric is None:
        raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                            "l2" for regression.""")
            
    if labels is None:
        raise ValueError("No training labels provided.")
    
    base_features = data.columns
    # Empty array for feature importances
    feature_importance_values = {}
    for feature in base_features:
        if data[feature].dtype in ['object', 'bool']:
            features = pd.get_dummies(data[feature])
            features = np.array(features)
        else:
            features = data[feature]
            features = np.array(features).reshape(-1,1)

        # Convert to np array
        labels = np.array(labels).reshape((-1, ))
        
        # Iterate through each fold
        for _ in range(n_iterations):

            model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)
                
            # If training using early stopping need a validation set
            if early_stopping:
                
                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.15, random_state=1234)

                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric = eval_metric,
                            eval_set = [(valid_features, valid_labels)],
                            early_stopping_rounds = 100, verbose = 0)
                
            else:
                model.fit(features, labels)
        feature_importance_values[feature] = model.best_score_['valid_0'][eval_metric]

    feature_importances = pd.DataFrame(feature_importance_values, index=[0]).T.reset_index()
    feature_importances.columns = ['feature', 'importance']

    # Sort features according to importance
    feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

    # Normalize the feature importances to add up to one
    imp = feature_importances['importance']
    feature_importances['normalized_importance'] = imp / imp.sum()
    feature_importances['relative_importance'] = (imp - imp.min()) / (imp.max() - imp.min())
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

    return feature_importances


def get_feature_importances(dev_df, val_df, features, target, metric='auc', method='univariate', n_bins=20, iv_ll=.05, iv_ul=1.0, csi_ul=.25, corr=0.9):
    """
    Args:
        dev_df: development dataframe
        val_df: validation dataframe (to compute csi and filter)
        features: list of predictor variables or features
        target: target variable name
        metric: eval metric to be used for feature importance computation
        method: method to be used for feature importance computation. allowed values - `univariate` and `multivariate`
        n_bins: no of bins for IV or CSI computation (default=20)
        iv_ll: IV lower cutoff
        iv_ul: IV upper cutoff
        csi_ul: CSI upper cuttoff
        corr: correlation threshold
    Returns:
        feature importance data frame with IV, CSI, Feature Importance with Normalized and Relative Importance. And list of final features considered.
    """
    try:
        train = dev_df.copy()
        test = val_df.copy()
        X_train=train[features]
        X_test=test[features]
        y_train=train[target]
        y_test=test[target]
        if method=='univariate':
            feature_importance = get_univariate_feature_importances(X_train, y_train, eval_metric=metric).set_index('feature')
        elif method=='multivariate':
            feature_importance = get_multivariate_feature_importances(X_train, y_train, eval_metric=metric).set_index('feature')
        iv_df, _ = iv(X_train, y_train, n_bins=n_bins)
        csi_df = csi(dev_df, val_df, features, target, n_bins=n_bins)
        iv_csi = pd.merge(iv_df.groupby('var_name')['iv'].sum(),\
                            csi_df.groupby(['var_name']).csi.sum(), on='var_name', how='outer').sort_values(by=['iv','csi'], ascending=[False, False])
        iv_csi_importance = pd.merge(iv_csi, feature_importance, left_index=True, right_index=True).sort_values(by='relative_importance', ascending=False).rename_axis('features').reset_index()
        iv_csi_importance.style.bar(subset=['relative_importance'], color='#5fba7d')
        features_considered = iv_csi_importance[(iv_csi_importance.iv>=iv_ll) & (iv_csi_importance.iv<=iv_ul) & (iv_csi_importance.csi<=csi_ul)]
        corr_cols = correlated_cols(X_train, y_train, corr)
        if corr_cols is None:
            features_considered = features_considered.sort_values(by='importance', ascending=False)
        else:
            features_considered = features_considered.loc[~features_considered.index.isin(corr_cols)].sort_values(by='importance', ascending=False)
        # features_considered.style.bar(subset=['relative_importance'], color='#5fba7d')
        return iv_csi_importance, features_considered.features.tolist()
    except Exception as e:
        logger.error(f"Error occured - {e}")
        raise