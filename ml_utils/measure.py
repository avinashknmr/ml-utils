import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve

from .woe_binning import woe_binning, woe_binning_2, woe_binning_3

class Metrics:
    def __init__(self, df, actual, prediction):
        self.df = df
        self.target = actual
        self.actual = df[actual]
        self.prediction = df[prediction]
        self.gains = self.calculate_gains()
        self.ks = self.ks()
        self.gini = self.gini()
        self.tn, self.fp, self.fn, self.tp, self.precision, self.recall, self.f1_score = self.precision_recall_f1_score()

    def calculate_gains(self):
        """Returns a pandas dataframe with gains along with KS and Gini calculated"""
        self.df['scaled_score'] = (self.df['positive_probability']*1000000).round(0)
        gains = self.df.groupby('scaled_score')[self.target].agg(['count','sum'])
        gains.columns = ['total','responders']
        gains.reset_index(inplace=True)
        gains.sort_values(by='scaled_score', ascending=False)
        gains['non_responders'] = gains['total'] - gains['responders']
        gains['cum_resp'] = gains['responders'].cumsum()
        gains['cum_non_resp'] = gains['non_responders'].cumsum()
        gains['total_resp'] = gains['responders'].sum()
        gains['total_non_resp'] = gains['non_responders'].sum()
        gains['perc_resp'] = (gains['responders']/gains['total_resp'])*100
        gains['perc_non_resp'] = (gains['non_responders']/gains['total_non_resp'])*100
        gains['perc_cum_resp'] = gains['perc_resp'].cumsum()
        gains['perc_cum_non_resp'] = gains['perc_non_resp'].cumsum()
        gains['k_s'] = gains['perc_cum_resp'] - gains['perc_cum_non_resp']
        return gains

    def get_threshold(self):
        """Returns a pandas dataframe with y_pred based on threshold from roc_curve."""
        fpr, tpr, threshold = roc_curve(self.actual, self.prediction)
        threshold_cutoff_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
        threshold_cutoff_df['distance'] = ((threshold_cutoff_df['fpr']-0)**2+(threshold_cutoff_df['tpr']-1)**2)**0.5
        threshold_cutoff_df['distance_diff'] = abs(threshold_cutoff_df['distance'].diff(periods=1))
        for index, rows in threshold_cutoff_df.iterrows():
            if index != 0 and index != threshold_cutoff_df.shape[0]-1:
                curr_val = threshold_cutoff_df.loc[index, 'distance_diff']
                prev_val = threshold_cutoff_df.loc[index-1, 'distance_diff']
                next_val = threshold_cutoff_df.loc[index+1, 'distance_diff']
                if curr_val>prev_val and curr_val>next_val:
                    threshold_cutoff = threshold_cutoff_df.loc[index, 'threshold']
                    break
        return threshold_cutoff

    def gini(self):
        fpr, tpr, threshold = roc_curve(self.actual, self.prediction)
        auroc = auc(fpr, tpr)
        gini  = 2*auroc -1
        return gini

    def ks(self):
        gains  = self.gains()
        return gains['k_s'].max()

    def precision_recall_f1_score(self):
        threshold_cutoff = self.get_threshold()
        self.y_pred = np.where(self.prediction>=threshold_cutoff,1,0)
        self.df['y_pred'] = self.y_pred
        tn, fp, fn, tp = confusion_matrix(self.actual, self.y_pred).ravel()
        precision = precision_score(self.actual, self.y_pred)
        recall = recall_score(self.actual, self.y_pred)
        f1 = f1_score(self.actual, self.y_pred)
        return tn, fp, fn, tp, precision, recall, f1

    def to_dict(self):
        return {'ks': self.ks, 'gini': self.gini, 'tn': self.tn, 'tp': self.tp, 'fn': self.fn, 'fp': self.fp, 'precision': self.precision, 'recall': self.recall, 'f1_score': self.f1_score}

def standard_metrics(df, target_col, prediction_col):
    """Returns a dict with all metrics - Gini, KS, Precision, Recall, F1 Score, True Negative, True Positive, False Positive, False Negative."""
    metrics = Metrics(df, target_col, prediction_col)
    return metrics.to_dict()

def quick_psi(dev, val):
    """Calculate PSI from 2 arrays - dev and val"""
    return sum([(a-b)*np.log(a/b) for (a,b) in zip(dev,val)])

def psi(dev, val, target='positive_probability', n_bins=10):
    """
    Returns a pandas dataframe with psi column (Population Stability Index) after creating 10 deciles.
    Code includes creating score calculation using round(500-30 x log(100 x (p/(1-p))), 0) where p is probability.
    We need to pass both dev and val at same time to apply same bins created on dev dataframe.
    """
    dev['score'] = dev[target].apply(lambda x: round(500-30*np.log2(100*(x/(1-x))), 0))
    val['score'] = val[target].apply(lambda x: round(500-30*np.log2(100*(x/(1-x))), 0))
    
    dev['bins'], bins = pd.qcut(dev.score, n_bins, retbins=True, precision=0)
    bins = [int(i) if abs(i)!=np.inf else i for i in bins]
    val[bins] = pd.cut(val.score, bins)

    dev_bins = dev.bins.value_counts(sort=False, normalize=True)
    val_bins = val.bins.value_counts(sort=False, normalize=True)

    psi_ = pd.concat([dev_bins, val_bins], axis=1)
    psi_.columns = ['dev', 'val']
    psi_['psi'] = (psi_.dev - psi_.val)*np.log(psi_.dev/psi_.val)
    return psi_

def gsi(data, col='GENDER', col_val='F', target='positive_probability', n_bins=10):
    """
    Returns a pandas dataframe with gsi columns (Group Stability Index) after creating n bins.
    Args:
        data: pandas dataframe
        col: Columns on which GSI has to be calculated (ex: Gender column)
        col_val: selected value will be compared with rest of the values (ex: F vs Rest)
        target: score column
        n_bins: number of bins to be created (Default=10)
    """
    df = data.copy()
    df['decile'] = pd.qcut(df[target], n_bins, labels=False)
    df.loc[df[col]!=col_val, col] = 'Rest'
    pivot_ = df.groupby(['decile', col])[target].count().unstack()
    pivot = pivot_.div(pivot_.sum(axis=0), axis=1)
    pivot['gsi'] = (pivot[col_val]-pivot['Rest'])*np.log(pivot[col_val]/pivot['Rest'])
    return pivot

def chi_square(df, suffix='_dev'):
    """Returns a pandas dataframe with calculated fields - resp_rate, perc_dist, perc_non_resp, perc_resp, raw_odds, ln_odds, iv_bins, exp_resp, exp_non_resp, chi_square."""
    df['resp_rate'+suffix] = (df['responders'+suffix]*100)/df['total'+suffix]
    df['perc_dist'+suffix] = (df['total'+suffix]*100)/df.groupby('var_name')['total'+suffix].transform('sum')
    df['perc_non_resp'+suffix] = (df['non_responders'+suffix]*100)/df.groupby('var_name')['non_responders'+suffix].transform('sum')
    df['perc_resp'+suffix] = (df['responders'+suffix]*100)/df.groupby('var_name')['responders'+suffix].transform('sum')
    df['raw_odds'+suffix] = df.apply(lambda r: 0 if r['perc_resp'+suffix]==0 else r['perc_non_resp'+suffix]/r['perc_resp'+suffix], axis=1)
    df['ln_odds'+suffix] = df['raw_odds'+suffix].apply(lambda x: 0 if abs(np.log(x))==np.inf else np.log(x))
    df['iv_bins'+suffix] = (df['perc_non_resp'+suffix]-df['perc_resp'+suffix])*df['ln_odds'+suffix]/100
    df['exp_resp'+suffix] = df['total'+suffix]*df.groupby('var_name')['responders'+suffix].transform('sum')/df.groupby('var_name')['total'+suffix].transform('sum')
    df['exp_non_resp'+suffix] = df['total'+suffix]*df.groupby('var_name')['non_responders'+suffix].transform('sum')/df.groupby('var_name')['total'+suffix].transform('sum')
    df['chi_square'+suffix] = (((df['responders'+suffix]-df['exp_resp'+suffix])**2)/df['exp_resp'+suffix]) + (((df['non_responders'+suffix]-df['exp_non_resp'+suffix])**2)/df['exp_non_resp'+suffix])
    return df

def woe_bins(df, var_name, resp_name, suffix='_dev', var_cuts=None):
    """
    Returns a pandas dataframe, var_cuts after creating bins basd on `ml_utils.woe_binning`.
    Returns:
        df: pandas dataframe has var_cuts_string, total, responders, non_responders, var_name (with _dev or _val suffix)
        var_cuts: list of Interval items to be used on val file.
    """
    df1 = df[[resp_name, var_name]]
    if (np.issubdtype(df1[var_name].dtype, np.number)):
        n = df1[var_name].nunique()
        if var_cuts is None:
            suffix = '_dev'
            var_cuts = woe_binning_3(df1, resp_name, var_name, 0.05, 0.00001, 0, 50, 'bad', 'good')
        df['var_binned'] = pd.cut(df[var_name], var_cuts, right=True, labels=None, retbins=False, precision=10, include_lowest=False)
        var_min = float(df1[var_name].min())
        var_max = float(df1[var_name].max())
        summ_df = df1.groupby('var_binnied')[resp_name].agg(['count','sum']).reset_index()
        summ_df['delta'] = summ_df['count'] - summ_df['sum']
        summ_df['var_name'] = var_name
        summ_df.columns = ['var_cuts', 'total'+suffix, 'responders'+suffix, 'non_responders'+suffix, 'var_name']
        summ_df['var_cuts_string'+suffix] = summ_df.var_cuts.apply(lambda x: str(x.left if x.left!=-np.inf else var_min)+' To '+str(x.right if x.right!=np.inf else var_max))
    else:
        df1[var_name].fillna('Blank', inplace=True)
        summ_df = df1.groupby(var_name)[resp_name].agg(['count','sum']).reset_index()
        summ_df['delta'] = summ_df['count'] - summ_df['sum']
        summ_df['var_name'] = var_name
        summ_df.columns = ['var_cuts_string'+suffix, 'total'+suffix, 'responders'+suffix, 'non_responders'+suffix, 'var_name']
        summ_df['var_cuts'] = summ_df['var_cuts_string'+suffix]
    return summ_df[summ_df['total'+suffix]!=0], var_cuts

def csi(dev_df, val_df, var_list, resp_name):
    """Returns a pandas dataframe with csi, csi_var, perc_csi columns (Charecteristic Stability Index) calculated based on both dev and val dataframes."""
    dev_dfs = []
    var_cuts = {}
    for var_name in var_list:
        summ_df, cut = woe_bins(dev_df, var_name, resp_name, '_dev')
        dev_dfs.append(summ_df)
        var_cuts[var_name] = cut
    
    dev = pd.concat(dev_dfs, axis=0)
    dev = chi_square(dev, '_dev')

    val_dfs = []
    val_cuts = {}
    for var_name in var_list:
        val_summ_df, val_cut = woe_bins(val_df, var_name, resp_name, '_val', var_cuts[var_name])
        val_dfs.append(val_summ_df)
        val_cuts[var_name] = val_cut

    val = pd.concat(val_dfs, axis=0)
    val = chi_square(val, '_val')

    final = pd.merge(dev, val, how='left', on=['var_name', 'var_cuts'], suffixes=['_dev','_val'])
    
    final['csi'] = ((final['perc_dist_dev']-final['perc_dist_val'])/100)*np.log(final['perc_dist_dev']/final['perc_dist_val'])
    final['csi_var'] = final.groupby('var_name')['csi'].transform('sum')
    final['perc_csi'] = (100*final.groupby('var_name')['csi'].transform('cumsum'))/final.groupby('var_name')['csi'].transform('sum')
    return final

def get_decilewise_counts(df, target, bins=10, cutpoints=None):
    """Returns a summarized pandas dataframe with total and responders for each decile based on positive_probability."""
    if cutpoints is None:
        cutpoints = df['positive_probability'].quantile(np.arange(0, bins+1)/bins).reset_index(drop=True)
        cutpoints = [0] + list(cutpoints) + [1]
    df['bins'] = pd.cut(df['positive_probability'], cutpoints)
    out_df = df.groupby('bins')[target].agg(['count','sum']).sort_values(by=['bins'], ascending=False).reset_index()
    out_df.columns = ['band', 'total', 'responders']
    return out_df