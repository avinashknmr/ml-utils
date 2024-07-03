"""
Helper functions to draw various plots like gains plot (lift chart), correlation plot,
response rate & ln odds vs variable bins, response rate in dev and oot for each variable bins.
All functions return a matplotlib plot that can be saved.
```python
f = roc_plot(y, y_pred)
f.savefig('<name>.png|jpg')
```
"""
from numpy.core.shape_base import block
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import logging, os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
_logger = logging.getLogger("ML UTILS")

sns.set_theme(style="whitegrid")

def density_plot(data, feature, target=None):
    """
    Args:
        data: dataframe
        feature: feature for which plot has to be created
        target: (optional) create sub charts based on target in same plot
    """
    if data[feature].dtype in ['object','bool']:
        f = sns.displot(data=data, x=feature, hue=target, stat="density", common_norm=False, element='bars', aspect=2)
    else:
        f = sns.displot(data=data, x=feature, hue=target, bins=20, stat="density", common_norm=False, element='step', aspect=2)
        # sns.boxplot(data=data, y=feature, x=target)
    return f

def export_density_plots(data, features, out_folder, target=None):
    """
    Args:
        data: dataframe
        features: list of features for which plot has to be created
        out_folder: folder to which plots has to be saved
        target: (optional) create sub charts based on target in same plot
    """
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for f in features:
        plot = density_plot(data, f, target=target)
        try:
            fig = plot.get_figure()
            fig.savefig(os.path.join(out_folder, f+'.png'))
        except:
            plot.fig.savefig(os.path.join(out_folder, f+'.png'))

def roc_plot(actual, predicted, figsize=(7,5)):
    """
    Returns a lift chart or gains plot against True Positive Rate vs False Positive Rate.
    ![Gains Chart](./gains_plot.png "Lift Curve")
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    fpr, tpr, threshold = roc_curve(actual, predicted)
    gmean = np.sqrt(tpr*(1-fpr))
    index = np.argmax(gmean)
    gmeanOpt = round(gmean[index], 4)
    thresholdOpt = round(threshold[index], 4)
    fprOpt = round(fpr[index], 4)
    tprOpt = round(tpr[index], 4)
    sns.lineplot(x=fpr, y=tpr)
    sns.lineplot(x=np.linspace(0,1,10), y=np.linspace(0,1,10), linestyle='dashed')
    ax.text(fprOpt, tprOpt, f'Optimal Threshold {thresholdOpt}')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Gains Plot / Lift Chart')
    f.tight_layout()
    plt.close(f)
    return f

def precision_recall_plot(actual, predicted, figsize=(7,5)):
    """
    Returns a Recall vs Precision plot.
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    p, r, threshold = precision_recall_curve(actual, predicted)
    fscore = 2*p*r/(p+r)
    index = np.argmax(fscore)
    fscoreOpt = round(fscore[index], 4)
    thresholdOpt = round(threshold[index], 4)
    recallOpt = round(r[index], 4)
    precisionOpt = round(p[index], 4)
    sns.lineplot(x=r, y=p)
    # sns.lineplot(x=np.linspace(0,1,10), y=np.linspace(0,1,10))
    ax.text(recallOpt, precisionOpt, f'Optimal Threshold {thresholdOpt}')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision Recall Plot')
    f.tight_layout()
    plt.close(f)
    return f

def correlation_plot(data, features=None, annot=True, rotate_xlabels=True, figsize=(7,5)):
    """
    Returns a correlation plot as `seaborn.heatmap` headmap with annotations.
    ![Correlation Plot](./corr_plot.png "HeatMap")
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    mdf = data if features is None else data[features]
    corr = mdf.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] =True
    ax = sns.heatmap(corr, vmin=-1, vmax=1, square=True, annot=annot, fmt=".1f", mask=mask, cmap=sns.diverging_palette(20,220,n=200))
    if rotate_xlabels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    f.tight_layout()
    plt.close(f)
    return f

def _trend(Y):
    y = np.array(Y)
    x = np.arange(0,len(y))
    z = np.polyfit(x,y,1)
    trend_line = z[1]+z[0]*(x+1)
    return trend_line

def iv_plot(data, feature=None, suffix='_dev', figsize=(7,5)):
    """
    Returns a plot with reponse rate and ln odds on y axis for various variable bins
    created from `ml_utils.measure.woe_bins`. Also called as IV chart internally.
    ![IV Plot](./iv_plot.png "IV Plot")
    """
    p_suffix = suffix.replace('_','').upper()
    sub_df = data if feature is None else data.loc[data.var_name==feature, ['var_cuts_string'+suffix,'ln_odds'+suffix,'resp_rate'+suffix,'iv'+suffix]]
    sub_df['resp_rate_trend'+suffix]= _trend(sub_df['resp_rate'+suffix])
    iv_val = round(sub_df['iv'+suffix].sum(),4)
    
    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax.twinx()
    sns.lineplot(x='var_cuts_string'+suffix, y='resp_rate'+suffix, data=sub_df, color='red', ax=ax)
    sns.lineplot(x='var_cuts_string'+suffix, y='resp_rate_trend'+suffix, data=sub_df, color='red', linestyle='--', ax=ax)
    sns.lineplot(x='var_cuts_string'+suffix, y='ln_odds'+suffix, data=sub_df, color='darkgreen', ax=ax2)
    
    ax.set_xticklabels(list(sub_df['var_cuts_string'+suffix]),rotation=45, ha='right')
    ax.set(xlabel='Variable Bins', ylabel=f'Resp Rate ({p_suffix})', title=f'IV of {feature} ({iv_val})')
    ax2.set(ylabel=f'Log Odds ({p_suffix})')
    ax.legend(handles=[l for a in [ax, ax2] for l in a.lines], labels=[f'Resp Rate ({p_suffix})',f'Resp Rate Trend ({p_suffix})', f'Log Odds ({p_suffix})'], loc=0)
    f.tight_layout()
    plt.close(f)
    return f

def csi_plot(data, feature, figsize=(7,5)):
    """
    Returns a plot response rate and their trend line for both dev and oot
    for various variable bins created in the process of CSI calculation
    using `ml_utils.measure.iv` and `ml_utils.measure.csi`.
    Also called as CSI chart internally.
    """
    sub_df = data.loc[data.var_name==feature, ['var_cuts_string_dev','resp_rate_dev', 'resp_rate_oot','csi']]
    sub_df['resp_rate_trend_dev']=_trend(sub_df['resp_rate_dev'])
    sub_df['resp_rate_trend_oot']=_trend(sub_df['resp_rate_oot'])
    csi_val = round(sub_df['csi'].sum(),4)
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_dev', data=sub_df, color='red')
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_trend_dev', data=sub_df, color='red', linestyle='--')
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_oot', data=sub_df, color='darkgreen')
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_trend_oot', data=sub_df, color='darkgreen', linestyle='--')
    ax.set_xticklabels(list(sub_df['var_cuts_string_dev']),rotation=45, ha='right')
    ax.set(xlabel='Variable Bins', ylabel=f'Resp Rate', title=f'CSI of {feature} ({csi_val})')
    ax.legend(handles=[l for a in [ax] for l in a.lines], labels=['Resp Rate (Dev)','Resp Rate (Dev) Trend','Resp Rate (OOT)','Resp Rate (OOT) Trend'], loc=0)
    f.tight_layout()
    plt.close(f)
    return f