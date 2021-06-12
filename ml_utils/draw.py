import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def gain_plot(y_actual, y_pred):
    """Returns a lift chart or gains plot against True Positive Rate vs False Positive Rate"""
    f, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_actual, y_pred)
    sns.lineplot(fpr, tpr)
    sns.lineplot(np.linspace(0,1,10), np.linspace(0,1,10))
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Gains Plot / Lift Chart')
    return f

def _trend(Y):
    y = np.array(Y)
    x = np.arange(0, len(y))
    z = np.polyfit(x,y,1)
    trend_line = z[1] + z[0]*(x+1)
    return trend_line

def iv_plot(df, var_name=None, suffix='_dev'):
    """Returns an IV plot for a specified variable"""
    p_suffix = suffix.replace('_','').upper()
    sub_df = df if var_name is None else df.loc[df.var_name==var_name, ['var_cuts_string'+suffix, 'ln_odds'+suffix, 'resp_rate'+suffix, 'iv'+suffix]]
    sub_df['resp_rate_trend'+suffix] = _trend(sub_df['resp_rate'+suffix])
    iv_val = round(sub_df['iv'+suffix].sum(), 4)

    f, ax = plt.subplots()
    ax2 = ax.twinx()
    sns.lineplot(x='var_cuts_string'+suffix, y='resp_rate'+suffix, data=sub_df, color='red', ax=ax)
    sns.lineplot(x='var_cuts_string'+suffix, y='resp_rate_trend'+suffix, data=sub_df, color='red', linestyle='--', ax=ax)
    sns.lineplot(x='var_cuts_string'+suffix, y='ln_odds'+suffix, data=sub_df, color='darkgreen', ax=ax2)

    ax.set_xticklabels(list(sub_df['var_cuts_string'+suffix]), rotation=45, ha='right')
    ax.set(xlabel='Variable Bins', ylabel=f'Resp Rate ({p_suffix})', title=f'IV of {var_name} ({iv_val})')
    ax2.set(ylabel=f'Log Odds ({p_suffix})')
    ax.legend(handles=[l for a in [ax, ax2] for l in a.lines], labels=[f'Resp Rate ({p_suffix})', f'Resp Rate Trend ({p_suffix})', f'Log Odds ({p_suffix})'], loc=0)
    return f

def csi_plot(df, var_name):
    """Returns a CSI plot for a specified variable"""
    sub_df = df.loc[df.var_name==var_name, ['var_cuts_string_dev','resp_rate_dev','resp_rate_val']]
    sub_df['resp_rate_trend_dev'] = _trend(sub_df['resp_rate_dev'])
    sub_df['resp_rate_trend_val'] = _trend(sub_df['resp_rate_val'])
    f, ax = plt.subplots()
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_dev', data=sub_df, color='red')
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_trend_dev', data=sub_df, color='red', linestyle='--')
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_val', data=sub_df, color='darkgreen')
    sns.lineplot(x='var_cuts_string_dev', y='resp_rate_trend_val', data=sub_df, color='darkgreen', linestyle='--')
    ax.set_xticklabels(list(sub_df['var_cuts_string_dev']), rotation=45, ha='right')
    ax.set(xlabel='Variable Bins', ylabel='Resp Rate')
    ax.legend(handles=[l for a in [ax] for l in a.lines], labels=['Resp Rate (Dev)', 'Resp Rate Trend (Dev)', 'Resp Rate (Val)', 'Resp Rate Trend (Val)'], loc=0)