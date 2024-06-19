import pandas as pd
import numpy as np

from ..feature_selection import iv

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger("ML UTILS")

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