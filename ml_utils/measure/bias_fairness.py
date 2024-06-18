import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger("ML UTILS")

def gsi(data, col='GENDER', col_val='F', target='positive_probability', n_bins=10):
    """
    Returns a pandas dataframe with gsi column for 10 bins created.
    Args:
        data: pandas dataframe
        col: column on which Group Stability Index has to be calculated (ex - Gender column)
        col_val: the selected value will be compared with reset of values (ex - F vs Rest for Gender column)
        target: score column using jar file (default=positive_probability)
        n_bins: number of bins to be created (default=10)
    """
    df = data.copy()
    df['decile'] = pd.qcut(df[target], n_bins, labels=False)
    df.loc[df[col]!=col_val, col] = 'Rest'
    pivot_ = df.groupby(['decile', col])[target].count().unstack()
    pivot = pivot_.div(pivot_.sum(axis=0),axis=1)
    pivot['gsi'] = (pivot[col_val]-pivot['Rest'])*np.log(pivot[col_val]/pivot['Rest']) # this is wrong
    return pivot