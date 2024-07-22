from loguru import logger

def get_exploratory_analysis(df, features):
    """
    Args:
        df: pandas dataframe for which summary has to be created
        features: predictor variables or features for which summary has to be created
    Returns:
        summary information of features like - mean, standard deviation, missing, unique, top, frequency, min, 25%, 50%, 75%, max of values - as a pandas dataframe.
    """
    try:
        nmiss = df[features].isna().sum().rename('missing')
        desc = df[features].describe(include='all').T.merge(nmiss, how='left', left_index=True, right_index=True).rename_axis('features').reset_index()
        return desc.fillna('')
    except Exception as e:
        logger.error(f"Error occured - {e}")
        raise

def correlated_cols(df, features, target, threshold=0.9):
    rank_cols = features.copy()
    rank_cols.append(target)
    rank_df = df[rank_cols].corr().reset_index()[['index', target]]
    rank_df['corr_val'] = rank_df[target]**2
    rank_df.sort_values(['corr_val', 'index'], ascending=[False, True], inplace=True)
    rank_df = rank_df.reset_index(drop=True)
    rank_df['rank'] = rank_df.index + 1
    rank_df.rename(columns={'index': 'var1'}, inplace=True)
    rank_df.drop([target], axis=1, inplace=True)
    rank_df
    corr_df = df[features].corr().unstack().sort_values().reset_index() # should add abs()
    corr_df.columns = ['var1','var2','corr_val']
    corr_df = corr_df[(corr_df.var1 > corr_df.var2) & (corr_df.corr_val > threshold)]

    del_var_df = corr_df.merge(rank_df, on='var1', how='left')
    del_var_df = del_var_df.merge(rank_df, left_on='var2', right_on='var1', how='left')
    del_var_df.rename(columns={'var1_x': 'var1', 'rank_x': 'var1_rank', 'rank_y': 'var2_rank'}, inplace=True)
    del_var_df['del_var'] = del_var_df.apply(lambda r: r.var1 if r.var1_rank>r.var2_rank else r.var2, axis=1)
    drop_cols = del_var_df[['del_var']].drop_duplicates()['del_var'].tolist()
    return drop_cols
