"""
Helper functions to generate commonly used reports for final comparison purposes.
Todo:
    Include psi, decilewise counts, rank order, top2 decile capture...
"""
import os, sys
import pandas as pd
import numpy as np
import openpyxl
from loguru import logger
# import mlflow

from .draw import csi_plot, iv_plot
from .measure.metrics import Metrics
from .feature_selection import _quick_psi

if hasattr(sys.modules['__main__'],'get_ipython'):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def leaderboard_report(input_path, resp_col, prediction_col='positive_probability', outfile_name='results.xlsx', leaderboard_file='algo_leaderboard.csv', log_mlflow=False):
    """
    Generates a comparison report with standard metrics between dev and oot datafiles
    """
    lb = pd.read_csv(leaderboard_file)
    models = lb.model_id.unique()
    featurelist_name = leaderboard_file.replace('.csv','').split('_')[-1]

    final_df = []
    psi_metrics = {}
    dw_cutoffs = {}
    logger.info('Calculating Metrics...')
    for m in tqdm(models):
        metrics = {}
        dw_counts = {}
        files = [f for f in os.listdir(input_path) if f.endswith('.csv') and m in f]
        dev_file = [f for f in files if 'S1' in f][0]
        dev = pd.read_csv(os.path.join(input_path, dev_file))
        dev_cuts = dev[prediction_col].quantile(np.arange(0,10+1)/10).reset_index(drop=True)
        dw_pop = {}
        dw_resp = {}
        for f in files:

            src = 'S1' if 'S1' in f else 'S2' if 'S2' in f else 'S3' if 'S3' in f else 'S4060'
            df = pd.read_csv(os.path.join(input_path, f))
            metrics[f] = Metrics(df[resp_col], df[prediction_col]).values
            try:
                dw = decilewise_counts(df, resp_col, prediction_col, cutpoints=dev_cuts)
            except:
                print(f)
            dw_dict = {'total': dw.total.sum(), 'resp': dw.resp.sum(), 'top2count': dw[:2].resp.sum(), 'top2capturerate': dw[:2].resp.sum()/dw.resp.sum(), 'rankbreak': dw[dw.resp.diff().fillna(0)>0].index.min(), 'rankbreak_flag': 1 if dw.resp.diff().fillna(0).max()>0 else 0, 'capturerate_before_ro_break': dw.resp[:int(np.nan_to_num(dw[dw.resp.diff().fillna(0)>0].index.min(),nan=10))].sum()/dw.resp.sum()}
            dw_pop_dict = {'countD'+str(i+1): r.total for i, r in dw.iterrows()}
            dw_resp_dict = {'respD'+str(i+1): r.resp for i, r in dw.iterrows()}
            dw_counts[f] = {**dw_dict, **dw_pop_dict, **dw_resp_dict}
            dw_pop[src] = (dw.total/dw.total.sum())
            dw_resp[src] = (dw.resp/dw.resp.sum())
        dw_cutoffs[m] = {'cutoffD'+str(i+1): v for i, v in enumerate(dev_cuts[::-1])}
        psi_metrics[m] = {'psi_S2': _quick_psi(dw_pop['S1'], dw_pop['S2']),'psi_S3': _quick_psi(dw_pop['S1'], dw_pop['S3']),'psi_S4060': _quick_psi(dw_pop['S1'], dw_pop['S4060'])}
        metrics_df = pd.DataFrame(metrics).T
        dw_counts_df = pd.DataFrame(dw_counts).T
        merge_df = pd.merge(metrics_df, dw_counts_df, how='inner', left_index=True, right_index=True)
        merge_df['algo'] = merge_df.index.map(lambda x: x.split('_')[1])
        merge_df['stage'] = merge_df.index.map(lambda x: x.split('_', maxsplit=2)[2].split('_')[0].replace('.csv','').upper())
        merge_df.reset_index(drop=True)
        final_df.append(merge_df)
    out_df = pd.concat(final_df, axis=0)
    out_df = out_df.pivot(index='algo', columns='stage')
    out_df.columns = [f'{x}_{y}' for x,y in out_df.columns]
    out_df.reset_index(drop=True)
    out_df['gini_variance'] = out_df.apply(lambda r: abs((r.gini_S1 - r.gini_S2)/(r.gini_S1+0.00001)), axis=1)
    psi_metrics_df = pd.DataFrame(psi_metrics).T
    psi_metrics_df['model_id'] = psi_metrics_df.index
    dw_cutoffs_df = pd.DataFrame(dw_cutoffs).T
    dw_cutoffs_df['model_id'] = dw_cutoffs_df.index
    final = pd.merge(lb, out_df, how='left', left_on='model_id', right_on='algo')
    final = pd.merge(final, psi_metrics_df, how='left', on='model_id')
    final = pd.merge(final, dw_cutoffs_df, how='left', on='model_id')
    final.to_excel(outfile_name, sheet_name='Model Eval', engine='openpyxl', index=False)
    # if log_mlflow:
    #     mlflow.set_tracking_uri('http://uklvadsb0358.uk.dev.net:5000')
    #     mlflow.set_experiment('SG CC')
    #     logger.info('Writing results to MLFlow')
    #     for i, r in tqdm(final[:2].iterrows()):
    #         with mlflow.start_run(run_name=r.model_id) as run:
    #             mlflow.log_params({'model_name': r.model, 'featurelist_name': featurelist_name})
    #             mlflow.log_metrics({'gini_S1': r.gini_S1, 'gini_S2': r.gini_S2, 'gini_variance': r.gini_variance, 'psi': r.psi_S2, 'rankbreak_flag_S1': r.rankbreak_flag_S1, 'rankbreak_flag_S2': r.rankbreak_flag_S2, 'capturerate_before_ro_break_S1': r.capturerate_before_ro_break_S1, 'capturerate_before_ro_break_S2': r.capturerate_before_ro_break_S2})
    #             mlflow.log_dict(r.to_dict(), 'metrics.json')
    logger.info(f'Model Evaluation Report Generated - {outfile_name}')

def csi_report(csi_df, csi_excel_name, feature_cols=None):
    """Generate CSI Report."""
    feature_cols = feature_cols if feature_cols is not None else csi_df.var_name.unique()
    writer = pd.ExcelWriter(csi_excel_name, engine='openpyxl')
    csi_summary = csi_df.groupby('var_name')[['iv_dev','iv_oot','csi']].sum().sort_values(by=['iv_dev','iv_oot','csi'], ascending=[False, False, True]).reset_index()
    csi_summary.to_excel(writer, sheet_name='CSI_Summary', engine='openpyxl', index=False)
    i=0
    logger.info('Generating CSI Data...')
    for col in tqdm(feature_cols):
        sub_df = csi_df[csi_df['var_name']==col]
        sub_df.to_excel(writer, sheet_name='CSI', engine='openpyxl', index=False, startrow=i, startcol=0)
        i += max(sub_df.shape[0],20)+4
        writer.save()
    writer.close()
    wb = openpyxl.load_workbook(csi_excel_name)
    ws = wb['CSI']
    i=1
    logger.info('Generating CSI Plots...')
    for col in tqdm(feature_cols):
        try:
            f = csi_plot(csi_df,col)
            f.savefig(f'csi_plots\\csi_{col}.jpg')
            img = openpyxl.drawing.image.Image(f'csi_plots\\csi_{col}.jpg')
            img.anchor='AI'+str(i)
            ws.add_image(img)
            wb.save(csi_excel_name)
        except:
            logger.error(f'Could not generate plot for {col}')
        i += max(sub_df.shape[0],20)+4
    wb.close()
    # os.remove('figure.jpg')
    logger.info(f'CSI Report Generated - {csi_excel_name}')

def iv_report(iv_df, iv_excel_name, feature_cols=None, suffix='_dev'):
    """Generate IV Report."""
    feature_cols = feature_cols if feature_cols is not None else iv_df.var_name.unique()
    writer = pd.ExcelWriter(iv_excel_name, engine='openpyxl')
    iv_summary = iv_df.groupby('var_name')['iv'+suffix].sum().sort_values(ascending=False).reset_index()
    iv_summary.to_excel(writer, sheet_name='IV_Summary', engine='openpyxl', index=False)
    i=0
    logger.info('Generating IV Data...')
    for col in tqdm(feature_cols):
        sub_df = iv_df[iv_df['var_name']==col]
        sub_df.to_excel(writer, sheet_name='IV', engine='openpyxl', index=False, startrow=i, startcol=0)
        i += max(sub_df.shape[0],20)+4
        writer.save()
    writer.close()
    wb = openpyxl.load_workbook(iv_excel_name)
    ws = wb['IV']
    i=1
    logger.info('Generating IV Plots...')
    for col in tqdm(feature_cols):
        try:
            f = iv_plot(iv_df, col)
            f.savefig(f'iv_plots\\iv_{col}.jpg')
            img = openpyxl.drawing.image.Image(f'iv_plots\\iv_{col}.jpg')
            img.anchor='R'+str(i)
            ws.add_image(img)
            wb.save(iv_excel_name)
        except:
            logger.error(f'Could not generate plot for {col}')
        i += max(sub_df.shape[0],20)+4
    wb.close()
    # os.remove('figure.jpg')
    logger.info(f'IV Report Generated - {iv_excel_name}')