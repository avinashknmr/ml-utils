"""
Helper functions to create dev and oot splits along with partition groups
for holdout and cross validation following a process of 60:40 split for model development purposes
and rest for validation.
"""
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import paramiko
import logging, os, sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger("ML UTILS")

from configparser import ConfigParser
cp = ConfigParser()
file = os.path.join(os.path.expanduser('~'),'.config','ml_utils','config.ini')
if not os.path.isfile(file):
    os.makedirs(os.path.join(os.path.expanduser('~'),'.config','ml_utils'), exist_ok=True)
    cp.add_section('sas')
    cp['sas']['host'] = ''
    cp['sas']['username'] = ''
    with open(file, 'w') as f:
        cp.write(f)

cp.read(file)
config = cp['sas']

def read_remote_csv(filepath, host=config['host'], username=config['username'], password=config.get('password','')):
    """Read remote csv file into a pandas DataFrame."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if (password=='') or (password is None):
        client.connect(host, username=username)
    else:
        client.connect(host, username=username, password=password)
    sftp = client.open_sftp()
    remote_file = sftp.open(filepath)
    df = pd.read_csv(remote_file)
    remote_file.close()
    sftp.close()
    client.close()
    return df

def read_remote_sas(filepath, host=config['host'], username=config['username'], password=config.get('password','')):
    """Read remote sas7bdat file into a pandas DataFrame."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if (password=='') or (password is None):
        client.connect(host, username=username)
    else:
        client.connect(host, username=username, password=password)
    sftp = client.open_sftp()
    remote_file = sftp.open(filepath)
    df = pd.read_sas(remote_file, format='sas7bdat')
    remote_file.close()
    sftp.close()
    client.close()
    return df

def mix_snapshots(S1, S2, target, random_state=1234, create_partition=True):
    """
    Retuns 2 `pandas.DataFrame` for model development purpose (created with 60% of S1 and 40% of S2)
    and validation purpose (created with 40% of S1 and 60% of S2) from S1, S2 by creating a 60/40 split.
    Additionally, can also create partitions, if enabled for holdout and cross validation.
    Args:
        S1 (df): DEV dataset creted in SAS
        S2 (df): VAL dataset created in SAS
        target (str): Target column name on which stratified split is performed (examples - RESP, F_BAD)
        create_partition (bool): To create or not create **Group** and **Partition** columns. (Defaults to True)
    """
    # read input files
    # df_S1 = pd.read_csv(S1)
    # df_S2 = pd.read_csv(S2)
    df_S1 = S1
    df_S2 = S2

    # split 60:40 for development and 40:60 for validation
    df_S1_60, df_S1_40 = train_test_split(df_S1, test_size=0.4, random_state=random_state, stratify=df_S1[target])
    df_S2_40, df_S2_60 = train_test_split(df_S2, test_size=0.6, random_state=random_state, stratify=df_S2[target])

    # concat datasets for development and validation
    df_DEV = pd.concat([df_S1_60, df_S2_40]).reset_index()
    df_VAL = pd.concat([df_S1_40, df_S2_60]).reset_index()

    # to make group and partition columns
    if create_partition:
        df_DEV = partition(df_DEV, target, holdout_pct=0.2, n_splits=5, out_col='Group', random_state=random_state)
        df_VAL = partition(df_VAL, target, holdout_pct=0.2, n_splits=5, out_col='Group', random_state=random_state)
        
        df_DEV = partition(df_DEV, target, holdout_pct=0.4, n_splits=2, out_col='Partition', random_state=random_state)
        df_VAL = partition(df_VAL, target, holdout_pct=0.4, n_splits=2, out_col='Partition', random_state=random_state)
    

    return df_DEV, df_VAL

def partition(df, target, holdout_pct=0.2, n_splits=5, out_col='Group', random_state=1234):
    """
    Returns a `pandas.DataFrame` with additional column that contains the groups/ partitions
    for model building purposes.
    Args:
        df (df): Input data as `pandas.DataFrame`
        target (str): Name of target column on which stratified split is performed.
        holdout_pct (float): Holdout percentage (Defaults to 0.2 = 20%)
        n_splits (int): No of splits to be made excluding holdout (Defaults to 5)
        out_col (str): Name of output column (Defaults to Group)
    """
    cv, holdout = train_test_split(df, test_size=holdout_pct, random_state=random_state, stratify = df[target])
    cv = cv.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y = cv[target]

    for i, idx in enumerate(skf.split(cv.loc[:,cv.columns!=target], y)):
        cv.loc[idx[1], out_col] = int(i+1)

    out = pd.concat([cv,holdout], axis=0).reset_index(drop=True)
    out[out_col].fillna(n_splits+1, inplace=True)
    return out