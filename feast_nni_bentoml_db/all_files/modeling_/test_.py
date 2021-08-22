#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np
import random
import os
import sys
import torch
import torch.nn.functional as F
import time
import feast

from sklearn.utils import shuffle
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# %%
import nni
import argparse
import logging
import logging.handlers


# %%
from utils_ import save_checkpoint


# %% [markdown]
# ### Preprocessing_from_offline_store

# %%
def get_train_from_offline_store(path):
    # Connect to your feature store provider
    fs = feast.FeatureStore(repo_path=f"{path}")


    ### entity_df 
    parquet_ = pd.read_parquet(f'{path}/data/ppr_data_.parquet', engine='pyarrow')
    orders = parquet_[['ticket_id','event_timestamp']]

    # Retrieve training data
    training_df = fs.get_historical_features(
        entity_df=orders,
        features=[
            "dr_lauren_stat:time",
            "dr_lauren_stat:weekday",
            "dr_lauren_stat:weekend",
            "dr_lauren_stat:instlo_1",
            "dr_lauren_stat:instlo_2",
            "dr_lauren_stat:inst_code",
            "dr_lauren_stat:sysname_lo",
            "dr_lauren_stat:sysname_eq",
            "dr_lauren_stat:ntt_label",
        ],
    ).to_df()


    ### training_part_before 7 month

    criterion = '2021-07-01'

    training_df_ = training_df[training_df['event_timestamp'] < criterion]
    
    return training_df_


# %%
### train_test_split
def split_tr_te(df_,test_size_, seed_):
    print(df_)
    x = df_[['time', 'weekday', 'weekend', 'instlo_1', 'instlo_2', 'inst_code', 'sysname_lo', 'sysname_eq']]
    y = df_['ntt_label']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_, random_state=seed_)
    
    return x_train, x_test, y_train, y_test


# %% [markdown]
# ## model

# %% [markdown]
# ### LGBM

# %%
def get_lgbm_score(model_df, test_size_, ne_, seed_):
    x_train, x_test, y_train, y_test = split_tr_te(model_df, test_size_, seed_)

    mo_ = lgb.LGBMClassifier(n_estimators=ne_)
    mo_.fit(x_train, y_train)

    y_pred = mo_.predict(x_test)
    
    y_prob = mo_.predict_proba(x_test)[:,1]

    acc = accuracy_score(y_test, y_pred)

    f1_score_ = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)    
    
    return mo_, acc, f1_score_, auc


# %% [markdown]
# ### XGBM

# %%
def get_xgb_score(model_df, test_size_, ne_, seed_):
    x_train, x_test, y_train, y_test = split_tr_te(model_df, test_size_, seed_)

    mo_ = XGBClassifier(n_estimators=ne_)
    mo_.fit(x_train, y_train)

    y_pred = mo_.predict(x_test)
    
    y_prob = mo_.predict_proba(x_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)

    f1_score_ = f1_score(y_test, y_pred)
    
    auc = roc_auc_score(y_test, y_prob)
    
    return mo_, acc, f1_score_, auc


# %% [markdown]
# ### main

# %%
parser = argparse.ArgumentParser(description='sklearn Training')

parser.add_argument('--model-dir', default='./models_', type=str)
parser.add_argument('--seed', default=42, type=int,
                    help='the number of seed')
parser.add_argument('--request-from-nni', default=False, action='store_true')
parser.add_argument('--model-n', default="lgb", type=str,
                    help='initial learning rate')

parser.add_argument('--ne', '--n-estimators', default=100, type=int,
                    help='initial learning rate')

parser.add_argument('--gpu', default=None, type=int,
                    help='initial learning rate')


# %%
def main():
    args = parser.parse_args()
    
    if args.request_from_nni:
        import nni
        tuner_params = nni.get_next_parameter()
#         logger.info(str(tuner_params))

            
        if "ne" in tuner_params:
            args.ne_ = tuner_params["ne"]

        if "model_n" in tuner_params:
            args.model_ = tuner_params["model_n"]
            
        if "seed" in tuner_params:
            args.seed_ = tuner_params["seed"]


#         logger.info(str(args))

        
        # demonstrate that intermediate result is actually sent
        nni.report_intermediate_result(0.)

        args.model_dir = '/workspace/ML_Ops/feast/models_'



    if args.gpu is not None:
        args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    path_ = '/workspace/ML_Ops/feast/fea_/'
    #### from feature offline store
    train_df = get_train_from_offline_store(path_)
    
    args.train_df_ = train_df

    # Simply call main_worker function
    main_worker(args.gpu, args)


# %%
def main_worker(gpu, args):
    best_acc = 0
    best_auc = 0

    device = args.gpu

    df_ = args.train_df_


    if args.model_ == "lgb":

        mo_, acc_, f1_score_, auc_ = get_lgbm_score(args.train_df_, 0.3, args.ne_, args.seed_)

    elif args.model_ == "xgb":

        mo_, acc_, f1_score_, auc_ = get_xgb_score(args.train_df_, 0.3, args.ne_, args.seed_)



    # remember best acc@1 and save checkpoint

    acc_best = acc_ > best_acc
    best_acc = max(acc_, best_acc)

    auc_best = auc_ > best_auc
    best_auc = max(auc_, best_auc)

    save_checkpoint({
        'acc': acc_,
        'auc': auc_,
        'seed' : args.seed_,
        'model': mo_,
    }, args.model_, acc_, auc_, args.model_dir)


    try:
        if args.request_from_nni:
            import nni
            nni.report_final_result(acc_)
    except NameError:

        pass

# %%
if __name__ == '__main__':
    main()

# %%

# %%
