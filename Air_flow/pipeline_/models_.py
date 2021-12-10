import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
import torch
import torch.nn as nn
import torch.utils.data as data_utils

class models_:
    def __init__(self, path_):
        self.path_ = path_
        self.mo_ = torch.load(self.path_)['model']

    def get_predict(self, ppred_df):
        result = self.mo_.predict(ppred_df[['time', 'weekday', 'weekend',
                                            'instlo_1', 'instlo_2', 'inst_code',
                                            'sysname_lo', 'sysname_eq']])
        return result 
    
    def get_proba(self, ppred_df):
        prob_ = self.mo_.predict_proba(ppred_df[['time', 'weekday', 'weekend',
                                            'instlo_1', 'instlo_2', 'inst_code',
                                            'sysname_lo', 'sysname_eq']])
        return prob_

