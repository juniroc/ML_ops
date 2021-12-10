# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.dates as md
import pickle
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns


import torch
import torch.nn as nn
import torch.utils.data as data_utils

from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


### new_class_encoding
from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)



# 
class preprocess:
    def __init__(self, inf_df_, trained_df_):
        self.inf_df = inf_df_
        self.trained_df = trained_df_

    def get_train_le(self, train_array_):
        le = LabelEncoderExt()
        le.fit(train_array_)

        return le.transform(train_array_)

    def get_inf_le(self, inf_array_, train_array_):
        le = LabelEncoderExt()
        le.fit(train_array_)
        
        return le.transform(inf_array_)

    def ppr_inf(self, inf_df_):
        ### change_columns_name

        

        inf_df_ = inf_df_[['ticketId','faultTime',
                           'rootCauseInstlocationa','rootCauseInstlocationz',
                           'rootCauseSysnamea', 'rootCauseSysnamez'
                          ]]


        inf_df_ = inf_df_.set_axis(['ticket_id', 'fault_time',
            'root_cause_instlocationa', 'root_cause_instlocationz',
            'root_cause_sysnamea', 'root_cause_sysnamez'
            ], axis='columns')
        

        inf_df_['fault_time'] = inf_df_['fault_time'].apply(lambda x : datetime.fromtimestamp(x//1000).isoformat())

        ### Time_table_ppr
        inf_df_['fault_time'] = pd.to_datetime(inf_df_['fault_time'])

        inf_df_['hour'] = inf_df_['fault_time'].dt.hour
        inf_df_['minute'] = inf_df_['fault_time'].dt.minute
        inf_df_['weekday'] = inf_df_['fault_time'].dt.weekday
        
        inf_df_['weekend'] = [0]*len(inf_df_)
        inf_df_['weekday_']= [0]*len(inf_df_)
        inf_df_['weekend'][inf_df_['weekday']>=5]=1
        inf_df_['weekday_'][inf_df_['weekday']<5]=1
        
        
        ### instlocation_a/z 
        inf_df_['root_cause_instlocationa'][inf_df_['root_cause_instlocationa']=='0'] = float('nan')
        inf_df_['root_cause_instlocationz'][inf_df_['root_cause_instlocationz']=='0'] = float('nan')
        
        ### sysname_a/z
        inf_df_['root_cause_sysnamea'][inf_df_['root_cause_sysnamea']=='0'] = float('nan')
        inf_df_['root_cause_sysnamez'][inf_df_['root_cause_sysnamez']=='0'] = float('nan')
        
        ### instlo_a/z divide
        inf_df_['instlo_a_1'] = inf_df_['root_cause_instlocationa'].apply(lambda x: x[:2] if type(x) == str else x)
        inf_df_['instlo_a_2'] = inf_df_['root_cause_instlocationa'].apply(lambda x: x[2:4] if type(x) == str else x)
        inf_df_['inst_code_a'] = inf_df_['root_cause_instlocationa'].apply(lambda x: x[4:] if type(x) == str else x)
        inf_df_ = inf_df_.drop(['root_cause_instlocationa'],axis=1)

        inf_df_['instlo_z_1'] = inf_df_['root_cause_instlocationz'].apply(lambda x: x[:2] if type(x) == str else x)
        inf_df_['instlo_z_2'] = inf_df_['root_cause_instlocationz'].apply(lambda x: x[2:4] if type(x) == str else x)
        inf_df_['inst_code_z'] = inf_df_['root_cause_instlocationz'].apply(lambda x: x[4:] if type(x) == str else x)
        inf_df_ = inf_df_.drop(['root_cause_instlocationz'],axis=1)
        
        ### sysname_a/z divide
        if type(inf_df_['root_cause_sysnamea'][0]) == str and len(inf_df_['root_cause_sysnamea'][0].split('_')) > 1:
            inf_df_['sysname_a_lo'] = inf_df_['root_cause_sysnamea'].apply(lambda x : x.split('_')[0] if type(x)==str else x)
            inf_df_['sysname_a_eq'] = inf_df_['root_cause_sysnamea'].apply(lambda x : x.split('_')[1] if type(x)==str else x)

        else:
            inf_df_['sysname_a_lo'] = inf_df_['root_cause_sysnamea'].apply(lambda x : x.split('-')[0] if type(x)==str else x)
            inf_df_['sysname_a_eq'] = 'nan'
    

        # print(inf_df_['root_cause_sysnamez'][0])
        if type(inf_df_['root_cause_sysnamez'][0]) == str and len(inf_df_['root_cause_sysnamez'][0].split('_')) > 1:
            inf_df_['sysname_z_lo'] = inf_df_['root_cause_sysnamez'].apply(lambda x : x.split('_')[0] if type(x)==str else x)
            inf_df_['sysname_z_eq'] = inf_df_['root_cause_sysnamez'].apply(lambda x : x.split('_')[1] if type(x)==str else x)
        
        else:
            inf_df_['sysname_z_lo'] = inf_df_['root_cause_sysnamez'].apply(lambda x : x.split('-')[0] if type(x)==str else x)
            inf_df_['sysname_z_eq'] = 'nan'


        
        
        ### df
        inf_df_ = inf_df_[['ticket_id','fault_time', 'hour', 'minute', 'weekday', 'weekday_', 'weekend',
            'instlo_a_1','instlo_a_2', 'inst_code_a',
            'instlo_z_1','instlo_z_2', 'inst_code_z',
            'sysname_a_lo','sysname_a_eq',
            'sysname_z_lo','sysname_z_eq'
            ]]
        
        return inf_df_
    
    def ppr_train(self, inf_df_):
        ### change_columns_name
        inf_df_ = inf_df_[['ticketId','faultTime',
                           'rootCauseInstlocationa','rootCauseInstlocationz',
                           'rootCauseSysnamea', 'rootCauseSysnamez',
                           'ntt_label'
                          ]]
        
        inf_df_.columns = [['ticket_id','fault_time',
            'root_cause_instlocationa','root_cause_instlocationz',
            'root_cause_sysnamea','root_cause_sysnamez',
            'ntt_label'
            ]]

    
        ### Time_table_ppr
        inf_df_['fault_time'] = pd.to_datetime(inf_df_['fault_time'])

        inf_df_['hour'] = inf_df_['fault_time'].dt.hour
        inf_df_['minute'] = inf_df_['fault_time'].dt.minute
        inf_df_['weekday'] = inf_df_['fault_time'].dt.weekday
        
        inf_df_['weekend'] = [0]*len(inf_df_)
        inf_df_['weekday_']= [0]*len(inf_df_)
        inf_df_['weekend'][inf_df_['weekday']>=5]=1
        inf_df_['weekday_'][inf_df_['weekday']<5]=1
        
        
        ### instlocation_a/z 
        inf_df_['root_cause_instlocationa'][inf_df_['root_cause_instlocationa']=='0'] = float('nan')
        inf_df_['root_cause_instlocationz'][inf_df_['root_cause_instlocationz']=='0'] = float('nan')
        
        ### sysname_a/z
        inf_df_['root_cause_sysnamea'][inf_df_['root_cause_sysnamea']=='0'] = float('nan')
        inf_df_['root_cause_sysnamez'][inf_df_['root_cause_sysnamez']=='0'] = float('nan')
        
        ### instlo_a/z divide
        inf_df_['instlo_a_1'] = inf_df_['root_cause_instlocationa'].apply(lambda x: x[:2] if type(x) == str else x)
        inf_df_['instlo_a_2'] = inf_df_['root_cause_instlocationa'].apply(lambda x: x[2:4] if type(x) == str else x)
        inf_df_['inst_code_a'] = inf_df_['root_cause_instlocationa'].apply(lambda x: x[4:] if type(x) == str else x)
        inf_df_ = inf_df_.drop(['root_cause_instlocationa'],axis=1)

        inf_df_['instlo_z_1'] = inf_df_['root_cause_instlocationz'].apply(lambda x: x[:2] if type(x) == str else x)
        inf_df_['instlo_z_2'] = inf_df_['root_cause_instlocationz'].apply(lambda x: x[2:4] if type(x) == str else x)
        inf_df_['inst_code_z'] = inf_df_['root_cause_instlocationz'].apply(lambda x: x[4:] if type(x) == str else x)
        inf_df_ = inf_df_.drop(['root_cause_instlocationz'],axis=1)
        
        ### sysname_a/z divide
        inf_df_['sysname_a_lo'] = inf_df_['root_cause_sysnamea'].apply(lambda x : x.split('-')[0] if type(x)==str else x)
        inf_df_['sysname_a_eq'] = inf_df_['root_cause_sysnamea'].apply(lambda x : x.split('-')[1] if type(x)==str else x)

        inf_df_['sysname_z_lo'] = inf_df_['root_cause_sysnamez'].apply(lambda x : x.split('-')[0] if type(x)==str else x)
        inf_df_['sysname_z_eq'] = inf_df_['root_cause_sysnamez'].apply(lambda x : x.split('-')[1] if type(x)==str else x)

        
        
        ### df
        inf_df_ = inf_df_[['ticket_id','fault_time', 'hour', 'minute', 'weekday', 'weekday_', 'weekend',
            'instlo_a_1','instlo_a_2', 'inst_code_a',
            'instlo_z_1','instlo_z_2', 'inst_code_z',
            'sysname_a_lo','sysname_a_eq',
            'sysname_z_lo','sysname_z_eq'
            ]]
        
        return inf_df_

    def ppr_le(self, inf_df_, trained_df_):
        ### label_encoding all column
        inf_df_['instlo_1'] = self.get_inf_le(inf_df_['instlo_1'], trained_df_['instlo_1'])
        inf_df_['instlo_2'] = self.get_inf_le(inf_df_['instlo_2'], trained_df_['instlo_2'])
        inf_df_['inst_code'] = self.get_inf_le(inf_df_['inst_code'], trained_df_['inst_code'])
        
        inf_df_['sysname_lo'] = self.get_inf_le(inf_df_['sysname_lo'], trained_df_['sysname_lo'])
        inf_df_['sysname_eq'] = self.get_inf_le(inf_df_['sysname_eq'], trained_df_['sysname_eq'])
        
#         inf_df_['ntt_label'] = self.get_inf_le(inf_df_['ntt_label'], trained_df_['ntt_label'])

        return inf_df_

    def ppr_train_le(self, train_df_):
        ### label_encoding all column
        train_df_['instlo_1'] = self.get_train_le(train_df_['instlo_1'])
        train_df_['instlo_2'] = self.get_train_le(train_df_['instlo_2'])
        train_df_['inst_code'] = self.get_train_le(train_df_['inst_code'])
        
        train_df_['sysname_lo'] = self.get_train_le(train_df_['sysname_lo'])
        train_df_['sysname_eq'] = self.get_train_le(train_df_['sysname_eq'])
        
        train_df_['ntt_label'] = self.get_train_le(train_df_['ntt_label'])

        return train_df_

    def ppr_2(self, inf_df_):
        inf_df_['instlo_1'] = inf_df_['instlo_z_1']
        inf_df_['instlo_1'][inf_df_['instlo_1'].isna()] = inf_df_['instlo_a_1'][inf_df_['instlo_1'].isna()]
        inf_df_['instlo_2'] = inf_df_['instlo_z_2']
        inf_df_['instlo_2'][inf_df_['instlo_2'].isna()] = inf_df_['instlo_a_2'][inf_df_['instlo_2'].isna()]
        inf_df_['inst_code'] = inf_df_['inst_code_z']
        inf_df_['inst_code'][inf_df_['inst_code'].isna()] = inf_df_['inst_code_a'][inf_df_['inst_code'].isna()]
        
        inf_df_['sysname_lo'] = inf_df_['sysname_z_lo']
        inf_df_['sysname_lo'][inf_df_['sysname_lo'].isna()] = inf_df_['sysname_a_lo'][inf_df_['sysname_lo'].isna()]

        inf_df_['sysname_eq'] = inf_df_['sysname_z_eq']
        inf_df_['sysname_eq'][inf_df_['sysname_eq'].isna()] = inf_df_['sysname_a_eq'][inf_df_['sysname_eq'].isna()]

        inf_df_['time'] = inf_df_['hour'] + inf_df_['minute']/60
        

        inf_df_ = inf_df_[['ticket_id','fault_time','time', 'weekday', 'weekend', 
                'instlo_1', 'instlo_2', 'inst_code', 
                'sysname_lo', 'sysname_eq'
                ]]
        
        return inf_df_

    def ppr_2_train(self, inf_df_):
        inf_df_['instlo_1'] = inf_df_['instlo_z_1']
        inf_df_['instlo_1'][inf_df_['instlo_1'].isna()] = inf_df_['instlo_a_1'][inf_df_['instlo_1'].isna()]
        inf_df_['instlo_2'] = inf_df_['instlo_z_2']
        inf_df_['instlo_2'][inf_df_['instlo_2'].isna()] = inf_df_['instlo_a_2'][inf_df_['instlo_2'].isna()]
        inf_df_['inst_code'] = inf_df_['inst_code_z']
        inf_df_['inst_code'][inf_df_['inst_code'].isna()] = inf_df_['inst_code_a'][inf_df_['inst_code'].isna()]
        
        inf_df_['sysname_lo'] = inf_df_['sysname_z_lo']
        inf_df_['sysname_lo'][inf_df_['sysname_lo'].isna()] = inf_df_['sysname_a_lo'][inf_df_['sysname_lo'].isna()]

        inf_df_['sysname_eq'] = inf_df_['sysname_z_eq']
        inf_df_['sysname_eq'][inf_df_['sysname_eq'].isna()] = inf_df_['sysname_a_eq'][inf_df_['sysname_eq'].isna()]

        inf_df_['time'] = inf_df_['hour'] + inf_df_['minute']/60
        

        inf_df_ = inf_df_[['ticket_id','fault_time','time', 'weekday', 'weekend', 
                'instlo_1', 'instlo_2', 'inst_code', 
                'sysname_lo', 'sysname_eq', 
                'ntt_label']]
        
        return inf_df_

    
    def run_(self):
        df = self.ppr_inf(self.inf_df)
        df = self.ppr_2(df)
        # label encoding 되지 않은 데이터
        df_b = df[:]
        df_b.drop_duplicates(ignore_index=True, inplace=True)
        df = self.ppr_le(df, self.trained_df)
        df_ = df.drop_duplicates(ignore_index=True)

        return df_, df_b

    def run_train_(self):
        df = self.ppr__train(self.inf_df)
        df = self.ppr_2_train(df)
        df.to_csv('/workspace/dr.Lauren/210729_/train_data/new_train_set.csv')
        df = self.ppr_train_le(df)
        df_ = df.drop_duplicates(ignore_index=True)

        return df_
# -


