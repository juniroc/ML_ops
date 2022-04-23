import os
import sys
import pandas as pd
import numpy as np
import pickle
import yaml
import random
import torch


import argparse


# getting yaml config that has a all parameter path parameter from argparse
parser = argparse.ArgumentParser(description='get argument from argparse')
parser.add_argument('--config-path', type=str, help='config file path', default='./')


args = parser.parse_args()

config_path_ = args.config_path
opt = yaml.load(open(config_path_), Loader=yaml.FullLoader)


## seed initialize
torch.manual_seed(opt['seed'])
# torch.cuda.manual_seed_all(opt['seed'])   # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(opt['seed'])
random.seed(opt['seed'])


class preprocessing:
    def __init__(self, data_path_, version_dir_):
        self.data_path = data_path_
        self.version_dir = version_dir_

    # get dataset format from csv or ftr
    def _get_dataset(self):
        _, extension = os.path.splitext(self.data_path)

        data_format = extension[1:]

        try:
            if data_format=='csv':
                df = pd.read_csv(self.data_path)

            elif data_format=='ftr':
                df = pd.read_feather(self.data_path)
        
        except ValueError:
            print("you can choose csv or ftr for data format")
        
        return df

    def _remove_null_data(self, df_):
        for i in df_.columns:
            df_ = df_[df_[i]!=' ']    # remove ' ' (empty space) value

        df_.dropna(inplace=True)
        df_.reset_index(drop=True, inplace=True)

        return df_

    # change 4~5 to 1, and 0~3 to 0
    def _change_values(self, df_):
        df_['Effectiveness'] = df_['Effectiveness'].apply(lambda x: 0 if (x<4) else 1)
        df_['EaseofUse'] = df_['EaseofUse'].apply(lambda x: 0 if (x<4) else 1)
        df_['Satisfaction'] = df_['Satisfaction'].apply(lambda x: 0 if (x<4) else 1)

        return df_

    def preprocess(self):
        df_ = self._get_dataset()
        df_ = self._remove_null_data(df_)
        df_ = self._change_values(df_)

        # get `patient_id` column
        df_['patient_id'] = [i for i in range(len(df_))]

        ### tmp_dir_path : /workspace/gnn/python_files/datas/v_1
        exist_dir = os.path.exists(self.version_dir)

        if not exist_dir:
            os.makedirs(self.version_dir)

        
        df_.to_pickle(f'{self.version_dir}/DataFrame.pickle')
        return df_



if __name__ == '__main__':
    pre_ = preprocessing(opt['dataset_path'], opt['version_dir'])
    pre_.preprocess()