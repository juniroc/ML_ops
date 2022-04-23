import os
import sys
import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import torchmetrics
import pickle
import model_file
import yaml
import random

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



class model_:
    def __init__(self, version_dir_):
        self.version_dir = version_dir_


    def _get_inf_graph(self):
        with open(f'{self.version_dir}/hetero_graph_inf.pickle', 'rb') as f:
            inf_graph_dict_ = pickle.load(f)

        return inf_graph_dict_


    def infer(self):
        inf_dict_ = self._get_inf_graph()

        # Create Hetero Graph for Inference
        hetero_graph_inf = inf_dict_['hetero_graph_inf']
        inf_length = inf_dict_['inf_length']

        dec_graph_inf = hetero_graph_inf['patient', :, 'drug']

        label_arr_inf = hetero_graph_inf.edges['satisfaction'].data['label']

        node_features_inf = {
            'patient': hetero_graph_inf.nodes['patient'].data['feature'],
            'drug':hetero_graph_inf.nodes['drug'].data['feature'],
            'condition':hetero_graph_inf.nodes['condition'].data['feature']
            }

        # Load model
        model = torch.load(f'{self.version_dir}/model.pth')

        model.eval()

        with torch.no_grad():
            test_logit = model(hetero_graph_inf, node_features_inf, dec_graph_inf)
        
        inference_acc = torchmetrics.functional.accuracy(test_logit[inf_length:], label_arr_inf[inf_length:])
        print("result : ", test_logit[inf_length:])
        print("acc : ", inference_acc.item())


if __name__ == '__main__':
    test = model_( opt['version_dir'])

    test.infer()