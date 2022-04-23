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
    def __init__(self, m_ratio_, input_f_, hidden_f_, output_f_, version_dir_):
        self.mask_ratio = m_ratio_
        self.input_f = input_f_
        self.hidden_f = hidden_f_
        self.output_f = output_f_
        self.version_dir = version_dir_

    def _get_train_graph(self):
        with open(f'{self.version_dir}/hetero_graph_t.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            train_graph_t = pickle.load(f)

        return train_graph_t


    def train(self):
        hetero_graph_t = self._get_train_graph()

        # edge length
        num_edges = len(hetero_graph_t.edata['label'][('patient', 'satisfaction', 'drug')])

        # train / validation masking
        train_mask = torch.zeros(num_edges, dtype=torch.bool).bernoulli(self.mask_ratio)
        val_mask = ~train_mask

        dec_graph_t = hetero_graph_t['patient', :, 'drug']

        label_arr_t = hetero_graph_t.edges['satisfaction'].data['label']

        model = model_file.Model(self.input_f, self.hidden_f, self.output_f, hetero_graph_t.etypes, True)

        node_features_t = {
            'patient': hetero_graph_t.nodes['patient'].data['feature'],
            'drug':hetero_graph_t.nodes['drug'].data['feature'],
            'condition':hetero_graph_t.nodes['condition'].data['feature']
            }
        
        opt = torch.optim.Adam(model.parameters())

        model.train()
        for epoch in range(300):
            logits = model(hetero_graph_t, node_features_t, dec_graph_t)
            loss = F.cross_entropy(logits[train_mask], label_arr_t[train_mask])
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            if epoch % 5 == 0:
                acc_val = torchmetrics.functional.accuracy(logits[val_mask], label_arr_t[val_mask])
                print(f"--------- {epoch} ---------")
                print('val_acc : ', acc_val)

    
        torch.save(model, f'{self.version_dir}/model.pth')



if __name__ == '__main__':
    test = model_(opt['m_ratio'], opt['input_f'], opt['hidden_f'], opt['output_f'], opt['version_dir'])
    test.train()