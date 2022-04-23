import os
import sys
import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import pickle
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



# create graph and training
class create_graph:
    def __init__(self, version_dir_, train_inf_):
        self.version_dir = version_dir_    # parameter version path
        self.train =  train_inf_   # choose train or inference


    # get dataset format from csv or ftr
    def _get_dataset(self):
        
        df = pd.read_pickle(f'{self.version_dir}/DataFrame.pickle')

        return df

    # get index each column
    def _get_dict(self, df, column: any):
        index = 0
        # values list
        val_lst = [i for i in df[column].value_counts().index]
        
        dictionary = {}
        
        # value list to dictionary {key : value of column, value : index}
        for i in val_lst:
            dictionary[i] = index
            index += 1
        
        return dictionary

    # get drug_feature by side effect
    def _get_drug_feat_dict(self, df_, drug_dictionary):
        # drug_side_dict
        drug_side_dict = {}

        ### most frequency side effect of each drug (dictionary)
        for i in drug_dictionary.keys():
            most_side = df_[df_['DrugId']==i]['Sides'].value_counts().index[0]
            drug_side_dict[i] = most_side


        # side_dict (Cause of sides duplicate, using set)
        side_dict = {}
        index = 0
        side_set = set()
        side_set.update(drug_side_dict.values())

        # side effect dictionary
        for j in side_set:
            side_dict[j] = index
            index += 1

        # drug_feat_dict
        drug_feat_dict = {}

        for i in drug_side_dict.keys():
            drug_feat_dict[drug_dictionary[i]] = side_dict[drug_side_dict[i]]

        return drug_side_dict, side_dict, drug_feat_dict


    def _get_n_arr(self, dataframe, dictionary, column):    
        num_lst = [int(dictionary[i]) for i in dataframe[column]]
        
        return np.array(num_lst)

    # save dictionary for preprocessing when training
    def _save_dict(self, dictionary, dir_path):
        
        ### tmp_dir_path : /workspace/gnn/python_files/datas/v_1
        exist_dir = os.path.exists(dir_path)

        if not exist_dir:
            os.makedirs(dir_path)

        with open(f'{dir_path}/total_dict.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save to npy file
        for i in dictionary.keys():
            if type(i).__name__=='ndarray':
                np.save(f'{dir_path}/{i}.npy', dictionary[i])

            else:
                with open(f'{dir_path}/{i}.pickle', 'wb') as handle:
                    pickle.dump(dictionary[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load dictionary for preprocessing when do inference
    def _load_dict(self, dir_path):
        with open(f'{dir_path}/total_dict.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            dict_ = pickle.load(f)

        return dict_

    # embedding features
    def _get_embed(self, len_keys, lst, embed_n):

        embedding_table = nn.Embedding(num_embeddings=len_keys, 
                                embedding_dim=embed_n)

        
        embed_feat = embedding_table(torch.LongTensor(lst))

        return embed_feat


    def get_graph(self):
        if self.train=='train':
            
            df_ = self._get_dataset()
            
            drug_dict = self._get_dict(df_, 'DrugId')
            patient_feat_dict = self._get_dict(df_, ['Age', 'Sex'])
            cond_dict = self._get_dict(df_, 'Condition')

            drug_side_dict, side_dict, drug_feat_dict = self._get_drug_feat_dict(df_, drug_dict)

            train_dict = {'drug_dict':drug_dict, 'patient_feat_dict':patient_feat_dict, 'cond_dict':cond_dict, 'drug_side_dict':drug_side_dict, 'side_dict':side_dict, 'drug_feat_dict':drug_feat_dict}
            
            ## get node_array
            patient_arr = np.array(df_['patient_id'])

            # mapping using dictionary
            drug_arr = self._get_n_arr(df_, drug_dict, 'DrugId')
            cond_arr = self._get_n_arr(df_, cond_dict, 'Condition')

            train_dict.update({'patient_arr':patient_arr, 'drug_arr':drug_arr, 'cond_arr':cond_arr})

            ## label_edge_array
            label_arr = torch.tensor(df_['Satisfaction'])
            train_dict.update({'satisfaction':label_arr})

            ### TRAIN data
            hetero_graph_t = dgl.heterograph({
                ('patient', 'satisfaction', 'drug'): (patient_arr, drug_arr),
                ('condition', 'symptom', 'patient'): (cond_arr, patient_arr),
                ('drug', 'Easy', 'patient'): (drug_arr[df_['EaseofUse']==1], patient_arr[df_['EaseofUse']==1]),
                ('drug', 'Effectiveness', 'condition'): (drug_arr[df_['Effectiveness']==1], cond_arr[df_['Effectiveness']==1])
                })

            ###  Train node feature embedding table
            # patient
            patient_f_lst = [patient_feat_dict[(df_['Age'][i], df_['Sex'][i])] for i in range(len(df_))]
            # patient embedding
            patient_embed = self._get_embed(len(patient_feat_dict.keys()), patient_f_lst, 10)    # ( 280127 * 10 ) -> 22 types

            # condition
            cond_f_lst = [i for i in range(len(cond_dict.values()))]
            # condition embedding
            cond_embed = self._get_embed(len(cond_dict.values()), cond_f_lst, 10)    # ( 1584 * 10 ) -> 1584 Condition types

            # drug
            drug_f_lst = [i for i in drug_feat_dict.values()]
            # drug embedding
            drug_embed = self._get_embed(len(drug_feat_dict.keys()), drug_f_lst, 10)   # ( 4522 * 10 ) -> 4522 drug types and 1557 side effect types
            

            train_dict.update({'patient_embed':patient_embed, 'cond_embed':cond_embed, 'drug_embed':drug_embed})

            # get nodes features and edge label
            hetero_graph_t.edges['satisfaction'].data['label'] = label_arr
            hetero_graph_t.nodes['patient'].data['feature'] = patient_embed
            hetero_graph_t.nodes['drug'].data['feature'] = drug_embed
            hetero_graph_t.nodes['condition'].data['feature'] = cond_embed

            train_dict.update({'hetero_graph_t':hetero_graph_t})

            self._save_dict(train_dict, self.version_dir)

            return hetero_graph_t

        elif self.train=='inference':
            
            df_ = self._get_dataset()
            
            # load dictionaries
            total_dict = self._load_dict(self.version_dir)

            ### added_inference array
            patient_arr_inf = np.array(df_['patient_id'])
            try:
                drug_arr_inf = self._get_n_arr(df_, total_dict['drug_dict'], 'DrugId')
            
            except ValueError:
                print('there is no DrugId in dictionary, Try training or Check dictionary some keys')

            try:
                cond_arr_inf = self._get_n_arr(df_, total_dict['cond_dict'], 'Condition')
            
            except ValueError:
                print('there is no Condition in dictionary, Try training or Check dictionary some keys')

            # label concat with train - inference
            label_arr_inf = torch.tensor(list(df_['Satisfaction']))

            label_arr_inf = torch.cat([total_dict['satisfaction'], label_arr_inf], 0)


            ### add nodes and edges
            hetero_graph_inf = dgl.add_edges(total_dict['hetero_graph_t'], patient_arr_inf, drug_arr_inf, etype='satisfaction')
            hetero_graph_inf = dgl.add_edges(hetero_graph_inf, cond_arr_inf, patient_arr_inf, etype='symptom')
            hetero_graph_inf = dgl.add_edges(hetero_graph_inf, drug_arr_inf[df_['EaseofUse']==1], patient_arr_inf[df_['EaseofUse']==1], etype='Easy')
            hetero_graph_inf = dgl.add_edges(hetero_graph_inf, drug_arr_inf[df_['Effectiveness']==1], cond_arr_inf[df_['Effectiveness']==1], etype='Effectiveness')

            # get inference node feature embedding
            patient_embed_inf = total_dict['patient_embed']
            drug_embed_inf = total_dict['drug_embed']
            cond_embed_inf = total_dict['cond_embed']

            hetero_graph_inf.edges['satisfaction'].data['label'] = label_arr_inf
            hetero_graph_inf.nodes['patient'].data['feature'] = patient_embed_inf
            hetero_graph_inf.nodes['drug'].data['feature'] = drug_embed_inf
            hetero_graph_inf.nodes['condition'].data['feature'] = cond_embed_inf

            inf_dict_={}
            inf_dict_['hetero_graph_inf']=hetero_graph_inf
            inf_dict_['inf_length']=len(df_)
            print(hetero_graph_inf)
            with open(f'{self.version_dir}/hetero_graph_inf.pickle', 'wb') as handle:
                pickle.dump(inf_dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return hetero_graph_inf, len(df_)

        else:
            print("you should check train or inference")


if __name__ == '__main__':
    creating_graph = create_graph(opt['version_dir'], opt['train'])
    hetero_graph_t = creating_graph.get_graph()

    print('creating graph done')