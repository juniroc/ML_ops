# Airflow - K8sPodOperator

## 0. Overview

- DockerOperator 와 비슷하지만, 각 프로세스가 POD 위에서 돌아가도록 Pipeline 구축
- DockerOperator 에서 쓰인 `config.yml`, `dataset`, `versions(directory)` 그대로 이용
- Data Pipeline 구축을 목표로 함
→ 학습 및 인퍼런스 데이터셋, 모델 구조에 대해서는 깊게 다루지않음(모델 학습이 목표가 아니므로 학습 데이터셋과, 인퍼런스 데이터셋은 같은 데이터를 이용함)

## 1. Prerequisite

- K8s Cluster install
    - K8s `~/.kube/config` 파일을 통해 접근이 가능해야함
    - Cmd 에 `kubectl cluster-info` 를 입력했을 때 아래와 같은 로그가 나오면 완료
    
    ```yaml
    Kubernetes control plane is running at https://<Cluster IP>:<Port>
    KubeDNS is running ~~ <Cluster ip>:<port>/api/v1/namespace ~~~
    ```
    

## 2. Process python and yaml file

### config yaml files

`config_train.yaml`

```yaml
dataset_path: ./dataset/webmd.ftr

# train: train or inference
train: train

version_dir: ./versions/v_1

m_ratio: 0.8
input_f: 10
hidden_f: 20
output_f: 5

seed: 123
```

- 학습시 이용될 config_file
- 학습할 데이터의 dataset_path 와 현재 데이터셋의 version 을 저장할 version_dir 을 지정
- 이후 학습에 사용될 parameter 값들을 지정

`config_inf.yaml`

```yaml
dataset_path: ./dataset/webmd_inf.ftr

# train: train or inference
train: inference

version_dir: ./versions/v_1

seed: 123
```

- 추론시 이용될 config_File
- 추론하기 위해서는 `train: inference` 로 지정
- 추론할 경우, dataset_path 와 version 만 필요하므로 이후 필요없는 parameter 들은 제거

### Dataset Load and Preprocessing file

`preprocessing.py`

```yaml
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
```

- config-path 를 받아서 Parameter setting 및 Train/Inference 환경을 지정
- 이후 전처리된 데이터를 version_dir 에 pickle 형태로 저장 (해당 version_dir 은 로컬 directory 와 마운트 되어있음)
- 해당 version_dir 및 dataset path 는 config file 에서 지정

### Create Graph

`create_graph.py`

```yaml
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
```

- 위에서 정의한 config-path 로 부터 config 데이터를 받아옴
- train or inference 경우에 따라 다른 그래프 생성
- preprocessed dataset 은 이전 프로세스에서 저장된 version_dir 에 존재하는 pickle을 가져옴 (해당 version_dir 은 로컬 directory 와 마운트 되어있음)
- 생성된 그래프는 version_dir 에 pickle 로 저장 (해당 version_dir 은 로컬 directory 와 마운트 되어있음)

---

### training

`training.py`

```yaml
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
```

`model_file.py`

```yaml
import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import torchmetrics
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__() 
        self.W = nn.Linear(in_dims * 2, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.W(x)
        return {'score': y}

    def forward(self, graph, h):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, bi_pred=False):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        
        if bi_pred==False:
            self.pred = HeteroMLPPredictor(out_features, len(rel_names))
        else:
            self.pred = HeteroMLPPredictor(out_features, 2)
        
    def forward(self, g, x, dec_graph):
        h = self.sage(g, x)
        h_2 = {'drug': h['drug'], 'patient': h['patient']}
        return self.pred(dec_graph, h_2)
```

- `import model_file` 을 통해 저장된 `model_file.py` 에서 모델의 정보를 불러옴
- config-path 로 부터 학습에 필요한 parameter 정보를 받음
- 학습이 완료된 모델은 model.pth 파일로 version_dir 에 저장됨 (해당 version_dir 은 로컬 directory 와 마운트 되어있음)

### Inference

- 위 process 중 1, 2 번은 일치하며, 데이터셋 위치만 재설정을 통해 inference dataset 불러옴
- version_dir 을 통해 해당 모델 및 데이터 inference 용 Graph 를 불러옴 (해당 version_dir 은 로컬 directory 와 마운트 되어있음)

`Inference.py`

```yaml
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
```

- 이전 프로세스에서 생성된 inference graph 를 version_dir 에서 가져옴 (해당 version_dir 은 로컬 directory 와 마운트 되어있음)
- 새로 생성된 Heterogeneous Graph 를 version_dir 에서 pickle 형태로 받음
- 위에서 가져온 Graph 를 이용해 inference
    - accuracy 를 출력하는 형태 (여기선 따로 저장하진 않음)

### Creating it Container (Dockerfile)

---

### Dockerfile 및 requirement.txt

- image 생성 후 docker hub 에 푸쉬

`Dockerfile`

```yaml
# 베이스 이미지 불러오기
FROM python:3.8.12-slim

# 작업 디렉토리 설정
WORKDIR /gnn_py

# 현재 디렉토리 안의 모든 파일 작업 디렉토리로 이동
## python 파일 및 requirements
COPY . /gnn_py

# 권한 재설정
RUN chmod -R 755 /gnn_py

# 버전 디렉토리, Config 디렉토리 생성
RUN mkdir /gnn_py/versions
RUN mkdir /gnn_py/config_files
RUN mkdir /gnn_py/dataset

# vim 및 리눅스 패키지 인스톨
RUN apt update && apt install -y vim
RUN apt install -y procps

# 패키지 설치 & dgl 설치
RUN pip install --upgrade pip
RUN pip install dgl -f dgl -f https://data.dgl.ai/wheels/repo.html
RUN pip install -r requirements.txt

# KST로 시간대 변경
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
```

`requirement.txt`

```yaml
numpy==1.20.3
pandas==1.3.4
PyYAML==6.0
torch==1.11.0
torchmetrics==0.7.3
pyarrow==7.0.0
gunicorn==20.0.4
```

- 여기서 조심해야할 것은 dgl 의 경우 requirement.txt 에서 install 할 수 없음
- Dockerfile 자체에서 RUN 명령어를 통해 install
- 모든 파이썬 파일을 container 내부로 이동
- docker build 및 Push

```yaml
docker build . <id>:<image_name>

docker push <id>:<image_name>
```

---

- 윗 부분까지는 DockerOperator와 같음

**차이점**

- 볼륨 마운트 방식
- Process 처리 방식

## Airflow - K8sPodOperator

---

- Pod 형태로 프로세스를 띄워주기 때문에 PV/PVC 를 생성해 Pod 와 원하는 Node 의 directory 를 연결
- PV의 StorageClass 방식은 NFS 방식으로 특정 Node 의 특정 directory로 마운트

### 0. NFS 설정

- 마운트되길 원하는 Node의 `etc/exports` 에서 마운트할 directory 정의

`/etc/exports`

```yaml
# /etc/exports: the access control list for filesystems which may be exported
#		to NFS clients.  See exports(5).
#
# Example for NFSv2 and NFSv3:
# /srv/homes       hostname1(rw,sync,no_subtree_check) hostname2(ro,sync,no_subtree_check)
#
# Example for NFSv4:
# /srv/nfs4        gss/krb5i(rw,sync,fsid=0,crossmnt,no_subtree_check)
# /srv/nfs4/homes  gss/krb5i(rw,sync,no_subtree_check)

/home/minjun/airflow_k8s *(rw,no_root_squash,async)
```

---

### 1. StorageClass 생성

`nfs_storage_class.yml`

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: air-k8s
provisioner: example.com/external-nfs
parameters:
  server: <Node IP>  ## master node 인 7번 서버 이용
  path: /home/minjun/airflow_k8s
  readOnly: "false"
```

`kubectl apply -f nfs_storage_class.yml` 을 이용해 띄워줌 

- pv와 연결해줄 StorageClass 생성

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled.png)

- 위와같이 띄워졌으면 완료

---

### 2. PV 생성

`pv-volume.yml`

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: airflow-pv
  labels:
    type: local
spec:
  storageClassName: air-k8s
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  nfs:
    path: "/home/minjun/airflow_k8s/config_files" # 마운트하고자하는 위치
    server: <Server IP> # 원하는 노드의 IP 를 작성
```

- 위 `yaml file` 은 config_files 을 마운트하기 위해 작성
- `spec.storageClassName` 을 위에서 작성한 `StorageClass.yml` 이름과 일치시켜줌

`pv-volume-data.yml`

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: airflow-dataset
  labels:
    type: local
spec:
  storageClassName: air-k8s
  capacity:
    storage: 3Gi
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  nfs:
    path: "/home/minjun/airflow_k8s/dataset"
    server: <Server IP> # 원하는 노드의 IP 를 작성
```

- 위 `yaml file` 은 dataset directory를 마운트하기 위해 작성
- `metadata.name` 을 다르게 구성
- `spec.storageClassName` 을 위해서 작성한 `StorageClass.yml` 이름과 일치시켜줌

`pv-volume-version.yml`

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: airflow-dataset
  labels:
    type: local
spec:
  storageClassName: air-k8s
  capacity:
    storage: 3Gi
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  nfs:
    path: "/home/minjun/airflow_k8s/versions"
    server: <Server IP> # 원하는 노드의 IP 를 작성
```

- 위 `yaml file` 은 versions directory를 마운트하기 위해 작성
- `metadata.name` 을 다르게 구성
- `spec.storageClassName` 을 위해서 작성한 `StorageClass.yml` 이름과 일치시켜줌

---

### 3. PVC 생성

- PVC 는 `namespace` 를 지정

`pvc-volume.yml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-pv
  namespace: airflow
spec:
  storageClassName: air-k8s
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 3Gi
```

- 위에서 작성한 PV와 일치하는 PVC 생성
- `metadata.name` 을 일치시켜주면됨

`pvc-volume-data.yml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-dataset
  namespace: airflow
spec:
  storageClassName: air-k8s
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 3Gi
```

- Dataset PV 와 일치하는 PVC 생성

`pvc-volume-version.yml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-version
  namespace: airflow
spec:
  storageClassName: air-k8s
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 3Gi
```

- Version PV 와 일치하는 PVC 생성

- 작성한 yaml 파일 실행

```yaml
kubectl apply -f <yaml file name>
```

- 아래와 같이 Bound 되었다면 완료

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%201.png)

`docker_operator.py`

```yaml
import datetime as dt
import os

from kubernetes.client import models as k8s

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (KubernetesPodOperator,)

with DAG(
    dag_id="k8s_test",
    start_date=dt.datetime(2022, 4, 24),
    end_date=dt.datetime(2022, 5, 1),
    schedule_interval="@daily",
) as dag:

    volume_claim = k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="airflow-pv"
    )

    volume = k8s.V1Volume(
        name="airflow-pv",
        persistent_volume_claim=volume_claim
    )

    volume_mount = k8s.V1VolumeMount(
        name="airflow-pv",
        mount_path="/gnn_py/config_files"
    )

    volume_claim_dataset = k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="airflow-dataset"
    )

    volume_dataset = k8s.V1Volume(
        name="airflow-dataset",
        persistent_volume_claim=volume_claim_dataset
    )

    volume_mount_dataset = k8s.V1VolumeMount(
        name="airflow-dataset",
        mount_path="/gnn_py/dataset"
    )

    volume_claim_version = k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="airflow-version"
    )

    volume_version = k8s.V1Volume(
        name="airflow-version",
        persistent_volume_claim=volume_claim_version
    )

    volume_mount_version = k8s.V1VolumeMount(
        name="airflow-version",
        mount_path="/gnn_py/versions"
    )

    preprocessing = KubernetesPodOperator(
        task_id="load_and_preprocessing",
        image="lmj3502/airflow_gnn",
        cmds=["python", "/gnn_py/preprocessing.py", "--config-path", "/gnn_py/config_files/config_train.yml"],
        namespace="airflow",
        name="load_and_preprocessing",
        cluster_context="kubernetes-admin@kubernetes",
        in_cluster=False,
        volumes=[volume, volume_dataset, volume_version],
        volume_mounts=[volume_mount, volume_mount_dataset, volume_mount_version],
        is_delete_operator_pod=True,
    )

    create_graph = KubernetesPodOperator(
        task_id="create_graph",
        image="lmj3502/airflow_gnn",
        cmds=["python", "create_graph.py", "--config-path", "/gnn_py/config_files/config_train.yml"],
        namespace="airflow",
        name="create_graph",
        cluster_context="kubernetes-admin@kubernetes",
        in_cluster=False,
        volumes=[volume, volume_dataset, volume_version],
        volume_mounts=[volume_mount, volume_mount_dataset, volume_mount_version],
        is_delete_operator_pod=True,
    )

    training = KubernetesPodOperator(
        task_id="training",
        image="lmj3502/airflow_gnn",
        cmds=["python", "training.py", "--config-path", "/gnn_py/config_files/config_train.yml"],
        namespace="airflow",
        name="training",
        cluster_context="kubernetes-admin@kubernetes",
        in_cluster=False,
        volumes=[volume, volume_dataset, volume_version],
        volume_mounts=[volume_mount, volume_mount_dataset, volume_mount_version],
        is_delete_operator_pod=True,
    )

    preprocessing_inf = KubernetesPodOperator(
        task_id="load_and_preprocessing_inf",
        image="lmj3502/airflow_gnn",
        cmds=["python", "preprocessing.py", "--config-path", "/gnn_py/config_files/config_inf.yml"],
        namespace="airflow",
        name="inference_dataset_preprocessing",
        cluster_context="kubernetes-admin@kubernetes",
        in_cluster=False,
        volumes=[volume, volume_dataset, volume_version],
        volume_mounts=[volume_mount, volume_mount_dataset, volume_mount_version],
        is_delete_operator_pod=True,
    )

    create_graph_inf = KubernetesPodOperator(
        task_id="create_graph_inf",
        image="lmj3502/airflow_gnn",
        cmds=["python", "create_graph.py", "--config-path", "/gnn_py/config_files/config_inf.yml"],
        namespace="airflow",
        name="create_inference_graph",
        cluster_context="kubernetes-admin@kubernetes",
        in_cluster=False,
        volumes=[volume, volume_dataset, volume_version],
        volume_mounts=[volume_mount, volume_mount_dataset, volume_mount_version],
        is_delete_operator_pod=True,
    )

    inference = KubernetesPodOperator(
        task_id="inferecne",
        image="lmj3502/airflow_gnn",
        cmds=["python", "inference.py", "--config-path", "/gnn_py/config_files/config_inf.yml"],
        namespace="airflow",
        name="inference",
        cluster_context="kubernetes-admin@kubernetes",
        in_cluster=False,
        volumes=[volume, volume_dataset, volume_version],
        volume_mounts=[volume_mount, volume_mount_dataset, volume_mount_version],
        is_delete_operator_pod=True,
    )

    preprocessing >> create_graph >> training
    preprocessing >> preprocessing_inf 
    [preprocessing_inf, create_graph] >> create_graph_inf 
    [create_graph_inf, training]>> inference
```

- DockerOperator와 같이 내부에 mounts 옵션을 통해 config_files, dataset, versions directory 를 마운트
- 해당 파일은 airflow 환경의 `~/airflow/dags` 로 옮긴 후 실행

- 프로세스 순서와 dependency process를 지정해줄 수 있다.

```yaml
    preprocessing >> create_graph >> training
    preprocessing >> preprocessing_inf 
    [preprocessing_inf, create_graph] >> create_graph_inf 
    [create_graph_inf, training]>> inference
```

- preprocessing 이 끝나야 create_graph 프로세스가 진행되고, create_graph 가 끝나야 training이 가능
- preprocessing_inf 는 preprocessing 이 끝나야 진행되고
- create_graph_inf 의 경우 preprocessing_inf, create_graph 가 모두 끝나야 실행

---

### Pipeline Monitoring

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%202.png)

- 실패하거나 성공한 경우 다음과 같이 Monitoring 가능

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%203.png)

- 실패한 부분이나 성공한 부분을 클릭하여 Log를 찍어볼 수있음

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%204.png)

- `training` 부분의 `log`

- 아래와 같이 Pod 가 생성됨을 알 수 있음

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%205.png)

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%206.png)

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%207.png)

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%208.png)

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%209.png)

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%2010.png)

- (위 두개의 사진은 K9s 로 확인)

- 그래프 형태로도 모니터링 가능

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%2011.png)

![Untitled](Airflow%20-%20K8sPodOperator%2079da55cd607c434d87e70a3509b16021/Untitled%2012.png)