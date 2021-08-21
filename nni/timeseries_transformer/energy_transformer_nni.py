# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import os
import sys
import torch
import torch.nn.functional as F
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from pytorch_forecasting.metrics import SMAPE, MAE
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules   import Module
from torch.nn import init
import math
from tensorboardX import SummaryWriter

import nni
import argparse
import logging
import logging.handlers

from utils_ import save_checkpoint, AverageMeter, ProgressMeter, adjust_learning_rate, accuracy, LabelSmoothingLoss, EMA

# ### NNI_Argparse

# +
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--model-dir', default='/tmp', type=str)
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='(default: 32) this is the total ')
parser.add_argument('-e', '--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--seed', default=42, type=int,
                    help='the number of seed')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--request-from-nni', default=False, action='store_true')
parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

# optimizer_env
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('-g','--gamma', default=0.7, type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--moving-average-decay', default=0.9999, type=float)
parser.add_argument('--dont-adjust-learning-rate', default=False, action='store_true')

# Transformer parser_argument
parser.add_argument('-ts','--time-steps', default=24, type=int)
parser.add_argument('-nh','--num-head', default=8, type=int)
parser.add_argument('-em','--embed-dim', default=256, type=int)
parser.add_argument('-ne','--num-encoderlayers', default=6, type=int)
parser.add_argument('-nd','--num-decoderlayers', default=6, type=int)
parser.add_argument('-d','--dropout', default=0.2, type=int)
# -

head = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=head)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# +
def make_seq(df, seq_len):
    seqs=[]
    for i in range(len(df)-seq_len+1):
        seq = df.iloc[i:i+seq_len]
        seq = seq.to_numpy()
        seq = np.transpose(seq)
        seqs.append(seq)
    return seqs

# Mask for attention
def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)
# future_mask(50)


# +
#Model 
class FFN(nn.Module):
    def __init__(self, state_size=200, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


# Mask for attention
def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class Energy_Model(nn.Module):
    def __init__(self, 
                 y_max,
                 val_maxs,
                 time_steps=24, 
                 n_head = 8, 
                 embed_dim=128,
                 num_encoding_layers=2,
                 num_decoding_layers=2,
                 dropout=0.2):
        super(Energy_Model, self).__init__()
        
        self.time_steps = time_steps
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.num_encoding_layers = num_encoding_layers
        self.num_decoding_layers = num_decoding_layers
        self.dropout = dropout

        ## embedding layer
        self.pos_embedding= nn.Embedding(self.time_steps+1, self.embed_dim) ## position
        
        # building info
        self.num_emb = nn.Embedding(61, self.embed_dim)
        self.cool_emb = nn.Embedding(3, self.embed_dim)
        self.sun_e_emb = nn.Embedding(3, self.embed_dim)
        
        # date, hour
        self.doy_emb = nn.Embedding(300+1, self.embed_dim) # day of year.   # TODO: compress range
        self.dow_emb = nn.Embedding(7+1, self.embed_dim) # day of week
        self.wd_emb = nn.Embedding(2+1, self.embed_dim) # weekday or not
        self.hr_emb = nn.Embedding(24+1, self.embed_dim) # hour of day

        # input for t, w, h, r ,s
        self.t_embs = nn.Embedding(val_maxs[0]+1, self.embed_dim)
        self.w_embs = nn.Embedding(val_maxs[1]+1, self.embed_dim)
        self.h_embs = nn.Embedding(val_maxs[2]+1, self.embed_dim)
        self.r_embs = nn.Embedding(val_maxs[3]+1, self.embed_dim)
        self.s_embs = nn.Embedding(val_maxs[4]+1, self.embed_dim)

    
        self.energy_emb = nn.Embedding(y_max+200, self.embed_dim)

        #transformer
        self.transformer = nn.Transformer(nhead=self.n_head, 
                                          d_model = self.embed_dim, 
                                          num_encoder_layers= self.num_encoding_layers, 
                                          num_decoder_layers= self.num_decoding_layers, 
                                          dropout = self.dropout)

        self.dropout = nn.Dropout(self.dropout)
        self.layer_normal = nn.LayerNorm(self.embed_dim) 
        self.ffn = FFN(self.embed_dim)
        self.pred = nn.Linear(self.embed_dim, 1)


    def forward(self, num, temp, wind, humid, rain, sun, cool, sun_e, doy, dow, wd, hr, energy_, debug=False):
        device = num.device
        
        # -1 padding for train
        # energy = torch.ones_like(energy_)*-1
        energy = torch.zeros_like(energy_)
        energy = energy.to(device)
        energy[:,1:] = energy_[:,:-1]
        
        ## embedding layer
        pos_id = torch.arange(self.time_steps).unsqueeze(0).to(device)
        pos_id = self.pos_embedding(pos_id)
        
        if debug: print(num.shape)

        num = self.num_emb(num.to(device).long())
        cool = self.cool_emb(cool.to(device).long())
        sun_e = self.sun_e_emb(sun_e.to(device).long())
        if debug: print(doy.shape)

        doy = self.doy_emb(doy.to(device).long())
        dow = self.dow_emb(dow.to(device).long())
        wd = self.wd_emb(wd.to(device).long())
        hr = self.hr_emb(hr.to(device).long())

        if debug: print(temp.shape)
        if debug: print(temp.type())

        energy = self.energy_emb(energy.to(device).long())
        temp = self.t_embs(temp.to(device).long())
        wind = self.w_embs(wind.to(device).long())
        humid = self.h_embs(humid.to(device).long())
        rain = self.r_embs(rain.to(device).long())
        sun = self.s_embs(sun.to(device).long())

        # print(num.shape, temp.shape, sun_e.shape)
        ## input
        enc = num + cool + sun_e + doy + dow + wd + hr
        dec = energy + temp + wind + humid + rain + sun

        if debug: print(enc.shape)

        enc = enc.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)

        ## masking
        att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask, memory_mask = mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)


# -

# TODO : change for multi nums
class Dacon_Dataset(Dataset):
    def __init__(self, dataset, time_steps = 24, test = False):
        self.dataset = dataset # LIST
        self.time_steps = time_steps
        self.test = test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        num, temp, wind, humid, rain, sun, cool, sun_e, doy, dow, wd, hr, energy = self.dataset[index]

        # seq_len = len(num_)
        num = torch.from_numpy(num).type(torch.IntTensor)

        temp = torch.from_numpy(temp).type(torch.IntTensor)
        wind = torch.from_numpy(wind).type(torch.IntTensor)
        humid = torch.from_numpy(humid).type(torch.IntTensor)
        rain = torch.from_numpy(rain).type(torch.IntTensor)
        sun = torch.from_numpy(sun).type(torch.IntTensor)

        cool = torch.from_numpy(cool).type(torch.IntTensor)
        sun_e = torch.from_numpy(sun_e).type(torch.IntTensor)

        doy = torch.from_numpy(doy).type(torch.IntTensor)
        dow = torch.from_numpy(dow).type(torch.IntTensor)
        wd = torch.from_numpy(wd).type(torch.IntTensor)
        hr = torch.from_numpy(hr).type(torch.IntTensor)

        energy = torch.from_numpy(energy).type(torch.IntTensor)

        return num, temp, wind, humid, rain, sun, cool, sun_e, doy, dow, wd, hr, energy


def preprocessing_(SEQ_LEN):
    train_df = pd.read_csv('./trainp.csv')
    test_df = pd.read_csv('./testp.csv')
    num_cols = ['t','w','h','r','s']
    for col in num_cols:
        train_df[col] = train_df[col].apply(int)
        test_df[col] = test_df[col].apply(int)
    train_df.y = train_df.y.apply(int) + 1

    Y_MAX = train_df.y.max()
    NUM_MAXS = [max(train_df[c].max(),test_df[c].max()) for c in num_cols] 
    
    train_group = train_df.groupby('num')
    
    cols = ['num','t','w','h','r','s','cooling','sun_e','day_of_year','day_of_week','weekday','hour']
    
    num_to_df={}
    for i,df in train_group:
        num_to_df[i]=df[cols+['y']]
    
    #학습의 단위가 되는 길이
    AGU_SET = 5 # 특정 개수를 다른 데이터로 치환 하여 agumentation
    
    seqs = []
    for df in num_to_df.values():
        seqs += make_seq(df, seq_len=SEQ_LEN)
    
    return Y_MAX, NUM_MAXS, seqs


def smape(true, pred):

    v = 2 * abs(np.array(pred) - np.array(true)) / (abs(np.array(pred)) + abs(np.array(true)))

    output = np.mean(v) * 100

    return output


def main():
    args = parser.parse_args()
    
    if args.request_from_nni:
        import nni
        tuner_params = nni.get_next_parameter()
        logger.info(str(tuner_params))

        if "batch_size" in tuner_params:
            args.batch_size = int(tuner_params["batch_size"])
            
        if "lr" in tuner_params:
            args.lr = tuner_params["lr"]
            
        if "time_steps" in tuner_params:
            args.time_steps = tuner_params["time_steps"]
            
        if "num_head" in tuner_params:
            args.num_head = tuner_params["num_head"]
            
        if "embed_dim" in tuner_params:
            args.embed_dim = tuner_params["embed_dim"]
            
        if "num_encoderlayers" in tuner_params:
            args.num_encoderlayers = tuner_params["num_encoderlayers"]
            
        if "num_decoderlayers" in tuner_params:
            args.num_decoderlayers = tuner_params["num_decoderlayers"]
            
        if "dropout" in tuner_params:
            args.dropout = tuner_params["dropout"]

        logger.info(str(args))
        
        # demonstrate that intermediate result is actually sent
        nni.report_intermediate_result(200.)

        args.model_dir = './models'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)


    if args.gpu is not None:
        args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.Y_MAX, args.NUM_MAXS, args.seqs = preprocessing_(args.time_steps)
       
    train, val = train_test_split(args.seqs, test_size=0.2, shuffle=True)
    train_set = Dacon_Dataset(train, args.time_steps, False)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)

    val_set = Dacon_Dataset(val, args.time_steps, False)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=True)

    args.train_loader = train_loader
    args.val_loader = val_loader

    
    # Simply call main_worker function
    main_worker(args.gpu, args)


# +
def main_worker(gpu, args):
    best_smape = 200
    
    device = args.gpu

    model = Energy_Model(args.Y_MAX,
                         args.NUM_MAXS,
                         time_steps=args.time_steps,
                         n_head=args.num_head,
                         embed_dim=args.embed_dim,
                         num_encoding_layers=args.num_encoderlayers,
                         num_decoding_layers=args.num_decoderlayers,
                         dropout=args.dropout
                            )

    
    criterion = nn.MSELoss()
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)


    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    
    else:
        raise NotImplementedError("Your requested optimizer '%s' is not found" % args.optimizer)

    
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    
#     args.criterion = criterion
#     args.optimizer = optimizer
    
    writer = SummaryWriter(os.path.join(args.model_dir, "summary"))
    
    for epoch in range(args.epochs):
        if not args.dont_adjust_learning_rate:
            adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        if not train(model, args.train_loader, optimizer, criterion, epoch, args, device):
            break
  
        # evaluate on validation set
        smape = validate(model, args.val_loader, criterion, args, device)

        if args.request_from_nni:
            import nni
            nni.report_intermediate_result(smape)

        # remember best smape@1 and save checkpoint
        is_best = smape < best_smape
        best_smape_ = min(smape, best_smape)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'best_smape': best_smape_,
            'optimizer': args.optimizer,
            'model': model,
        }, is_best, best_smape_, filename=os.path.join(args.model_dir, "checkpoint.pth.tar"))

    writer.close()

    try:
        if args.request_from_nni:
            import nni
            nni.report_final_result(best_smape_)
            logger.info("Reported intermediate results to nni successfully")
    except NameError:
        logger.info("No accuracy reported")
        pass


# -

def train(model, train_dataloader, optimizer, criterion, epoch, args, device, print_freq=30):
    logger.info("Epoch %d starts" % epoch)
    
    forward_time = AverageMeter('Forward', ':6.3f')
    criterion_time = AverageMeter('Criterion', ':6.3f')
    backward_time = AverageMeter('Backward', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    train_loss = []

    start_time = time.time()
    
    model.train()
    
    ## training
    for item in train_dataloader:
        for i, _ in enumerate(item):
            item[i] = item[i].to(device)

        energy = item[-1].type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        output = model(*item)

        loss = criterion(output, energy)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
        
        elapsed_time = time.time() - start_time
        
    train_loss = np.mean(train_loss)
    
    losses.update(train_loss)
    

    return True


def validate(model, val_dataloader, criterion, args, device, print_freq=30):
    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Batch', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    smape_ = AverageMeter('smape_', ':6.2f')

    # switch to evaluate mode
    model.eval()
    logger.info("Evaluation starts")

    start_time = time.time()
    
    energies = []
    outs = []
    val_loss = []

    # validation
    model.eval()
    for item in val_dataloader:

        for i, _ in enumerate(item):
            item[i] = item[i].to(device)

        energy = item[-1].to(device)

        output = model(*item)

        loss = criterion(output, energy)
        
        energies.extend(energy.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())
        
        val_loss.append(loss.item())
    
    smape_item = smape(energies,outs)
    
    val_loss = np.mean(val_loss)
    
    elapsed_time = time.time() - start_time 

    losses.update(val_loss)
    smape_.update(smape_item)


    return smape_.val

if __name__ == '__main__':
    logger.info("Process launched")
    main()
    logger.info("Process succesfully terminated")
