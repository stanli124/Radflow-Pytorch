'''
@author: lpc
@time:  2021/10/13 20:48
@file:  radflow_with_gcn.py
'''
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from util.preparedata import *
import logging
import math
logger = logging.getLogger(__name__)



class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.hidden_size = args.lstm_hidden_size
        self.H = args.H
        self.feedforw_1 = nn.Linear(self.hidden_size, self.H)
        self.feedforw_2 = nn.Linear(self.H, self.H)
        self.gelu = nn.GELU()

    def forward(self, h):
        h = self.gelu(self.feedforw_1(h))
        h = self.feedforw_2(h)
        return h

class Recurrent_Block(nn.Module):
    def __init__(self, args):
        super(Recurrent_Block, self).__init__()
        self.hidden_size = args.lstm_hidden_size
        self.H = args.H
        self.dp = args.dropout

        self.dropout = nn.Dropout(self.dp)
        self.lstm = nn.LSTM(self.H, self.hidden_size,batch_first=True)
        self.FC_p = FeedForward(args)
        self.FC_q = FeedForward(args)
        self.FC_u = FeedForward(args)

    def forward(self, x):
        x = self.dropout(x) #B N 64

        h, _ = self.lstm(x)   #B N hidden_size
        p = self.FC_p(h)
        q = self.FC_q(h)
        u = self.FC_u(h)

        return p, q, u


class Recurrent(nn.Module):
    def __init__(self, args):
        super(Recurrent, self).__init__()
        H = args.H
        D = args.D
        self.blocks_num = args.recurrent_blocks
        self.blocks = nn.ModuleList()
        self.steps = args.seq_len
        self.WR = nn.Linear(H, D)
        self.WF = nn.Linear(H, D)
        for l in range(self.blocks_num):
            self.blocks.append(Recurrent_Block(args))

    def forward(self, x): #B N H
        input = x

        q = torch.zeros((x.shape[0], x.shape[1], x.shape[2])).to(torch.device('cuda:0'))#B N H
        u = torch.zeros((x.shape[0], x.shape[1], x.shape[2])).to(torch.device('cuda:0'))

        for l in range(self.blocks_num):
            p_, q_, u_ = self.blocks[l](input)
            input = input - p_
            q = q + q_
            u = u + u_ #节点嵌入表示

        q = self.WR(q)  #B N 1
        q = q.squeeze(-1)
        u = u/self.blocks_num

        return q, u




class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        H = args.H
        D = args.D
        self.K = args.K
        self.W = nn.ModuleList()
        for k in range(self.K):
            self.W.append(nn.Linear(H, H))

        self.WA = nn.Linear(H, D)
        self.relu = nn.ReLU()

    def forward(self, X, adj): #N N  B N H
        output = torch.zeros(*X.shape).to(torch.device('cuda:0')) #B N H
        for k in range(self.K):
            X = torch.einsum('bnh,nn->bnh', X, adj)
            X = self.W[k](X)
            output += X



        output = self.relu(self.WA(output))
        return output.squeeze(-1)


class Radflow(nn.Module):
    def __init__(self, args):
        super(Radflow, self).__init__()
        D = args.D
        H = args.H
        self.steps = args.seq_len

        self.recur = Recurrent(args)
        self.gcn = GCN(args)
        self.WD = nn.Linear(D, H)

    def construct_neigh_embedding(self, node_t, node_neigh):
        node_next = []
        for i in range(307):
            indices = list(node_neigh[i])
            if len(indices)==0: #若有邻域，则聚合邻域的值
                temp = node_t[:, i, :]
            else:    #若没有邻域就等于自己
                indices = torch.tensor(indices).to(torch.device('cuda:0'))
                temp = torch.index_select(node_t, 1, torch.tensor(indices))
                temp = torch.sum(temp, dim=1)  # 对邻域节点进行聚合
            node_next.append(temp)

        return torch.stack(node_next, dim=1)

    def forward(self, x, adj):  #B T N 1
        input = self.WD(x)  
        forecast = []
        node_embed = []

        for step in range(self.steps):
            #recurrent
            R_t, node_t = self.recur(input[:, step, :, :]) #node_t  B N H
            A_t = self.gcn(input[:, step, :, :], adj)

            f = R_t + A_t

            forecast.append(f.unsqueeze(1))

        f = torch.cat(forecast, dim=1)

        return f