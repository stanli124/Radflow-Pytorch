'''
AUTHOR :li peng cheng

DATE :2021/10/10 21:23
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


class Flow(nn.Module):
    def __init__(self,args):
        super(Flow, self).__init__()
        H = args.H
        D = args.D
        self.n_heads = args.n_heads
        self.dmodel = H

        self.WE = nn.Linear(H, H)
        self.WN = nn.Linear(H, H)
        self.WA = nn.Linear(H, D)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax()

        self.WQ = nn.Linear(self.dmodel, self.dmodel)#第二个维度也是dmodel是为了并行计算n个头，
        self.WK = nn.Linear(self.dmodel, self.dmodel)
        self.WV = nn.Linear(self.dmodel, self.dmodel)
        self.WO = nn.Linear(self.dmodel, self.dmodel)

    def forward(self, node_t, node_next): #b n h
        #B N dmodel
        Q = self.WQ(node_t)
        K = self.WK(node_next)
        V = self.WV(node_next)

        K_T = K.transpose(1,2)  #B dmodel N

        att = torch.einsum('bid,bdk->bik', Q, K_T)
        att = self.softmax(att / (self.dmodel / self.n_heads))
        att = self.gelu(self.WO(torch.einsum('bnn,bnd->bnd', att, V)))   #bnd

        A_t = self.WE(node_t) + self.WN(att)
        A_t = self.WA(A_t)
        return A_t.squeeze(-1)

class Radflow(nn.Module):
    def __init__(self, args):
        super(Radflow, self).__init__()
        D = args.D
        H = args.H
        self.steps = args.seq_len

        self.recur = Recurrent(args)
        self.flow = Flow(args)
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
                temp = torch.sum(temp, dim=1)  
            node_next.append(temp)

        return torch.stack(node_next, dim=1)

    def forward(self, x, node_neigh):  #B T N 1
        input = self.WD(x) 
        forecast = []
        node_embed = []

        for step in range(self.steps):
            #recurrent
            R_t, node_t = self.recur(input[:, step, :, :]) #  B N H

            #flow 
            node_next = self.construct_neigh_embedding(node_t, node_neigh) # B N H 
            A_t = self.flow(node_t, node_next)

            f = R_t + A_t

            forecast.append(f.unsqueeze(1))

        f = torch.cat(forecast, dim=1)

        return f
