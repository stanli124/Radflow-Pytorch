'''
@author: lpc
@time:  2021/10/12 18:54
@file:  radflow_nonetwork.py
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

    def forward(self, x): #B T N H
        forecast = [] #保存所有预测步的数据 每个的shape B N
        node_embed = []

        for i in range(self.steps):
            q = torch.zeros((x.shape[0], x.shape[2], x.shape[3])).to(torch.device('cuda:0'))#B N H
            u = torch.zeros((x.shape[0], x.shape[2], x.shape[3])).to(torch.device('cuda:0'))

            input = x[:,i,:,:] #B N H
            for l in range(self.blocks_num):
                p_, q_, u_ = self.blocks[l](input)
                input = input - p_
                q = q + q_
                u = u + u_ #节点嵌入表示

            q = self.WR(q)  #B N 1
            forecast.append(q.unsqueeze(1).squeeze(-1))
            u = u/self.blocks_num
            node_embed.append(u)

        return forecast, node_embed


class Flow(nn.Module):
    def __init__(self,args):
        super(Flow, self).__init__()

    def forward(self,x):

        return x

class Radflow(nn.Module):
    def __init__(self, args):
        super(Radflow, self).__init__()
        D = args.D
        H = args.H

        self.recur = Recurrent(args)
        self.flow = Flow(args)
        self.WD = nn.Linear(D, H)

    def forward(self, x):
        x = self.WD(x) 
        recurrent_pre, node_embed = self.recur(x)
        recurrent_pre = torch.cat(recurrent_pre, dim=1)

        return recurrent_pre
