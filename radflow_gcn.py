'''
@author: lpc
@time:  2021/10/13 20:49
@file:  radflow_gcn.py
'''
import numpy as np
import torch
import pandas as pd
import torch
import time
import argparse
from util.preparedata import *
from torch.utils.data import DataLoader
from model.radflow_with_gcn import Radflow
from util.metric import *

parser = argparse.ArgumentParser(description='some program parameters')
parser.add_argument('--device', type=str, default='cuda:0', help='which gpu to use')
parser.add_argument('--data_file_path', type=str, default='./data/PEMS04/pems04.npz', help='data file path')
parser.add_argument('--adj_file_path', type=str, default='./data/PEMS04/distance.csv', help='node distances file')
parser.add_argument('--recurrent_blocks', type=int, default=8, help='num of blocks of recurrent_blocks')
parser.add_argument('--lstm_hidden_size', type=int, default=64, help='')
parser.add_argument('--H',type=int,default=64,help='')
parser.add_argument('--K',type=int,default=3,help='num of hops of gcn')
parser.add_argument('--D',type=int,default=1,help='feature dims')
parser.add_argument('--n_heads',type=int,default=4,help='num of multi heads')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--num_of_nodes', type=int, default=307, help='')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--pre_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--epochs', type=int, default=20, help='')
parser.add_argument('--train_ratio', type=float, default=0.7, help='')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='')
parser.add_argument('--save', type=str, default='./save/PEMS04', help='')
# parser.add_argument('',type=,default='',help='')

args = parser.parse_args()
print(args)
device = torch.device(args.device)
torch.manual_seed(0)

model = Radflow(args)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# metric = torch.nn.MSELoss()
metric = masked_rmse

#看请梯度后，加一个梯度裁剪

#准备需要用到的数据

def main(args):
    device = torch.device(args.device)
    adj, normalization_adj, dis_adj = get_adj_mm(args.adj_file_path,args.num_of_nodes)
    node_neigh = construct_node_neighbor(adj)
    train_dataset, test_dataset, val_dataset, stats = get_dataset(args.data_file_path, args.seq_len, args.pre_len, args.train_ratio)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    mean = stats['mean'][0,0,0,2]
    std = stats['std'][0,0,0,2]

    train_loss = []
    test_loss = []
    print(model)

    for epoch in range(args.epochs):
        model.train()

        #训练模型
        for iter, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.unsqueeze(-1).to(device), torch.FloatTensor(normalization_adj).to(device))
            out = out * std + mean
            #还原到正常区间

            # loss = metric(out, y.to(device))
            loss = metric(out[:,:,:], y.to(device)[:,:,:], 0.0)

            loss.backward()
            optimizer.step()
            train_loss.append(float(loss))
            if iter % 50  == 0:
                print('Epoch %d. iter %d. batch mse is %f' % (epoch, iter, float(loss.data)))
        print('---------------Epoch %d. total train mse is %f----------------' % (epoch, sum(train_loss)/ len(train_loss)))

        #测试集测试模型
        model.eval()
        with torch.no_grad():
            for iter,(x, y) in enumerate(test_loader):
                pre = model(x.unsqueeze(-1).to(device), torch.FloatTensor(normalization_adj).to(device))
                pre = pre * std + mean
                # l = metric(pre, y.to(device))
                l = metric(pre, y.to(device), 0.0)
                # l.backward()
                # optimizer.step()
                test_loss.append(float(l))
        print('---------------Epoch %d. total test mse is %f----------------' % (epoch, sum(test_loss)/ len(test_loss)))
        scheduler.step()


if __name__ == '__main__':
    start = time.time()
    main(args)
    end = time.time()
    print("Total time spent: {:.4f}".format(end-start)) #单位是s
