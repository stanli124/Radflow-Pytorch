'''
AUTHOR :li peng cheng

DATE :2021/10/10 15:37
'''
import random
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def construct_node_neighbor(adj):
    num_of_node = adj.shape[0]
    node_neigh = []
    for i in range(num_of_node):
        neighs = adj[i].nonzero()[0]
        node_neigh.append(neighs)
    return node_neigh

def symmetric_normalization_adj(adj_mm, num_of_node):
    A = adj_mm
    A = A + np.identity(num_of_node)
    D = np.diag(A.sum(1)**-0.5)
    return np.dot(np.dot(D, A), D)

def get_adj_mm(adj_file, num_of_node):
    distance = pd.read_csv(adj_file)
    adj = np.zeros((num_of_node, num_of_node))
    distance_matrix = np.zeros((num_of_node, num_of_node))
    for i in range(len(distance)):
        row = distance.iloc[i]
        from_, to_, dis = int(row[0]), int(row[1]), int(row[2])
        adj[from_, to_] = 1
        distance_matrix[from_, to_] = dis
    return adj, symmetric_normalization_adj(adj, num_of_node), distance_matrix

def normalization(x_train, x_test, x_val):
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)

    x_train=(x_train - mean) / std
    x_test=(x_test - mean) / std
    x_val=(x_val - mean) / std

    return x_train,x_test,x_val, {'mean':mean, 'std':std}

def get_dataset(history_data_file, seq_len, pre_len, ratio):
    data = np.load(history_data_file)
    data = data['data']
    lenth = len(data)
    x = []
    y = []
    for i in range(lenth - seq_len - pre_len):
        start_x = i
        start_y = start_x + seq_len
        x.append(data[start_x:start_x+seq_len][np.newaxis, :])
        y.append(data[start_y:start_y+pre_len][np.newaxis, :])
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    lenth=len(x)
    train_split = int(ratio*lenth)
    test_split = int(0.2*lenth+train_split)
    val_split = int(lenth-test_split)

    x_train = x[:train_split]
    y_train = y[:train_split]
    x_test = x[train_split:test_split]
    y_test = y[train_split:test_split]
    x_val = x[-val_split:]
    y_val = y[-val_split:]

    x_train, x_test, x_val, stats = normalization(x_train,x_test,x_val)

    train_dataset = TensorDataset(torch.FloatTensor(x_train[:,:,:,2]), torch.FloatTensor(y_train[:,:,:,2]))
    test_dataset = TensorDataset(torch.FloatTensor(x_test[:,:,:,2]), torch.FloatTensor(y_test[:,:,:,2]))
    val_dataset = TensorDataset(torch.FloatTensor(x_val[:,:,:,2]), torch.FloatTensor(y_val[:,:,:,2]))

    print('train:', x_train[:,:,:,2].shape, y_train[:,:,:,2].shape)
    print('test:', x_test[:,:,:,0].shape, y_test[:,:,:,2].shape)
    print('val:', x_val[:,:,:,2].shape, y_val[:,:,:,2].shape)

    return train_dataset, test_dataset, val_dataset, stats

