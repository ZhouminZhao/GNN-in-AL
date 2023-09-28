# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data utils for RS-GNN."""
import random

import numpy as np
from load_dataset import load_dataset
import torch
import torch.nn as nn
from config import *
from torch.utils.data import DataLoader
from sampler import SubsetSequentialSampler
import resnet as resnet
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix


# 将labels转为one-hot编码，接收一个标签列表
def onehot(labels):
    unique_labels = np.unique(labels)
    return np.identity(len(unique_labels))[np.array(labels)]


# 对称化邻接矩阵
def symmetrize(edges):
    """Symmetrizes the adjacency."""
    inv_edges = {(d, s) for s, d in edges}
    return edges.union(inv_edges)


# 在图中添加自环，返回边集合
def add_self_loop(edges, n_node):
    """Adds self loop."""
    self_loop_edges = {(s, s) for s in range(n_node)}
    return edges.union(self_loop_edges)


# 从邻接矩阵和特征矩阵获取边集合和边数
def get_graph_edges(adj, features):
    coo = adj.coalesce().indices()  # TODO: for me this .to_sparse() cause an error
    rows = coo[0]
    cols = coo[1]
    edges = {(row.item(), col.item()) for row, col in zip(rows, cols)}
    edges = symmetrize(edges)
    edges = add_self_loop(edges, features.shape[0])
    return edges, len(edges)


def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features#.detach().cpu().numpy()
    return feat


def get_features_train(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features#.detach().cpu().numpy()
    return feat


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj += -1.0 * np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0)  # rowise sum
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj

def knn_similarity_graph(data, k):
    n = data.shape[0]
    adj = lil_matrix((n, n))
    # Create Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=k + 1)
    nn_model.fit(data)
    # Find k nearest neighbors for each data point
    distances, indices = nn_model.kneighbors(data)
    # Create adjacency matrix
    for i in range(n):
        adj[i, indices[i, 1:]] = 1.0
    # Symmetrize the adjacency matrix
    adj = adj.maximum(adj.transpose())
    adj = adj.tocsr()
    adj_data = np.array(adj.data)
    adj_nonzero = np.array(adj.nonzero())
    adj = torch.sparse_coo_tensor(torch.tensor(adj_nonzero), torch.tensor(adj_data), adj.shape)

    return adj


# 创建jraph，返回图表示、labels和类别数
def create_jraph(models, subset, labeled_set, select_round):
    """Creates a jraph graph for a dataset."""
    data_train, data_unlabeled, _, _, NO_CLASSES, no_train = load_dataset('cifar10')

    if select_round == 'first':
        original_indices = list(range(no_train))
        data_loader = DataLoader(data_train, batch_size=BATCH,
                                 sampler=SubsetSequentialSampler(original_indices),
                                 pin_memory=True)
        features = get_features_train(models, data_loader)
    elif select_round == 'sequential':
        data_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                 sampler=SubsetSequentialSampler(subset + labeled_set),
                                 pin_memory=True)
        features = get_features(models, data_loader)

    features = nn.functional.normalize(features)
    features = features.detach().cpu().numpy()
    labels = onehot(data_train.targets)
    #adj = aff_to_adj(features)
    adj = knn_similarity_graph(features, 15)

    edges, n_edge = get_graph_edges(adj, features)
    n_node = len(features)

    features = torch.from_numpy(features)

    graph = {
        'n_node': np.array([n_node]),
        'n_edge': np.array([n_edge]),
        'nodes': features,
        'adj': adj,
        'edges': None,
        'globals': None,
        'senders': np.array([edge[0] for edge in edges]),
        'receivers': np.array([edge[1] for edge in edges])
    }

    return graph, np.asarray(labels), labels.shape[1]
