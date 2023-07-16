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

import jax.numpy as jnp
import jraph
import numpy as np
from scipy.sparse import csr_matrix
from load_dataset import load_dataset
from torchvision.datasets import CIFAR10
import torch
from config import *
from torch.utils.data import DataLoader
from sampler import SubsetSequentialSampler
import resnet as resnet
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix


# 将labels转为one-hot编码，接收一个标签列表
def onehot(labels):
    unique_labels = np.unique(labels)
    return jnp.identity(len(unique_labels))[jnp.array(labels)]


# 从npz加载数据集，接收path和数据集名称，返回邻接矩阵adj_matrix、特征矩阵attr_matrix和labels
def load_from_npz(path, dataset):
    """Loads datasets from npz files."""
    file_name = "C:/Users/Zhoumin/Desktop/thesis/rs_gnn/data/cora.npz"
    with np.load(open(file_name, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = csr_matrix(
                (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                shape=loader['attr_shape']).todense()
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            raise Exception('No attributes in the data file', file_name)

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = csr_matrix((loader['labels_data'], loader['labels_indices'],
                                 loader['labels_indptr']),
                                shape=loader['labels_shape'])
            labels = labels.nonzero()[1]
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            raise Exception('No labels in the data file', file_name)

    return adj_matrix, attr_matrix, onehot(labels)


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
    rows = adj.tocoo().row
    cols = adj.tocoo().col
    edges = {(row, col) for row, col in zip(rows, cols)}
    edges = symmetrize(edges)
    edges = add_self_loop(edges, features.shape[0])
    return edges, len(edges)


def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features  # .detach().cpu().numpy()
    return feat


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

    return adj.tocsr()


# 创建jraph，接收path和数据集名称，返回jraph中的图表示、labels和类别数
def create_jraph():
    """Creates a jraph graph for a dataset."""
    data_train, _, _, _, NO_CLASSES, no_train = load_dataset('cifar10')
    original_indices = list(range(no_train))
    data_train_loader = DataLoader(data_train, batch_size=BATCH,
                                   sampler=SubsetSequentialSampler(original_indices),
                                   pin_memory=True)
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).cuda()
    model = {'backbone': resnet18}
    torch.backends.cudnn.benchmark = True

    features = get_features(model, data_train_loader).cpu().numpy()
    adj = knn_similarity_graph(features, 15)
    labels = onehot(data_train.targets)

    # adj, features, labels = load_from_npz(data_path, dataset)
    edges, n_edge = get_graph_edges(adj, np.array(features))
    n_node = len(features)
    features = jnp.asarray(features)
    graph = jraph.GraphsTuple(
        n_node=jnp.asarray([n_node]),
        n_edge=jnp.asarray([n_edge]),
        nodes=features,
        edges=None,
        globals=None,
        senders=jnp.asarray([edge[0] for edge in edges]),
        receivers=jnp.asarray([edge[1] for edge in edges]))

    return graph, np.asarray(labels), labels.shape[1]
