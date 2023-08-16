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

import torch
import torch.nn as nn


# 激活函数层
class Activation(nn.Module):
    """Activation function."""

    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activation

        if activation == 'ReLU':
            self.act_fn = nn.ReLU()
        elif activation == 'SeLU':
            self.act_fn = nn.SELU()
        else:
            raise Exception('Activation not recognized')

    def forward(self, x):
        return self.act_fn(x)


# 双线性层
class Bilinear(nn.Module):
    """A Bilinear Layer."""

    def __init__(self, in_features_l, in_features_r):
        super(Bilinear, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(in_features_l, in_features_r))
        nn.init.normal_(self.kernel)

    def forward(self, x_l, x_r):
        return torch.matmul(torch.matmul(x_l, self.kernel), x_r)


# KMeans聚类层，接收输入的x，返回聚类结果、最小距离和聚类中心
class EucCluster(nn.Module):
    """Learnable KMeans Clustering."""

    def __init__(self, num_reps, init_fn, init):
        super(EucCluster, self).__init__()
        self.num_reps = num_reps
        self.init_fn = init_fn

        if init is None:
            self.centers_known = []
        else:
            self.centers_known = init

    def forward(self, x):
        centers_new = nn.Parameter(self.init_fn(torch.empty(self.num_reps, x.shape[-1])))
        if self.centers_known == []:
            centers = centers_new
        else:
            centers = torch.cat([self.centers_known, centers_new], dim=0)
        print(centers.shape)
        dists = torch.cdist(x, centers_new, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        print(dists.shape)
        return find_unique_min_indices(dists), torch.min(dists, dim=1)[0], centers


def find_unique_min_indices(dists):
    n, m = dists.shape
    unique_min_indices = torch.zeros(m, dtype=torch.long)
    found_indices = set()

    for i in range(m):
        col = dists[:, i]
        min_val = float('inf')
        min_idx = -1
        for j in range(n):
            if j not in found_indices and col[j] < min_val:
                min_val = col[j]
                min_idx = j
        if min_idx != -1:
            unique_min_indices[i] = min_idx
            found_indices.add(min_idx)

    return unique_min_indices


# DGI (Deep Graph Infomax)的读出函数，接收节点表示node_embs，应用sigmoid函数
def dgi_readout(node_embs):
    return torch.sigmoid(torch.mean(node_embs, dim=0))


# 减去均值的函数
def subtract_mean(embs):
    return embs - torch.mean(embs, dim=0)


# 除以L2范数的函数
def divide_by_l2_norm(embs):
    norm = torch.norm(embs, dim=1, keepdim=True)
    return embs / norm


# 归一化节点表示，先减去均值，再除以L2范数
def normalize(node_embs):
    return divide_by_l2_norm(subtract_mean(node_embs))
