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

"""GNN Models in jraph/flax."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import layers


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.to(torch.float32)
        support = torch.mm(input, self.weight)
        adj = adj.to(torch.float32)
        support = support.to(torch.float32)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# GCN类，接收图数据对象graph的nodes，返回最终的节点表示
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, drop_rate, activation):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.activation_fn = layers.Activation(activation)
        self.dropout = drop_rate
        self.nfeat = nfeat
        self.linear = nn.Linear(nclass, 1)
        self.fc_layer = nn.Linear(nfeat, nhid)

    def forward(self, x, adj, train):
        '''
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=train)
        x = self.gc3(feat, adj)
        x = self.linear(x)
        '''

        x = self.fc_layer(x)
        x = self.activation_fn(self.gc2(x, adj))
        feat = F.dropout(x, self.dropout, training=train)
        x = F.relu(self.gc3(feat, adj))
        return feat, torch.sigmoid(x)


# DGI类，接收两个图数据对象graph和c_graph，通过bilinear产生表示的摘要和预测的logits
class DGI(nn.Module):

    def __init__(self, nfeat, nhid):
        super(DGI, self).__init__()
        self.gcn = GCN(nfeat, nhid, 10, 0.5, 'SeLU')

    def forward(self, graph, c_graph):
        nodes1, _ = self.gcn(graph['nodes'], graph['adj'], train=True)
        nodes2, _ = self.gcn(c_graph['nodes'], c_graph['adj'], train=True)
        summary = layers.dgi_readout(nodes1)
        nodes = torch.cat([nodes1, nodes2], dim=0)
        bilinear = layers.Bilinear(nodes.shape[-1], summary.shape[-1])
        logits = bilinear(nodes, summary)
        return (nodes1, nodes2, summary), logits


# RSGNN类，接收两个图数据对象graph和c_graph，生成节点表示，并进行归一化，计算聚类中心、表示的标识符和聚类损失
class RSGNN(nn.Module):
    """The RSGNN model."""

    def __init__(self, nfeat, hid_dim, num_reps, centers):
        super(RSGNN, self).__init__()
        self.num_reps = num_reps
        self.dgi = DGI(nfeat, hid_dim)
        self.cluster = Cluster(num_reps, centers)

    def forward(self, graph, c_graph):
        (h, _, _), logits = self.dgi(graph, c_graph)
        h = layers.normalize(h)
        centers, rep_ids, cluster_loss = self.cluster(h[:10000])
        return h, centers, rep_ids, cluster_loss, logits


# Cluster类，接收节点表示embs，计算聚类中心、表示的标识符和聚类损失
class Cluster(nn.Module):
    """Finds cluster centers given embeddings."""

    def __init__(self, num_reps, centers):
        super(Cluster, self).__init__()
        self.num_reps = num_reps
        self.cluster = layers.EucCluster(num_reps=num_reps, centers=centers)

    def forward(self, embs):
        rep_ids, cluster_dists, centers = self.cluster(embs)
        loss = torch.sum(cluster_dists)
        return centers, rep_ids, loss
