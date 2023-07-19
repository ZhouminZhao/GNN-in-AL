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

"""Trainer for various models."""

import numpy as np
import torch
import torch.optim as optim

import rsgnn_models


# 跟踪训练中的最佳result和params
class BestKeeper:
    """Keeps best performance and model params during training."""

    def __init__(self, min_or_max):
        self.min_or_max = min_or_max
        self.best_result = np.inf if min_or_max == 'min' else 0.0
        self.best_params = None

    # 打印当前epoch的result
    def print_(self, epoch, result):
        if self.min_or_max == 'min':
            print('Epoch:', epoch, 'Loss:', result)
        elif self.min_or_max == 'max':
            print('Epoch:', epoch, 'Accu:', result)

    # 更新最佳result和params
    def update(self, epoch, result, params, print_=True):
        """Updates the best performance and model params if necessary."""
        if print_:
            self.print_(epoch, result)

        if self.min_or_max == 'min' and result < self.best_result:
            self.best_result = result
            self.best_params = params
        elif self.min_or_max == 'max' and result > self.best_result:
            self.best_result = result
            self.best_params = params

    # 返回最佳params
    def get(self):
        return self.best_params


# 创建模型：flags包含模型的超参配置的命名空间对象，inp输入数据，rng随机数生成器的状态
# 返回：模型、初始化参数、更新后随机数生成器的状态
def create_model(flags, model_name, rng):
    new_rng, init_rng, drop_rng = rng.integers(low=0, high=10, size=3)
    if model_name == 'gcn':
        features = [flags.hid_dim, flags.num_classes]
        model = rsgnn_models.GCN(features, flags.drop_rate, 'PReLU')
    elif model_name == 'rsgnn':
        model = rsgnn_models.RSGNN(flags.hid_dim, flags.num_reps)
    return model, new_rng

# 训练rsgnn：flags包含模型的超参配置的命名空间对象，graph，随机数生成器的状态
# 返回一个numpy数组，包含得到的representation的标识符IDs
def train_rsgnn(flags, graph, rng):
    """Trainer function for RS-GNN."""
    features = graph['nodes']
    n_nodes = graph['n_node'][0]
    labels = torch.cat([torch.ones(n_nodes), -torch.ones(n_nodes)], dim=0)
    model = rsgnn_models.RSGNN(nfeat=features.shape[1], hid_dim=flags.hid_dim, num_reps=flags.num_reps)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr, weight_decay=0.0)

    def corrupt_graph(corrupt_rng):
        permuted_nodes = torch.randperm(graph['nodes'].shape[0], generator=torch.Generator().manual_seed(int(corrupt_rng)))
        corrupted_nodes = graph['nodes'][permuted_nodes]
        corrupted_graph = {
            'n_node': graph['n_node'],
            'n_edge': graph['n_edge'],
            'nodes': corrupted_nodes,
            'adj': graph['adj'],
            'edges': graph['edges'],
            'globals': graph['globals'],
            'senders': graph['senders'],
            'receivers': graph['receivers']
        }
        return corrupted_graph

    def train_step(optimizer, graph, c_graph, drop_rng):
        def loss_fn():
            _, _, _, cluster_loss, logits = model(graph, c_graph)  # 将转换后的张量作为输入传递给模型
            dgi_loss = -torch.sum(torch.nn.functional.logsigmoid(labels * logits))
            return dgi_loss + flags.lambda_ * cluster_loss

        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        return optimizer, loss.item()

    best_keeper = BestKeeper('min')
    for epoch in range(1, flags.epochs + 1):
        rng, drop_rng, corrupt_rng = np.random.randint(low=0, high=10, size=3)
        c_graph = corrupt_graph(corrupt_rng)
        optimizer, loss = train_step(optimizer, graph, c_graph, drop_rng)
        if epoch % flags.valid_each == 0:
            best_keeper.update(epoch, loss, model.state_dict())

    model.load_state_dict(best_keeper.get())
    _, _, rep_ids, _, _ = model(graph, c_graph)

    return rep_ids.numpy()
