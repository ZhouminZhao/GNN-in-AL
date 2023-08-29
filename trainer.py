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


# 训练rsgnn：flags包含模型的超参配置的命名空间对象，graph，随机数生成器的状态
# 返回一个numpy数组，包含得到的representation的标识符IDs
def train_rsgnn(flags, graph, rng, init):
    """Trainer function for RS-GNN."""
    features = graph['nodes']
    n_nodes = graph['n_node'][0]
    labels = torch.cat([torch.ones(n_nodes), -torch.ones(n_nodes)], dim=0)
    #new_seed = rng.integers(0, np.iinfo(np.int32).max)
    #new_rng = np.random.default_rng(seed=new_seed)
    #rng = new_rng
    model = rsgnn_models.RSGNN(nfeat=features.shape[1], hid_dim=flags.hid_dim, num_reps=flags.num_reps, init=init)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr, weight_decay=0.0)

    def corrupt_graph(corrupt_rng):
        #seed_value = corrupt_rng.integers(0, np.iinfo(np.int32).max)
        #permuted_nodes = torch.randperm(graph['nodes'].shape[0], generator=torch.Generator().manual_seed(seed_value.item()))
        permuted_nodes = torch.randperm(graph['nodes'].shape[0],
                                        generator=torch.Generator().manual_seed(int(corrupt_rng)))
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

    def train_step(optimizer, graph, c_graph):
        def loss_fn():
            _, _, _, cluster_loss, logits = model(graph, c_graph)  # 将转换后的张量作为输入传递给模型
            dgi_loss = -torch.sum(torch.nn.functional.logsigmoid(labels * logits))
            return dgi_loss + flags.lambda_ * cluster_loss

        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward(retain_graph=True)
        optimizer.step()
        return optimizer, loss.item()

    best_keeper = BestKeeper('min')
    for epoch in range(1, flags.epochs + 1):
        #new_seed = rng.integers(0, np.iinfo(np.int32).max)
        #corrupt_rng = np.random.default_rng(seed=new_seed)
        rng, drop_rng, corrupt_rng = np.random.randint(low=0, high=10, size=3)
        c_graph = corrupt_graph(corrupt_rng)
        optimizer, loss = train_step(optimizer, graph, c_graph)
        print('Epoch:', epoch, 'Loss:', loss)
        if epoch % flags.valid_each == 0:
            best_keeper.update(epoch, loss, model.state_dict())

    model.load_state_dict(best_keeper.get())
    h, centers, rep_ids, _, _ = model(graph, c_graph)

    return centers, rep_ids.numpy()
