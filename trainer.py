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


class BestKeeper:
    """Keeps best performance and model params during training."""

    def __init__(self, min_or_max):
        self.min_or_max = min_or_max
        self.best_result = np.inf if min_or_max == 'min' else 0.0
        self.best_params = None

    def print_(self, epoch, result):
        if self.min_or_max == 'min':
            print('Epoch:', epoch, 'Loss:', result)
        elif self.min_or_max == 'max':
            print('Epoch:', epoch, 'Accu:', result)

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

    def get(self):
        return self.best_params


def train_rsgnn(flags, graph, rng, lbl, nlbl, new_centers_indices):
    """Trainer function for RS-GNN."""
    features = graph['nodes']
    n_nodes = graph['n_node'][0]
    labels = torch.cat([torch.ones(n_nodes), -torch.ones(n_nodes)], dim=0)
    num_reps = flags.num_reps if lbl is None else flags.num_reps + len(lbl) 
    model = rsgnn_models.RSGNN(nfeat=features.shape[1], hid_dim=flags.hid_dim, num_reps=num_reps,
                               new_centers_indices=new_centers_indices)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr, weight_decay=0.0)

    def corrupt_graph():
        permuted_nodes = torch.randperm(graph['nodes'].shape[0])
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
            outputs, _, _, cluster_loss, logits = model(graph, c_graph, lbl)
            dgi_loss = -torch.sum(torch.nn.functional.logsigmoid(labels * logits))
            print(dgi_loss)
            print(cluster_loss)
            outputs = torch.sigmoid(outputs).mean(dim=1, keepdim=True)

            if np.any(lbl == None):
                bce_loss = 0
            else:
                # Create target tensors for labeled and unlabeled samples
                labeled_targets = torch.ones_like(torch.from_numpy(lbl))
                unlabeled_targets = torch.zeros_like(torch.from_numpy(nlbl))
                # Combine labeled and unlabeled targets
                targets = torch.cat((labeled_targets, unlabeled_targets), dim=0).view(-1, 1).float()
                # Calculate BCE loss
                bceloss = torch.nn.BCEWithLogitsLoss(reduction='sum')
                bce_loss = bceloss(outputs, targets)

            print(bce_loss)
            return dgi_loss + flags.cluster_loss_lambda * cluster_loss + flags.bce_loss_lambda * bce_loss

        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward(retain_graph=True)
        optimizer.step()
        return optimizer, loss.item()

    best_keeper = BestKeeper('min')

    for epoch in range(1, flags.epochs + 1):
        c_graph = corrupt_graph()
        optimizer, loss = train_step(optimizer, graph, c_graph)
        print('Epoch:', epoch, 'Loss:', loss)
        if epoch % flags.valid_each == 0:
            best_keeper.update(epoch, loss, model.state_dict())

    model.load_state_dict(best_keeper.get())
    h, centers, rep_ids, _, _ = model(graph, c_graph, lbl)

    return centers, rep_ids.numpy()
