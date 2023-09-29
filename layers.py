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
from gcn_model import GCN
from load_dataset import load_dataset
from data_utils import knn_similarity_graph


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


class Bilinear(nn.Module):
    """A Bilinear Layer."""

    def __init__(self, in_features_l, in_features_r):
        super(Bilinear, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(in_features_l, in_features_r))
        nn.init.normal_(self.kernel)

    def forward(self, x_l, x_r):
        return torch.matmul(torch.matmul(x_l, self.kernel), x_r)


class EucCluster(nn.Module):
    """Learnable KMeans Clustering."""

    def __init__(self, num_reps, d_model, new_centers_indices=None, init_fn=nn.init.normal_):
        super(EucCluster, self).__init__()
        self.num_reps = num_reps
        self.init_fn = init_fn
        self.new_centers_indices = new_centers_indices
        self.are_centers_initialized = new_centers_indices is None
        self.centers = nn.Parameter(self.init_fn(torch.empty(self.num_reps, d_model)))

    def forward(self, x, lbl=None):
        if not self.are_centers_initialized:
            self.are_centers_initialized = True
            # Initialize representatives with embeddings of provided node indices
            with torch.no_grad():
                self.centers[-len(self.new_centers_indices):].set_(x[self.new_centers_indices])
        if lbl is not None:
            # Fix representatives for labeled nodes
            with torch.no_grad():
                self.centers[:len(lbl)].set_(x[lbl])
        dists = torch.cdist(x, self.centers, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        return find_unique_min_indices(dists, lbl=lbl), torch.min(dists, dim=1)[0], self.centers


# TODO: technically the result is only used at inference, so perhaps use costly version below but only outside training
@torch.no_grad()
def find_unique_min_indices(dists, lbl=None):
    # Ignore labeled nodes for selecting next representatives
    if lbl is not None:
        mask = torch.ones(dists.shape[0], dtype=bool, device=dists.device)
        mask[lbl] = False
        dists = dists[mask, len(lbl) - dists.shape[1]:]

    n, m = dists.shape

    closest_ids = dists.argmin(0)
    while True:
        closest_ids = closest_ids.unique()
        if closest_ids.shape[0] == m:
            break
        # Randomly sample further representatives
        closest_ids = torch.concatenate(
            (closest_ids, torch.randint(n, (m - closest_ids.shape[0],))))

    if lbl is not None:
        # Map back to the original indices
        closest_ids = torch.where(mask)[0][closest_ids]

    return closest_ids

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


def dgi_readout(node_embs):
    return torch.sigmoid(torch.mean(node_embs, dim=0))


def subtract_mean(embs):
    return embs - torch.mean(embs, dim=0)


def divide_by_l2_norm(embs):
    norm = torch.norm(embs, dim=1, keepdim=True)
    return embs / norm


def normalize(node_embs):
    return divide_by_l2_norm(subtract_mean(node_embs))
