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

"""RS-GNN Implementation."""

import os

# The experiments in the paper were done when lazy_rng's default value was false
# Since then, the default value has changed to true.
# Setting it back to false for consistency.
os.environ['FLAX_LAZY_RNG'] = 'false'
# pylint: disable=g-import-not-at-top
import types
import numpy as np
import torch
from kcenterGreedy import kCenterGreedy
from config import *

import data_utils
import trainer

import common_args

args = common_args.parser.parse_args()


def get_rsgnn_flags(num_classes):
    return types.SimpleNamespace(
        hid_dim=args.rsgnn_hid_dim,
        epochs=args.rsgnn_epochs,
        num_classes=num_classes,
        num_reps=args.num_reps_multiplier * num_classes,
        valid_each=args.valid_each,
        lr=args.lr,
        cluster_loss_lambda=args.cluster_loss_lambda,
        bce_loss_lambda=args.bce_loss_lambda
        )


def get_kcg(labeled_data_size, features):
    feat = features.detach().cpu().numpy()
    # index of datapoints already selected
    new_av_idx = np.arange(SUBSET, (SUBSET + labeled_data_size))
    sampling = kCenterGreedy(feat)
    # indices of selected unlabeled datapoints
    batch = sampling.select_batch_(new_av_idx, ADDENDUM)
    return batch


def representation_selection(models, subset, select_round, labeled_set, lbl, nlbl, cycle):
    """Runs node selector, receives selected nodes, trains GCN."""
    np.random.seed(args.seed)
    key = np.random.default_rng(args.seed)
    graph, labels, num_classes = data_utils.create_jraph(models, subset, labeled_set, select_round)
    new_centers_indices = get_kcg(ADDENDUM * (cycle + 1), graph['nodes'])
    rsgnn_flags = get_rsgnn_flags(num_classes)
    centers, selected = trainer.train_rsgnn(rsgnn_flags, graph, key, lbl, nlbl, new_centers_indices)
    return centers, selected.tolist()
