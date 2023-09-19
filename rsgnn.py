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

import data_utils
import trainer

import common_args

args = common_args.parser.parse_args()


# 为rsgnn和gcn_c创建命名空间对象
def get_rsgnn_flags(num_classes):
    return types.SimpleNamespace(
        hid_dim=args.rsgnn_hid_dim,
        epochs=args.rsgnn_epochs,
        num_classes=num_classes,
        num_reps=args.num_reps_multiplier * num_classes,
        valid_each=args.valid_each,
        lr=args.lr,
        lambda_=args.lambda_)


def representation_selection(subset, select_round, labeled_set, lbl, nlbl, centers):
    """Runs node selector, receives selected nodes, trains GCN."""
    np.random.seed(args.seed)
    key = np.random.default_rng(args.seed)  # 设置随机种子
    graph, labels, num_classes = data_utils.create_jraph(subset, labeled_set, select_round)  # 加载图、label和类别数
    rsgnn_flags = get_rsgnn_flags(num_classes)  # 获取rsgnn的超参配置
    centers, selected = trainer.train_rsgnn(rsgnn_flags, graph, key, lbl, nlbl, centers)  # 使用rsgnn并获取learned embeddings和选定的节点
    return centers, selected.tolist()
