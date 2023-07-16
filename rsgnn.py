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

import jax
import jax.numpy as jnp
import numpy as np

import data_utils
import trainer

import common_args
args = common_args.parser.parse_args()


# 从unlabeled nodes中选择验证集和测试集
def create_splits(train_nodes, num_nodes):
    train_idx = np.array([False] * int(num_nodes))
    train_idx[train_nodes] = True
    valid_nodes = np.random.choice(np.where(np.logical_not(train_idx))[0],
                                   args.num_valid_nodes, replace=False)
    valid_idx = np.array([False] * int(num_nodes))
    valid_idx[valid_nodes] = True
    test_idx = np.logical_not(np.logical_or(train_idx, valid_idx))
    return types.SimpleNamespace(train=jnp.array(train_idx),
                                 valid=jnp.array(valid_idx),
                                 test=jnp.array(test_idx))


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


def get_gcn_c_flags(num_classes):
    return types.SimpleNamespace(
        hid_dim= args.gcn_c_hid_dim,
        epochs=args.gcn_c_epochs,
        num_classes=num_classes,
        valid_each=args.valid_each,
        lr=args.lr,
        drop_rate=args.drop_rate,
        w_decay=args.gcn_c_w_decay)


def representation_selection():
    """Runs node selector, receives selected nodes, trains GCN."""
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)  # 设置随机种子
    graph, labels, num_classes = data_utils.create_jraph()  # 加载图、label和类别数
    rsgnn_flags = get_rsgnn_flags(num_classes)  # 获取rsgnn的超参配置
    selected = trainer.train_rsgnn(rsgnn_flags, graph, key)  # 使用rsgnn并获取选定的节点
    return selected.tolist()
    # key, gcn_key = jax.random.split(key)
    # splits = create_splits(selected, graph.n_node[0])  # 划分数据集为训练、验证和测试集
    # gcn_c_flags = get_gcn_c_flags(num_classes)  # 获取gcn_c的超参配置
    # gcn_accu = trainer.train_gcn(gcn_c_flags, graph, labels, gcn_key, splits)  # 使用gcn_c并获取测试准确率
    # print(f'GCN Test Accuracy: {gcn_accu}')

# if __name__ == '__main__':
#    app.run(main)
