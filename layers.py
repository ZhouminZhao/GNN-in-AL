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

"""Layers in jraph/flax."""

from typing import Any, Callable
from flax import linen as nn
import jax
import jax.numpy as jnp


# Parametric ReLU层
class PReLU(nn.Module):
  """A PReLU Layer."""
  init_fn: Callable[[Any], Any] = nn.initializers.uniform()

  @nn.compact
  def __call__(self, x):
    leakage = self.param('leakage', self.init_fn, [1])
    return jnp.maximum(0, x) + leakage * jnp.minimum(0, x)


# 激活函数层
class Activation(nn.Module):
  """Activation function."""
  activation: str

  def setup(self):
    if self.activation == 'ReLU':
      self.act_fn = nn.relu
    elif self.activation == 'SeLU':
      self.act_fn = jax.nn.selu
    elif self.activation == 'PReLU':
      self.act_fn = PReLU()
    else:
      raise 'Activation not recognized'

  def __call__(self, x):
    return self.act_fn(x)


# 双线性层
class Bilinear(nn.Module):
  """A Bilinear Layer."""
  init_fn: Callable[[Any], Any] = nn.initializers.normal()

  @nn.compact
  def __call__(self, x_l, x_r):
    kernel = self.param('kernel', self.init_fn, [x_l.shape[-1], x_r.shape[-1]])
    return x_l @ kernel @ jnp.transpose(x_r)


# KMeans聚类层，接收输入的x，返回聚类结果、最小距离和聚类中心
class EucCluster(nn.Module):
  """Learnable KMeans Clustering."""
  num_reps: int
  init_fn: Callable[[Any], Any] = nn.initializers.normal()

  @nn.compact
  def __call__(self, x):
    centers = self.param('centers', self.init_fn, [self.num_reps, x.shape[-1]])
    dists = jnp.sqrt(pairwise_sqeuc_dists(x, centers))
    return jnp.argmin(dists, axis=0), jnp.min(dists, axis=1), centers


# DGI (Deep Graph Infomax)的读出函数，接收节点表示node_embs，应用sigmoid函数
@jax.jit
def dgi_readout(node_embs):
  return jax.nn.sigmoid(jnp.mean(node_embs, axis=0))


# 减去均值的函数
def subtract_mean(embs):
  return embs - jnp.mean(embs, axis=0)


# 除以L2范数的函数
def divide_by_l2_norm(embs):
  norm = jnp.linalg.norm(embs, axis=1, keepdims=True)
  return embs / norm


# 归一化节点表示，先减去均值，再除以L2范数
@jax.jit
def normalize(node_embs):
  return divide_by_l2_norm(subtract_mean(node_embs))


# 计算两组样本之间的成对平方欧几里得距离
@jax.jit
def pairwise_sqeuc_dists(x, y):
  """Pairwise square Euclidean distances."""
  n = x.shape[0]
  m = y.shape[0]
  x_exp = jnp.expand_dims(x, axis=1).repeat(m, axis=1).reshape(n * m, -1)
  y_exp = jnp.expand_dims(y, axis=0).repeat(n, axis=0).reshape(n * m, -1)
  return jnp.sum(jnp.power(x_exp - y_exp, 2), axis=1).reshape(n, m)
