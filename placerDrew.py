# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests the graph placer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import graph_placer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class GraphPlacerTest():

  @staticmethod
  def _buildMnist(batch_size=128,
                  input_size=256,
                  num_classes=1024,
                  num_layers=10,
                  hidden_size=256,
                  name='mnist'):
    g = tf_ops.get_default_graph()
    with g.as_default():
      ops = {}
      x = random_ops.random_uniform(
          [batch_size, input_size], -0.1, 0.1, dtype=dtypes.float32)
      for layer_id in range(num_layers):
        with variable_scope.variable_scope('layer_{}'.format(layer_id)):
          a = input_size if layer_id == 0 else hidden_size
          b = hidden_size if layer_id < num_layers - 1 else num_classes
          w = variable_scope.get_variable('w', [a, b])
          x = math_ops.matmul(x, w)
          x = nn_ops.relu(x)
      ops['y_preds'] = math_ops.argmax(x, axis=1)

    train_op = g.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(ops['y_preds'])
    return g

  @staticmethod
  def _buildCluster(num_cpus=1, num_gpus=1):
    devices = []
    if num_gpus > 0:
        # Specs found here https://repository.asu.edu/attachments/178487/content/Kannan_asu_0010N_16599.pdf
        # https://www.microway.com/knowledge-center-articles/in-depth-comparison-of-nvidia-tesla-kepler-gpu-accelerators/
      device_properties = device_properties_pb2.DeviceProperties(
          type='GPU',
          vendor='NVidia',
          model='Tesla K40m',
          frequency=745, #745 MHZ
          num_cores= 2888, # CUDA Cores
          environment={'architecture': '5.2',
                       'cuda': '9000',
                       'cudnn': '7031'},
          num_registers=65536,
          l1_cache_size=65536, #64KB
          l2_cache_size=1572864, #1.5 MB
          shared_memory_size_per_multiprocessor=49152, #49152 bytes
          memory_size=12884901888, # 12GB
          bandwidth=288000000) #288 GBps
      for i in range(num_gpus):
        devices.append(
            device_properties_pb2.NamedDevice(
                properties=device_properties, name='/GPU:' + str(i)))

    assert num_cpus > 0
    device_properties = device_properties_pb2.DeviceProperties(
        # Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz
        type='CPU',
        vendor='Intel',
        model='Haswell',
        frequency=2400, #2.4 GHz
        num_cores= 8,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=20971520)
    for i in range(num_cpus):
      devices.append(
          device_properties_pb2.NamedDevice(
              properties=device_properties, name='/CPU:' + str(i)))
    for i in devices:
      print(i)
    return cluster.Cluster(devices=devices)

  def testBasic(self):
    """Place a trivial graph."""
    a = constant_op.constant(10, name='a')
    b = constant_op.constant(20, name='b')
    c = math_ops.add_n([a, b], name='c')
    d = math_ops.add_n([b, c], name='d')
    train_op = tf_ops.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=tf_ops.get_default_graph())

    gcluster = cluster.Cluster()
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=3600, cluster=gcluster)

  def testMNIST(self):
    graph = GraphPlacerTest._buildMnist()
    mg = meta_graph.create_meta_graph_def(graph=graph)
    gcluster = GraphPlacerTest._buildCluster(num_gpus=1)
    # Spend 15 seconds trying to optimize the placement of the model. This
    # should give us enough time to exercise the code, but not enough to find
    # a good placement, so we'll just check for legality.
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=120, cluster=gcluster, verbose=True)
    # node in placed_mg.graph_def.node:
     # print(node)


if __name__ == '__main__':
  placer = GraphPlacerTest()
  placer.testMNIST()
