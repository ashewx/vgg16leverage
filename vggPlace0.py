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
from tensorflow.python.grappler import cluster as clusters
from tensorflow.python.grappler import graph_placer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from nets import vgg
import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["172.23.10.3:2222", "172.23.10.4:2223"]})
server1 = tf.train.Server(cluster, job_name="local", task_index=0)

class GraphPlacerTest():

  @staticmethod
  def _buildVgg():
    g = tf.Graph()

    train_batch_size = 2
    eval_batch_size = 1
    train_height, train_width = 224, 224
    eval_height, eval_width = 256, 256
    num_classes = 1000
    with g.as_default():
      train_inputs = tf.random_uniform(
          (train_batch_size, train_height, train_width, 3))
      logits, _ = vgg.vgg_16(train_inputs)
      tf.get_variable_scope().reuse_variables()
      eval_inputs = tf.random_uniform(
          (eval_batch_size, eval_height, eval_width, 3))
      logits, _ = vgg.vgg_16(eval_inputs, is_training=False,
                             spatial_squeeze=False)
      logits = tf.reduce_mean(logits, [1, 2])
      predictions = tf.argmax(logits, 1)

    train_op = g.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(predictions)
    return g

  @staticmethod
  def _buildCluster(num_cpus=1, num_gpus=1):
    devices = []
    if num_gpus > 0:
      device_properties = device_properties_pb2.DeviceProperties(
          type='GPU',
          vendor='NVidia',
          model='Tesla K40m',
          frequency=745, #745 MHZ
          num_cores= 2888, # CUDA Cores
          environment={'architecture': '5.2',
                       'cuda': '10000',
                       'cudnn': '7031'},
          num_registers=65536,
          l1_cache_size=65536, #64KB
          l2_cache_size=1572864, #1.5 MB
          shared_memory_size_per_multiprocessor=49152, #49152 bytes
          memory_size=12884901888, # 12GB
          bandwidth=288000000) #288 GBps)
      devices.append(
        device_properties_pb2.NamedDevice(
            properties=device_properties, name='/job:local/task:0/device:GPU:0'))
      devices.append(
        device_properties_pb2.NamedDevice(
            properties=device_properties, name='/job:local/task:1/device:GPU:0'))

    device_properties = device_properties_pb2.DeviceProperties(
        type='CPU',
        frequency=2399,
        num_cores=32,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=20971520)
    devices.append(
      device_properties_pb2.NamedDevice(
          properties=device_properties, name='/job:local/task:0/device:CPU:0'))
    devices.append(
      device_properties_pb2.NamedDevice(
          properties=device_properties, name='/job:local/task:1/device:CPU:0'))

    return clusters.Cluster(devices=devices)

  def testBuild(self):
    graph = GraphPlacerTest._buildVgg()
    mg = meta_graph.create_meta_graph_def(graph=graph)
    #gcluster = cluster.Cluster(devices=None) # Automatically generates local machine cluster
    gcluster = GraphPlacerTest._buildCluster()
    print(gcluster.ListDevices()) # Print clust info
    # Spend 15 seconds trying to optimize the placement of the model. This
    # should give us enough time to exercise the code, but not enough to find
    # a good placement, so we'll just check for legality.
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=108000, cluster=gcluster, verbose=True)
    placed_g = placed_mg.graph_def;
    meta_graph.export_scoped_meta_graph(filename="./g/g.meta", graph_def=placed_g)
    # node in placed_mg.graph_def.node:
     # print(node)


if __name__ == '__main__':
  placer = GraphPlacerTest()
  placer.testBuild()
