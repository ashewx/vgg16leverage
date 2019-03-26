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
from vgg16mentor import Mentor
import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server1 = tf.train.Server(cluster, job_name="local", task_index=0)

batch_size = 45
image_height = 224
image_width = 224
num_channels = 3
num_classes = 102
temp_softmax = 1
seed = 1234
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5853
LEARNING_RATE_DECAY_FACTOR = 0.9809
NUM_EPOCHS_PER_DECAY = 1.0
learning_rate = 0.0001
learning_rate_pretrained = 0.0001

class GraphPlacerTest():

  @staticmethod
  def _buildVgg():
    mentor = Mentor(True)
    g = tf.Graph()

    with g.as_default():
      config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
      ## set the seed so that we have same loss values and initializations for every run.
      tf.set_random_seed(seed)
      images_placeholder, labels_placeholder = GraphPlacerTest.placeholder_inputs(batch_size)
      sess = tf.Session(config = config)
      coord = tf.train.Coordinator()
      global_step = tf.Variable(0, name='global_step', trainable=True)
      phase_train = tf.placeholder(tf.bool, name = 'phase_train')

      num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
      decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
      mentor_data_dict = mentor.build(images_placeholder, num_classes, temp_softmax, phase_train)
      loss = mentor.loss(labels_placeholder)
      lr = tf.train.exponential_decay(learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)

      variables_to_restore = GraphPlacerTest.get_mentor_variables_to_restore()
      train_op = mentor.training(loss, learning_rate_pretrained, lr, global_step, variables_to_restore,mentor.get_training_vars())
      softmax = mentor_data_dict.softmax
      init = tf.global_variables_initializer()

    train_op = g.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(init)
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

  @staticmethod
  def get_mentor_variables_to_restore():
    """
    Returns:: names of the weights and biases of the teacher model
    """
    return [var for var in tf.global_variables() if var.op.name.startswith("mentor") and (var.op.name.endswith("biases") or var.op.name.endswith("weights")) and (var.op.name != ("mentor_fc3/mentor_weights") and var.op.name != ("mentor_fc3/mentor_biases"))]

### placeholders to hold iamges and their labels of certain datasets
  @staticmethod
  def placeholder_inputs(batch_size):
    """
      Args:
        batch_size: batch size used to train the network

      Returns:
        images_placeholder: images_placeholder holds images of either caltech101 or cifar10 datasets
        labels_placeholder: labels_placeholder holds labels of either caltech101 or cifar10 datasets

    """
    images_placeholder = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_height,
                                                     image_width, num_channels))
    labels_placeholder = tf.placeholder(tf.int32,
                                          shape=(batch_size))

    return images_placeholder, labels_placeholder

  def testBuild(self):
    graph = GraphPlacerTest._buildVgg()
    mg = meta_graph.create_meta_graph_def(graph=graph)
    #gcluster = cluster.Cluster(devices=None) # Automatically generates local machine cluster
    gcluster = GraphPlacerTest._buildCluster()
    print(gcluster.ListDevices()) # Print clust info
    # Spend 15 seconds trying to optimize the placement of the model. This
    # should give us enough time to exercise the code, but not enough to find
    # a good placement, so we'll just check for legality.
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=300, cluster=gcluster, verbose=True)
    placed_g = placed_mg.graph_def;
    meta_graph.export_scoped_meta_graph(filename="./g/g.meta", graph_def=placed_g)
    # node in placed_mg.graph_def.node:
     # print(node)


if __name__ == '__main__':
  placer = GraphPlacerTest()
  placer.testBuild()
