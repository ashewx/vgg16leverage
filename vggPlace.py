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

  def get_mentor_variables_to_restore():
    """
    Returns:: names of the weights and biases of the teacher model
    """
    return [var for var in tf.global_variables() if var.op.name.startswith("mentor") and (var.op.name.endswith("biases") or var.op.name.endswith("weights")) and (var.op.name != ("mentor_fc3/mentor_weights") and var.op.name != ("mentor_fc3/mentor_biases"))]

### placeholders to hold iamges and their labels of certain datasets 
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
                       'cuda': '10000',
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

  def testBuild(self):
    graph = GraphPlacerTest._buildVgg()
    mg = meta_graph.create_meta_graph_def(graph=graph)
    gcluster = GraphPlacerTest._buildCluster(num_gpus=1)
    # Spend 15 seconds trying to optimize the placement of the model. This
    # should give us enough time to exercise the code, but not enough to find
    # a good placement, so we'll just check for legality.
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=15, cluster=gcluster, verbose=True)
    # node in placed_mg.graph_def.node:
     # print(node)


if __name__ == '__main__':
  placer = GraphPlacerTest()
  placer.testBuild()
