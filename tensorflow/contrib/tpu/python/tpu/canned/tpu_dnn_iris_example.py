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
"""Train a DNN using canned TPUDNNClassifier on Iris dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import model_selection

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu.canned import tpu_dnn

FLAGS = tf.flags.FLAGS

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
tf.flags.DEFINE_string(
    'tpu_zone', default=None,
    help='[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')
tf.flags.DEFINE_string(
    'gcp_project', default=None,
    help='[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')
tf.flags.DEFINE_bool(
    'use_tpu', True,
    help='Use TPU to execute the model for training and evaluation. If'
         ' --use_tpu=false, will use whatever devices are available to'
         ' TensorFlow by default (e.g. CPU and GPU)')

# Model specific parameters.
tf.flags.DEFINE_string(
    'model_dir', default=None,
    help='The directory where the model and training/evaluation summaries are'
         ' stored.')
tf.flags.DEFINE_integer(
    'batch_size', default=32, help='Batch size for training and eval.')
tf.flags.DEFINE_integer(
    'iterations_per_loop', default=10,
    help='Number of steps to run on TPU before outfeeding metrics to the CPU.')
tf.flags.DEFINE_integer(
    'num_epochs', default=100,
    help='Number of training epochs to run. Each epoch will run through the '
         'entire training set and then eval.')
tf.flags.DEFINE_integer(
    'num_shards', default=2,
    help='Number of TPU cores. For a single TPU device, this is 8 because each'
         ' TPU has 4 chips each with 2 cores.')


FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
NUM_CLASSES = 3


def _create_input_fn(np_features, np_labels, mode):
  """Creates an input_fn required by Estimator train/evaluate."""

  feature_dict = {
      key: np_features[:, i]
      for i, key in enumerate(FEATURE_KEYS)
  }

  def _input_fn(params):
    """The input_fn."""
    batch_size = params['batch_size']

    dataset = tf.data.Dataset.from_tensor_slices(
        (feature_dict, np_labels))

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=np_labels.size)
      dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return _input_fn


def main(unused_argv):
  iris = datasets.load_iris()

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      iris.data.astype(np.float32), iris.target,
      test_size=0.25, random_state=42)

  train_input_fn = _create_input_fn(mode=tf.estimator.ModeKeys.TRAIN,
                                    np_features=x_train,
                                    np_labels=y_train)
  eval_input_fn = _create_input_fn(mode=tf.estimator.ModeKeys.EVAL,
                                   np_features=x_test,
                                   np_labels=y_test)

  feature_columns = [tf.feature_column.numeric_column(key)
                     for key in FEATURE_KEYS]

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )

  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards),
  )

  # Build 3 layer DNN with 10, 5 units respectively.
  classifier = tpu_dnn.TPUDNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[10, 5],
      n_classes=NUM_CLASSES,
      use_tpu=FLAGS.use_tpu,
      config=config,
      batch_size=FLAGS.batch_size)

  for _ in range(FLAGS.num_epochs):
    classifier.train(
        input_fn=train_input_fn,
        steps=int(y_train.size // FLAGS.batch_size))
    classifier.evaluate(
        input_fn=eval_input_fn,
        steps=int(y_test.size // FLAGS.batch_size))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
