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
"""TPUDNNClassifier canned Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.estimator.canned import dnn as dnn_lib
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import training as train

_LEARNING_RATE = dnn_lib._LEARNING_RATE  # pylint: disable=protected-access


def _tpu_summary_no_op(*args, **kwargs):
  del args, kwargs
  if not _tpu_summary_no_op.has_logged_summary_warning:
    _tpu_summary_no_op.has_logged_summary_warning = True
    logging.warning(
        'tf.summary is suppressed on TPU (b/64980426, b/71940005).')


_tpu_summary_no_op.has_logged_summary_warning = False


@contextlib.contextmanager
def _rewire_summary_calls(
    use_tpu,
    tf_summary_scalar_replacement_fn=_tpu_summary_no_op,
    tf_summary_histogram_replacement_fn=_tpu_summary_no_op):
  """Rewires scalar() and histogram() to the methods given.

  Summaries are not yet supported on TPUs (b/64980426). So we provide this
  context manager that can swap out tf.summary.scalar and tf.summary.histogram
  to the replacement methods given. See brn_tpu_estimator.py for replacements.
  When the context exits, the rewiring is reset.

  Args:
    use_tpu: Whether the model is being executed on TPUs.
    tf_summary_scalar_replacement_fn: The replacement method for
      tf.summary.scalar.
    tf_summary_histogram_replacement_fn: The replacement method for
      tf.summary.histogram.

  Yields:
    None.
  """
  if use_tpu:
    original_tf_summary_scalar = summary.scalar
    original_tf_summary_histogram = summary.histogram
    summary.scalar = tf_summary_scalar_replacement_fn
    summary.histogram = tf_summary_histogram_replacement_fn
    yield
    summary.scalar = original_tf_summary_scalar
    summary.histogram = original_tf_summary_histogram
  else:
    yield


def _validate_optimizer(optimizer):
  if not isinstance(optimizer, optimizer_lib.Optimizer):
    raise ValueError(
        'For TPUDNNClassifier, `optimizer` argument must be a instance of '
        '`optimizer.Optimizer`.')
  if isinstance(optimizer, tpu_optimizer.CrossShardOptimizer):
    raise ValueError(
        '`optimizer` arg should not be of type '
        '`tf.contrib.tpu.CrossShardOptimizer`')


class TPUDNNClassifier(tpu_estimator.TPUEstimator):
  """A classifier for TensorFlow DNN models for TPUs."""

  def __init__(    # pylint: disable=invalid-name
      self,
      _sentinel=None,
      hidden_units=None,
      feature_columns=None,
      model_dir=None,
      n_classes=2,
      weight_column=None,
      label_vocabulary=None,
      optimizer=train.AdagradOptimizer(_LEARNING_RATE),
      activation_fn=nn.relu,
      dropout=None,
      input_layer_partitioner=None,
      config=None,
      loss_reduction=losses.Reduction.SUM,
      # TPU only args below
      batch_size=None,
      use_tpu=True,
  ):
    """Initializes a `TPUDNNClassifier` instance.

    Args:
      _sentinel: Used to prevent positional parameters. Internal, do not use.
      hidden_units: Iterable of number hidden units per layer. All layers are
        fully connected. Ex. `[64, 32]` means first layer has 64 nodes and
        second one has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `_FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: Number of label classes. Defaults to 2, namely binary
        classification. Must be > 1.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
      optimizer: An instance of `tf.Optimizer` used to train the model. Defaults
        to Adagrad optimizer.
      activation_fn: Activation function applied to each layer. If `None`, will
        use `tf.nn.relu`.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: `RunConfig` object to configure the runtime settings.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM`.
      batch_size: An int representing the global training batch size.
        TPUEstimator transforms this global batch size to a per-shard batch
        size, as params['batch_size'], when calling `input_fn` and `model_fn`.
        Cannot be `None` if `use_tpu` is `True`. Must be divisible by
        `config.tpu_config.num_shards`.
      use_tpu: A bool indicating whether TPU support is enabled. Currently,
        predict still happens on CPU.

    Raises:
      Exception: If arguments are not provided as key-word arguments or
        if required arguments are missing.
    """
    if _sentinel is not None:
      raise Exception('All arguments must be passed as key-word arguments.')
    if (not hidden_units) or (not feature_columns):
      raise Exception(
          'Both `hidden_units` and `feature_columns` are required arguments.')

    _validate_optimizer(optimizer)

    head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
        n_classes=n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
    if use_tpu:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # pylint: disable=unused-argument
    def _model_fn(features, labels, mode, config, params):
      """Implementation of DNN model_fn."""
      with _rewire_summary_calls(use_tpu):
        return dnn_lib._dnn_model_fn(  # pylint: disable=protected-access
            features=features,
            labels=labels,
            mode=mode,
            head=head,
            hidden_units=hidden_units,
            feature_columns=tuple(feature_columns or []),
            optimizer=optimizer,
            activation_fn=activation_fn,
            dropout=dropout,
            input_layer_partitioner=input_layer_partitioner,
            config=config,
            tpu_estimator_spec=use_tpu)

    embedding_config_spec = tpu_estimator.EmbeddingConfigSpec(
        feature_columns=feature_columns,
        learning_rate=_LEARNING_RATE,
    )
    super(TPUDNNClassifier, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        use_tpu=use_tpu,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        embedding_config_spec=embedding_config_spec,
    )
