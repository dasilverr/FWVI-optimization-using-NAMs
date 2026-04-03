# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Neural net models for tabular datasets."""

from typing import Union
import numpy as np
import tensorflow.compat.v1 as tf

TfInput = Union[np.ndarray, tf.Tensor]


def exu(x, weight, bias):
  """ExU hidden unit modification."""
  return tf.exp(weight) * (x - bias)


def relu(x, weight, bias):
  """ReLU activation."""
  return tf.nn.relu(weight * (x - bias))


def relu_n(x, n=1):
  """ReLU activation clipped at n."""
  return tf.clip_by_value(x, 0, n)


class ActivationLayer(tf.keras.layers.Layer):
  """Custom activation layer supporting ExU and ReLU hidden units."""

  def __init__(self, num_units, name=None, activation='exu', trainable=True):
    super(ActivationLayer, self).__init__(trainable=trainable, name=name)
    self.num_units = num_units
    self._trainable = trainable
    if activation == 'relu':
      self._activation = relu
      self._beta_initializer = 'glorot_uniform'
    elif activation == 'exu':
      self._activation = lambda x, w, b: relu_n(exu(x, w, b))
      self._beta_initializer = tf.initializers.truncated_normal(mean=4.0, stddev=0.5)
    else:
      raise ValueError(f'{activation} is not a valid activation')

  def build(self, input_shape):
    self._beta = self.add_weight(
        name='beta', shape=[input_shape[-1], self.num_units],
        initializer=self._beta_initializer, trainable=self._trainable)
    self._c = self.add_weight(
        name='c', shape=[1, self.num_units],
        initializer=tf.initializers.truncated_normal(stddev=0.5),
        trainable=self._trainable)
    super(ActivationLayer, self).build(input_shape)

  @tf.function
  def call(self, x):
    center = tf.tile(self._c, [tf.shape(x)[0], 1])
    return self._activation(x, self._beta, center)


class FeatureNN(tf.keras.layers.Layer):
  """Per-feature neural network used inside NAM.

  A shallow network uses a single ActivationLayer. A deep network adds
  two Dense ReLU layers (64 and 32 units) on top.
  """

  def __init__(self, num_units, num_outputs=1, dropout=0.5, trainable=True,
               shallow=True, feature_num=0, name_scope='model', activation='exu'):
    super(FeatureNN, self).__init__()
    self._num_units = num_units
    self._num_outputs = num_outputs
    self._dropout = dropout
    self._trainable = trainable
    self._tf_name_scope = name_scope
    self._feature_num = feature_num
    self._shallow = shallow
    self._activation = activation

  def build(self, input_shape):
    self.hidden_layers = [
        ActivationLayer(self._num_units, trainable=self._trainable,
                        activation=self._activation,
                        name=f'activation_layer_{self._feature_num}')
    ]
    if not self._shallow:
      self._h1 = tf.keras.layers.Dense(64, activation='relu', use_bias=True,
                                        trainable=self._trainable,
                                        name=f'h1_{self._feature_num}',
                                        kernel_initializer='glorot_uniform')
      self._h2 = tf.keras.layers.Dense(32, activation='relu', use_bias=True,
                                        trainable=self._trainable,
                                        name=f'h2_{self._feature_num}',
                                        kernel_initializer='glorot_uniform')
      self.hidden_layers += [self._h1, self._h2]
    self.linear = tf.keras.layers.Dense(self._num_outputs, use_bias=False,
                                         trainable=self._trainable,
                                         name=f'dense_{self._feature_num}',
                                         kernel_initializer='glorot_uniform')
    super(FeatureNN, self).build(input_shape)

  @tf.function
  def call(self, x, training):
    with tf.name_scope(self._tf_name_scope):
      for layer in self.hidden_layers:
        x = tf.nn.dropout(layer(x),
                          rate=tf.cond(training, lambda: self._dropout, lambda: 0.0))
      x = self.linear(x)
      if self._num_outputs == 1:
        x = tf.squeeze(x, axis=1)
    return x


class NAM(tf.keras.Model):
  """Neural Additive Model: one FeatureNN per input feature, outputs summed.

  Supports multi-class classification via num_outputs > 1.
  """

  def __init__(self, num_inputs, num_units, num_outputs=1, trainable=True,
               shallow=True, feature_dropout=0.0, dropout=0.0, **kwargs):
    super(NAM, self).__init__()
    self._num_inputs = num_inputs
    self._num_units = num_units if isinstance(num_units, list) else [num_units] * num_inputs
    assert not isinstance(num_units, list) or len(num_units) == num_inputs
    self._num_outputs = num_outputs
    self._trainable = trainable
    self._shallow = shallow
    self._feature_dropout = feature_dropout
    self._dropout = dropout
    self._kwargs = kwargs

  def build(self, input_shape):
    self.feature_nns = [
        FeatureNN(num_units=self._num_units[i], num_outputs=self._num_outputs,
                  dropout=self._dropout, trainable=self._trainable,
                  shallow=self._shallow, feature_num=i, **self._kwargs)
        for i in range(self._num_inputs)
    ]
    self._bias = self.add_weight(
        name='bias', initializer=tf.keras.initializers.Zeros(),
        shape=(self._num_outputs,), trainable=self._trainable)
    self._true = tf.constant(True, dtype=tf.bool)
    self._false = tf.constant(False, dtype=tf.bool)

  def call(self, x, training=True):
    individual_outputs = self.calc_outputs(x, training=training)
    stacked_out = tf.stack(individual_outputs, axis=-1)
    training_flag = self._true if training else self._false

    if self._feature_dropout > 0.0:
      noise_shape = ([1, tf.shape(stacked_out)[1]] if self._num_outputs == 1
                     else [1, 1, tf.shape(stacked_out)[2]])
      stacked_out = tf.nn.dropout(
          stacked_out,
          rate=tf.cond(training_flag, lambda: self._feature_dropout, lambda: 0.0),
          noise_shape=noise_shape)

    return tf.reduce_sum(stacked_out, axis=-1) + self._bias

  def _name_scope(self):
    tf_name_scope = self._kwargs.get('name_scope', None)
    base = super(NAM, self)._name_scope()  # pytype: disable=attribute-error  # typed-keras
    return f'{tf_name_scope}/{base}' if tf_name_scope else base

  def calc_outputs(self, x, training=True):
    """Returns a list of per-feature FeatureNN outputs."""
    training = self._true if training else self._false
    return [self.feature_nns[i](x_i, training=training)
            for i, x_i in enumerate(tf.split(x, self._num_inputs, axis=-1))]


class DNN(tf.keras.Model):
  """10-layer ReLU DNN baseline."""

  def __init__(self, trainable=True, dropout=0.15):
    super(DNN, self).__init__()
    self._dropout = dropout
    self.hidden_layers = [
        tf.keras.layers.Dense(100, activation='relu', use_bias=True,
                              trainable=trainable, name=f'dense_{i}',
                              kernel_initializer='he_normal')
        for i in range(10)
    ]
    self.linear = tf.keras.layers.Dense(1, use_bias=True, trainable=trainable,
                                         name='linear', kernel_initializer='he_normal')
    self._true = tf.constant(True, dtype=tf.bool)
    self._false = tf.constant(False, dtype=tf.bool)

  def call(self, x, training=True):
    training = self._true if training else self._false
    for layer in self.hidden_layers:
      x = tf.nn.dropout(layer(x),
                        rate=tf.cond(training, lambda: self._dropout, lambda: 0.0))
    return tf.squeeze(self.linear(x), axis=-1)
