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

"""Build a deep GAM model graph."""

import functools
from typing import Union, Callable, Dict
import warnings
import numpy as np
from sklearn import metrics as sk_metrics
import tensorflow.compat.v1 as tf

from neural_additive_models import models

warnings.filterwarnings('ignore')
TfInput = models.TfInput
LossFunction = Callable[[tf.keras.Model, TfInput, TfInput], tf.Tensor]
GraphOpsAndTensors = Dict[str, Union[tf.Tensor, tf.Operation, tf.keras.Model]]
EvaluationMetric = Callable[[tf.Session], float]


def cross_entropy_loss(model, inputs, targets, class_weights):
  """Sparse softmax cross-entropy loss, optionally weighted by ENS class weights."""
  predictions = model(inputs, training=True)
  labels = tf.cast(targets, tf.int32)
  loss_vals = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=predictions)
  if class_weights is not None:
    weights = tf.gather(tf.cast(class_weights, tf.float32), labels)
    return tf.reduce_mean(loss_vals * weights)
  return tf.reduce_mean(loss_vals)


def penalized_loss(loss_func, model, inputs, targets,
                   output_regularization, l2_regularization=0.0, use_dnn=False):
  """Computes penalized loss with optional output and L2 regularization."""
  loss = loss_func(model, inputs, targets)
  reg_loss = 0.0
  if output_regularization > 0:
    reg_loss += output_regularization * feature_output_regularization(model, inputs)
  if l2_regularization > 0:
    num_networks = 1 if use_dnn else len(model.feature_nns)
    reg_loss += l2_regularization * weight_decay(model, num_networks=num_networks)
  return loss + reg_loss


def penalized_cross_entropy_loss(model, inputs, targets, output_regularization,
                                  l2_regularization=0.0, use_dnn=False,
                                  class_weights=None):
  """Cross-entropy loss with optional ENS weighting, L2, and output regularization."""
  loss_fn = functools.partial(cross_entropy_loss, class_weights=class_weights)
  return penalized_loss(loss_fn, model, inputs, targets,
                        output_regularization, l2_regularization, use_dnn)


def penalized_mse_loss(model, inputs, targets, output_regularization,
                       l2_regularization=0.0, use_dnn=False):
  """Mean Squared Error with L2 regularization and output penalty."""
  return penalized_loss(mse_loss, model, inputs, targets,
                        output_regularization, l2_regularization, use_dnn)


def feature_output_regularization(model, inputs):
  """Penalizes the L2 norm of the prediction of each feature net."""
  per_feature_outputs = model.calc_outputs(inputs, training=False)
  per_feature_norm = [tf.reduce_mean(tf.square(o)) for o in per_feature_outputs]
  return tf.add_n(per_feature_norm) / len(per_feature_norm)


def weight_decay(model, num_networks=1):
  """Penalizes the L2 norm of weights in each feature net."""
  l2_losses = [tf.nn.l2_loss(x) for x in model.trainable_variables]
  return tf.add_n(l2_losses) / num_networks


def mse_loss(model, inputs, targets):
  """Mean squared error loss for regression."""
  predicted = model(inputs, training=True)
  return tf.losses.mean_squared_error(predicted, targets)


def macro_f1_score(sess, y_true, pred_tensor, dataset_init_op):
  """Calculates Macro F1 for multi-class classification.

  Used as the primary model selection metric to handle class imbalance.
  """
  y_pred_probs = generate_predictions(pred_tensor, dataset_init_op, sess)
  y_pred_class = np.argmax(y_pred_probs, axis=1)
  return sk_metrics.f1_score(y_true, y_pred_class, average='macro', zero_division=0)


def accuracy(sess, y_true, pred_tensor, dataset_init_op):
  """Calculates accuracy for multi-class classification."""
  y_pred_probs = generate_predictions(pred_tensor, dataset_init_op, sess)
  y_pred_class = np.argmax(y_pred_probs, axis=1)
  return sk_metrics.accuracy_score(y_true, y_pred_class)


def comprehensive_metrics(sess, y_true, pred_tensor, dataset_init_op):
  """Calculates Accuracy, Precision, Recall, and Macro F1."""
  y_pred_probs = generate_predictions(pred_tensor, dataset_init_op, sess)
  y_pred_class = np.argmax(y_pred_probs, axis=1)
  acc  = sk_metrics.accuracy_score(y_true, y_pred_class)
  prec = sk_metrics.precision_score(y_true, y_pred_class, average='macro', zero_division=0)
  rec  = sk_metrics.recall_score(y_true, y_pred_class, average='macro', zero_division=0)
  f1   = sk_metrics.f1_score(y_true, y_pred_class, average='macro', zero_division=0)
  return acc, prec, rec, f1


def generate_predictions(pred_tensor, dataset_init_op, sess):
  """Iterates over the dataset to collect model predictions."""
  sess.run(dataset_init_op)
  y_pred = []
  while True:
    try:
      y_pred.extend(sess.run(pred_tensor))
    except tf.errors.OutOfRangeError:
      break
  return y_pred


def sigmoid(x):
  """Numerically stable sigmoid function."""
  if isinstance(x, list):
    x = np.array(x)
  return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def calculate_metric(y_true, predictions, regression=True):
  """Returns RMSE for regression or ROC-AUC for classification."""
  if regression:
    return rmse(y_true, predictions)
  return sk_metrics.roc_auc_score(y_true, sigmoid(predictions))


def roc_auc_score(sess, y_true, pred_tensor, dataset_init_op):
  """Calculates the ROC AUC score (OvR)."""
  y_pred = generate_predictions(pred_tensor, dataset_init_op, sess)
  return sk_metrics.roc_auc_score(y_true, y_pred, multi_class='ovr')


def rmse_loss(sess, y_true, pred_tensor, dataset_init_op):
  """Calculates RMSE over the dataset."""
  y_pred = generate_predictions(pred_tensor, dataset_init_op, sess)
  return rmse(y_true, y_pred)


def rmse(y_true, y_pred):
  """Root mean squared error."""
  return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def grad(model, inputs, targets, loss_fn=None, train_vars=None):
  """Computes gradients of loss_fn w.r.t. train_vars."""
  if loss_fn is None:
    raise ValueError("loss_fn must be provided.")
  with tf.GradientTape() as tape:
    loss_value = loss_fn(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, train_vars)


def create_balanced_dataset(pos, neg, batch_size):
  """Creates a dataset that samples equal number of pos and neg examples."""
  pos_dataset = tf.data.Dataset.from_tensor_slices(pos).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(pos[0])))
  neg_dataset = tf.data.Dataset.from_tensor_slices(neg).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(neg[0])))
  dataset = tf.data.experimental.sample_from_datasets([pos_dataset, neg_dataset])
  return dataset.batch(batch_size)


def create_iterators(datasets, batch_size):
  """Creates a shared tf.data iterator over one or more numpy array datasets."""
  tf_datasets = [
      tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
      for data in datasets
  ]
  input_iterator = tf.data.Iterator.from_structure(
      tf_datasets[0].output_types, tf_datasets[0].output_shapes)
  init_ops = [input_iterator.make_initializer(data) for data in tf_datasets]
  x_batch = input_iterator.get_next()
  return x_batch, init_ops


def create_nam_model(x_train, dropout, feature_dropout=0.0,
                     num_basis_functions=1000, units_multiplier=2,
                     activation='exu', name_scope='model',
                     shallow=True, trainable=True, num_classes=3):
  """Instantiates a NAM model sized to the training data."""
  num_unique_vals = [len(np.unique(x_train[:, i])) for i in range(x_train.shape[1])]
  num_units = [min(num_basis_functions, i * units_multiplier) for i in num_unique_vals]
  return models.NAM(
      num_inputs=x_train.shape[-1],
      num_units=num_units,
      num_outputs=num_classes,
      dropout=np.float32(dropout),
      feature_dropout=np.float32(feature_dropout),
      activation=activation,
      shallow=shallow,
      trainable=trainable,
      name_scope=name_scope)


def build_graph(
    x_train, y_train, x_test, y_test, x_test_final, y_test_final,
    learning_rate, batch_size, output_regularization,
    dropout, decay_rate, shallow, l2_regularization=0.0,
    feature_dropout=0.0, num_basis_functions=1000, units_multiplier=2,
    activation='exu', name_scope='model',
    regression=False, use_dnn=False, trainable=True,
    class_weights=None, num_classes=3,
):
  """Constructs the TF1 computation graph with the given hyperparameters.

  Args:
    class_weights: Per-class ENS weights. If None, unweighted cross-entropy is used.

  Returns:
    graph_tensors: Dict of TF ops/tensors for training.
    eval_metric_scores: Dict of evaluation callables.
  """
  if regression:
    ds_tensors = tf.data.Dataset.from_tensor_slices((x_train, y_train)).apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=len(x_train)))
    ds_tensors = ds_tensors.batch(batch_size)
  else:
    ds_tensors = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_tensors = ds_tensors.shuffle(buffer_size=len(x_train)).repeat().batch(batch_size)

  x_batch, (train_init_op, test_init_op, final_test_init_op) = create_iterators(
      (x_train, x_test, x_test_final), batch_size)

  if use_dnn:
    nn_model = models.DNN(dropout=dropout, trainable=trainable)
  else:
    nn_model = create_nam_model(
        x_train=x_train, dropout=dropout, feature_dropout=feature_dropout,
        activation=activation, num_basis_functions=num_basis_functions,
        shallow=shallow, units_multiplier=units_multiplier,
        trainable=trainable, name_scope=name_scope, num_classes=num_classes)

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.Variable(learning_rate, trainable=False)
  lr_decay_op = learning_rate.assign(decay_rate * learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)

  predictions = nn_model(x_batch, training=False)
  train_vars = nn_model.trainable_variables

  if regression:
    y_pred = predictions
    loss_fn = penalized_mse_loss
  else:
    y_pred = tf.nn.softmax(predictions)
    loss_fn = functools.partial(penalized_cross_entropy_loss, class_weights=class_weights)

  loss_fn = functools.partial(
      loss_fn,
      output_regularization=output_regularization,
      l2_regularization=l2_regularization,
      use_dnn=use_dnn)

  iterator = ds_tensors.make_initializable_iterator()
  x1, y1 = iterator.get_next()
  loss_tensor, grads = grad(nn_model, x1, y1, loss_fn, train_vars)
  update_step = optimizer.apply_gradients(zip(grads, train_vars), global_step=global_step)
  avg_loss, avg_loss_update_op = tf.metrics.mean(loss_tensor, name='avg_train_loss')
  tf.summary.scalar('avg_train_loss', avg_loss)

  running_mean_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='avg_train_loss')
  running_vars_initializer = tf.variables_initializer(var_list=running_mean_vars)

  # Macro F1 is used as the primary selection metric for imbalanced multi-class data
  evaluation_metric = rmse_loss if regression else macro_f1_score

  test_metric = functools.partial(evaluation_metric, y_true=y_test,
                                  pred_tensor=y_pred, dataset_init_op=test_init_op)
  train_metric = functools.partial(evaluation_metric, y_true=y_train,
                                   pred_tensor=y_pred, dataset_init_op=train_init_op)
  final_test_metric = functools.partial(evaluation_metric, y_true=y_test_final,
                                        pred_tensor=y_pred, dataset_init_op=final_test_init_op)
  comp_test_metric = functools.partial(comprehensive_metrics, y_true=y_test,
                                       pred_tensor=y_pred, dataset_init_op=test_init_op)
  comp_final_test_metric = functools.partial(comprehensive_metrics, y_true=y_test_final,
                                             pred_tensor=y_pred, dataset_init_op=final_test_init_op)

  def get_predictions(sess, y_true, pred_tensor, dataset_init_op):
    y_pred_probs = generate_predictions(pred_tensor, dataset_init_op, sess)
    return y_true, np.argmax(y_pred_probs, axis=1)

  get_final_preds = functools.partial(get_predictions, y_true=y_test_final,
                                      pred_tensor=y_pred, dataset_init_op=final_test_init_op)

  summary_op = tf.summary.merge_all()
  feature_contributions = nn_model.calc_outputs(x_batch, training=False)

  graph_tensors = {
      'train_op': [update_step, avg_loss_update_op],
      'lr_decay_op': lr_decay_op,
      'summary_op': summary_op,
      'iterator_initializer': iterator.initializer,
      'running_vars_initializer': running_vars_initializer,
      'nn_model': nn_model,
      'global_step': global_step,
      'y_pred': y_pred,
      'test_init_op': test_init_op,
      'final_test_init_op': final_test_init_op,
      'feature_contributions': feature_contributions
  }

  eval_metric_scores = {
      'test': test_metric,
      'train': train_metric,
      'final_test': final_test_metric,
      'comp_test': comp_test_metric,
      'comp_final_test': comp_final_test_metric,
      'get_final_preds': get_final_preds
  }

  return graph_tensors, eval_metric_scores
