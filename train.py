import operator
import os
import warnings

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

from neural_additive_models import save_results
from neural_additive_models import data_utils
from neural_additive_models import graph_builder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)

_N_FOLDS = 5
_N_MODELS = 1
_N_TRAINIG_EPOCH = 10

gfile = tf.io.gfile
FLAGS = flags.FLAGS

flags.DEFINE_integer('training_epochs', _N_TRAINIG_EPOCH, 'Number of training epochs.')
flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate.')
flags.DEFINE_float('output_regularization', 0.0, 'Feature output regularization coefficient.')
flags.DEFINE_float('l2_regularization', 0.0, 'L2 weight decay coefficient.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_string('logdir', 'logs', 'Directory for saving summaries and checkpoints.')
flags.DEFINE_string('dataset_name', 'Wildfire', 'Name of the dataset to load.')
flags.DEFINE_float('decay_rate', 0.995, 'Learning rate decay rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate within FeatureNNs.')
flags.DEFINE_integer('data_split', 1, 'Dataset split index (1 to num_splits).')
flags.DEFINE_integer('tf_seed', 42, 'Random seed.')
flags.DEFINE_float('feature_dropout', 0.0, 'Probability of dropping entire FeatureNNs.')
flags.DEFINE_integer('num_basis_functions', 64, 'Hidden units in the first FeatureNN layer.')
flags.DEFINE_integer('units_multiplier', 2, 'Basis functions multiplier for categorical features.')
flags.DEFINE_integer('n_models', _N_MODELS, 'Number of models to train.')
flags.DEFINE_string('activation', 'exu', 'Hidden unit activation: "exu" or "relu".')
flags.DEFINE_boolean('regression', False, 'True for regression, False for classification.')
flags.DEFINE_integer('early_stopping_epochs', 20, 'Early stopping patience in epochs.')
flags.DEFINE_boolean('debug', False, 'Enable debug logging.')
flags.DEFINE_boolean('shallow', False, 'Use shallow (1-layer) FeatureNNs.')
flags.DEFINE_boolean('use_dnn', False, 'Use DNN baseline instead of NAM.')
flags.DEFINE_integer('save_checkpoint_every_n_epochs', 10, 'Checkpoint save interval.')
flags.DEFINE_integer('max_checkpoints_to_keep', 1, 'Maximum number of checkpoints to keep.')
flags.DEFINE_boolean('cross_val', False, 'Enable cross-validation mode.')
flags.DEFINE_integer('num_classes', 3, 'Number of output classes.')
flags.DEFINE_integer('n_optuna_trials', 50, 'Number of Optuna HPO trials.')
flags.DEFINE_boolean('run_hpo', True, 'Run Optuna HPO before final training.')
flags.DEFINE_boolean('use_ens', True, 'Apply ENS class weighting to the loss. True for imbalanced correction, False for baseline.')

GraphOpsAndTensors = graph_builder.GraphOpsAndTensors
EvaluationMetric = graph_builder.EvaluationMetric


@flags.multi_flags_validator(
    ['data_split', 'cross_val'],
    message='data_split should not be used together with cross_val')
def data_split_with_cross_validation(flags_dict):
  return (flags_dict['data_split'] == 1) or (not flags_dict['cross_val'])


def _get_train_and_lr_decay_ops(graph_tensors_and_ops, early_stopping):
  """Returns active training ops and lr decay ops (excluding early-stopped models)."""
  train_ops = [g['train_op'] for n, g in enumerate(graph_tensors_and_ops) if not early_stopping[n]]
  lr_decay_ops = [g['lr_decay_op'] for n, g in enumerate(graph_tensors_and_ops) if not early_stopping[n]]
  return train_ops, lr_decay_ops


def _update_latest_checkpoint(checkpoint_dir, best_checkpoint_dir):
  """Copies the latest checkpoint from checkpoint_dir to best_checkpoint_dir."""
  for filename in gfile.glob(os.path.join(best_checkpoint_dir, 'model.*')):
    gfile.remove(filename)
  for name in gfile.glob(os.path.join(checkpoint_dir, 'model.*')):
    gfile.copy(name, os.path.join(best_checkpoint_dir, os.path.basename(name)), overwrite=True)
  chkpt_src = os.path.join(checkpoint_dir, 'checkpoint')
  chkpt_dst = os.path.join(best_checkpoint_dir, 'checkpoint')
  if gfile.exists(chkpt_src):
    gfile.copy(chkpt_src, chkpt_dst, overwrite=True)


def _create_computation_graph(x_train, y_train, x_validation, y_validation,
                               x_test_final, y_test_final, batch_size, class_weights):
  """Builds the computation graph for all n_models.

  Args:
    class_weights: Per-class ENS weights. If None, unweighted loss is applied.
  """
  graph_tensors_and_ops, metric_scores = [], []
  for n in range(FLAGS.n_models):
    g, m = graph_builder.build_graph(
        x_train=x_train, y_train=y_train,
        x_test=x_validation, y_test=y_validation,
        x_test_final=x_test_final, y_test_final=y_test_final,
        activation=FLAGS.activation,
        learning_rate=FLAGS.learning_rate,
        batch_size=batch_size,
        shallow=FLAGS.shallow,
        output_regularization=FLAGS.output_regularization,
        l2_regularization=FLAGS.l2_regularization,
        dropout=FLAGS.dropout,
        num_basis_functions=FLAGS.num_basis_functions,
        units_multiplier=FLAGS.units_multiplier,
        decay_rate=FLAGS.decay_rate,
        feature_dropout=FLAGS.feature_dropout,
        regression=FLAGS.regression,
        use_dnn=FLAGS.use_dnn,
        trainable=True,
        class_weights=class_weights,
        num_classes=FLAGS.num_classes,
        name_scope=f'model_{n}')
    graph_tensors_and_ops.append(g)
    metric_scores.append(m)
  return graph_tensors_and_ops, metric_scores


def _create_graph_saver(graph_tensors_and_ops, logdir, num_steps_per_epoch):
  """Creates checkpoint saver hooks and directories for each model."""
  saver_hooks, model_dirs, best_checkpoint_dirs = [], [], []
  save_steps = num_steps_per_epoch * FLAGS.save_checkpoint_every_n_epochs * FLAGS.n_models
  for n in range(FLAGS.n_models):
    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            var_list=graph_tensors_and_ops[n]['nn_model'].trainable_variables,
            save_relative_paths=True,
            max_to_keep=FLAGS.max_checkpoints_to_keep))
    model_dirs.append(os.path.join(logdir, f'model_{n}'))
    best_checkpoint_dirs.append(os.path.join(model_dirs[-1], 'best_checkpoint'))
    gfile.makedirs(best_checkpoint_dirs[-1])
    saver_hooks.append(tf.train.CheckpointSaverHook(
        checkpoint_dir=model_dirs[-1], save_steps=save_steps, scaffold=scaffold))
  return saver_hooks, model_dirs, best_checkpoint_dirs


def _update_metrics_and_checkpoints(sess, epoch, metric_scores, curr_best_epoch,
                                     best_validation_metric, best_train_metric,
                                     model_dir, best_checkpoint_dir,
                                     metric_name='RMSE', optuna_mode=False):
  """Evaluates current epoch and updates best checkpoint if improved."""
  compare_metric = operator.lt if FLAGS.regression else operator.gt
  validation_metric = metric_scores['test'](sess)
  if FLAGS.debug:
    train_metric = metric_scores['train'](sess)
    tf.logging.info(f'Epoch {epoch} | Train {metric_name}: {train_metric:.4f} | '
                    f'Val {metric_name}: {validation_metric:.4f}')
  else:
    train_metric = best_train_metric

  if compare_metric(validation_metric, best_validation_metric):
    best_validation_metric = validation_metric
    curr_best_epoch = epoch
    _update_latest_checkpoint(model_dir, best_checkpoint_dir)

  return curr_best_epoch, best_validation_metric, train_metric


def training(x_train, y_train, x_validation, y_validation,
             x_test_final, y_test_final,
             logdir, column_names, optuna_mode=False, beta=0.999):
  """Runs the full training loop for a single fold.

  Args:
    beta: ENS smoothing parameter (used only when --use_ens=True).
    optuna_mode: If True, skips saving artifacts and returns early.

  Returns:
    (best_train_metric, best_val_metric, final_test_metric)
  """
  # Compute ENS class weights if enabled
  if FLAGS.use_ens:
    class_counts = np.bincount(y_train.astype(int), minlength=FLAGS.num_classes)
    ens_weights = (1 - beta) / (1 - np.power(beta, class_counts.astype(float) + 1e-8))
    class_weights = ens_weights / ens_weights.sum() * FLAGS.num_classes
    tf.logging.info(f'ENS class weights (beta={beta}): {class_weights}')
  else:
    class_weights = None

  batch_size = min(FLAGS.batch_size, len(x_train))
  num_steps_per_epoch = int(np.ceil(len(x_train) / batch_size))

  graph_tensors_and_ops, metric_scores = _create_computation_graph(
      x_train, y_train, x_validation, y_validation,
      x_test_final, y_test_final, batch_size, class_weights)

  early_stopping = [False] * FLAGS.n_models
  curr_best_epoch = [0] * FLAGS.n_models
  best_validation_metric = [(-np.inf if not FLAGS.regression else np.inf)] * FLAGS.n_models
  best_train_metric = [0.0] * FLAGS.n_models

  metric_name = 'RMSE' if FLAGS.regression else 'Macro-F1'
  saver_hooks, model_dirs, best_checkpoint_dirs = _create_graph_saver(
      graph_tensors_and_ops, logdir, num_steps_per_epoch)

  summary_writer = tf.summary.FileWriter(logdir)

  with tf.train.MonitoredTrainingSession(hooks=saver_hooks) as sess:
    for epoch in range(1, FLAGS.training_epochs + 1):
      train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
          graph_tensors_and_ops, early_stopping)
      if not train_ops:
        break

      for step in range(num_steps_per_epoch * FLAGS.n_models):
        model_idx = step % FLAGS.n_models
        if not early_stopping[model_idx]:
          sess.run(graph_tensors_and_ops[model_idx]['iterator_initializer'])
          sess.run(graph_tensors_and_ops[model_idx]['running_vars_initializer'])
          sess.run(train_ops[model_idx % len(train_ops)])

      sess.run([op for ops in lr_decay_ops for op in (ops if isinstance(ops, list) else [ops])])

      for n in range(FLAGS.n_models):
        if early_stopping[n]:
          continue
        curr_best_epoch[n], best_validation_metric[n], best_train_metric[n] = \
            _update_metrics_and_checkpoints(
                sess, epoch, metric_scores[n],
                curr_best_epoch[n], best_validation_metric[n], best_train_metric[n],
                model_dirs[n], best_checkpoint_dirs[n], metric_name, optuna_mode)

        if epoch - curr_best_epoch[n] > FLAGS.early_stopping_epochs:
          early_stopping[n] = True
          tf.logging.info(f'Model {n}: early stopping at epoch {epoch}')

      if all(early_stopping):
        break

    if not optuna_mode:
      best_n = int(np.argmax(best_validation_metric))
      g = graph_tensors_and_ops[best_n]
      m = metric_scores[best_n]

      save_results.save_classification_metrics(
          sess, g['y_pred'], g['test_init_op'], y_validation, logdir,
          epoch=curr_best_epoch[best_n], prefix='val')
      save_results.save_confusion_matrix(
          sess, g['y_pred'], g['test_init_op'], y_validation, logdir,
          epoch=curr_best_epoch[best_n], prefix='val')
      save_results.save_feature_importance(
          sess, g['feature_contributions'], g['test_init_op'],
          column_names, logdir, epoch=curr_best_epoch[best_n])

      comp = m['comp_final_test'](sess)
      tf.logging.info(f'Final Test — Acc: {comp[0]:.4f} | Prec: {comp[1]:.4f} | '
                      f'Rec: {comp[2]:.4f} | F1: {comp[3]:.4f}')

    final_test_score = np.mean([m['final_test'](sess) for m in metric_scores])

  return np.mean(best_train_metric), np.mean(best_validation_metric), final_test_score


def run_cross_validation(data_x, data_y, column_names, logdir, beta=0.999):
  """Runs 5-fold cross-validation and saves a summary CSV."""
  fold_val_metric, final_test_metric, best_train_metric = [], [], []
  final_acc, final_prec, final_rec, final_f1 = [], [], [], []

  csv_path = os.path.join(logdir, 'fold_results.csv')
  rows = []

  for current_fold in range(1, _N_FOLDS + 1):
    tf.logging.info(f'Starting fold {current_fold} / {_N_FOLDS}')
    (x_train, y_train), (x_val, y_val), x_test_scaled = create_test_train_fold(
        data_x, data_y, current_fold, x_test_final=None)

    fold_logdir = os.path.join(logdir, f'fold_{current_fold}')
    train_score, val_score, test_score = training(
        x_train, y_train, x_val, y_val, x_val, y_val,
        logdir=fold_logdir, column_names=column_names, beta=beta)

    best_train_metric.append(train_score)
    fold_val_metric.append(val_score)
    final_test_metric.append(test_score)

    tf.logging.info(f'Fold {current_fold} — Train: {train_score:.4f} | '
                    f'Val: {val_score:.4f} | Test: {test_score:.4f}')
    tf.reset_default_graph()

  rows.append({
      'Fold': 'Average',
      'Train_Macro_F1': np.mean(best_train_metric),
      'Val_Macro_F1': np.mean(fold_val_metric),
      'Final_Test_Macro_F1': np.mean(final_test_metric),
  })
  pd.DataFrame(rows).to_csv(csv_path, index=False)
  tf.logging.info(f'Saved fold results to {csv_path}')

  return np.mean(best_train_metric), np.mean(fold_val_metric), np.mean(final_test_metric)


def create_test_train_fold(data_x, data_y, fold_num, x_test_final=None):
  """Wraps data_utils.get_train_test_fold with the global fold settings."""
  return data_utils.get_train_test_fold(
      data_x, data_y,
      fold_num=fold_num,
      num_folds=_N_FOLDS,
      stratified=not FLAGS.regression,
      x_test_final=x_test_final)


def _apply_best_params(best_params):
  """Applies Optuna best parameters back to FLAGS."""
  FLAGS.learning_rate         = best_params['learning_rate']
  FLAGS.batch_size            = best_params['batch_size']
  FLAGS.dropout               = best_params['dropout']
  FLAGS.num_basis_functions   = best_params['num_basis_functions']
  FLAGS.training_epochs       = best_params['training_epochs']
  FLAGS.activation            = best_params['activation']
  FLAGS.shallow               = best_params['shallow']
  FLAGS.decay_rate            = best_params.get('decay_rate', FLAGS.decay_rate)
  FLAGS.early_stopping_epochs = best_params.get('early_stopping_epochs', FLAGS.early_stopping_epochs)


def objective(trial, x_train_val, y_train_val, column_names):
  """Optuna objective: 5-fold CV on the train+val set (final test set never touched)."""
  FLAGS.learning_rate         = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
  FLAGS.batch_size            = trial.suggest_categorical('batch_size', [32, 64, 128])
  FLAGS.dropout               = trial.suggest_float('dropout', 0.1, 0.6)
  FLAGS.num_basis_functions   = trial.suggest_categorical('num_basis_functions', [64, 128, 256])
  FLAGS.training_epochs       = trial.suggest_int('training_epochs', 50, 300, step=50)
  FLAGS.activation            = trial.suggest_categorical('activation', ['exu', 'relu'])
  FLAGS.shallow               = trial.suggest_categorical('shallow', [True, False])
  FLAGS.decay_rate            = trial.suggest_float('decay_rate', 0.95, 0.999)
  FLAGS.early_stopping_epochs = trial.suggest_int('early_stopping_epochs', 20, 100)

  beta = trial.suggest_categorical('beta', [0.9, 0.99, 0.999, 0.9999]) \
      if FLAGS.use_ens else 0.999

  fold_val_scores = []
  for current_fold in range(1, _N_FOLDS + 1):
    (x_train, y_train), (x_val, y_val) = create_test_train_fold(
        x_train_val, y_train_val, current_fold)
    trial_logdir = os.path.join(FLAGS.logdir, 'optuna',
                                f'trial_{trial.number}', f'fold_{current_fold}')
    _, val_score, _ = training(
        x_train, y_train, x_val, y_val, x_val, y_val,
        logdir=trial_logdir, column_names=column_names,
        optuna_mode=True, beta=beta)
    fold_val_scores.append(val_score)
    tf.reset_default_graph()

  return np.mean(fold_val_scores)



def main(argv):
  del argv
  tf.logging.set_verbosity(tf.logging.INFO)
  np.random.seed(FLAGS.tf_seed)

  ens_tag = 'ENS' if FLAGS.use_ens else 'NoENS'
  print(f"\n>>> Mode: {ens_tag}  (--use_ens={FLAGS.use_ens})")
  print(f"    Log directory: {FLAGS.logdir}")

  print(f"\n>>> Loading dataset: {FLAGS.dataset_name} ...")
  data_x, data_y, column_names = data_utils.load_dataset(FLAGS.dataset_name)

  x_train_val, x_test_final, y_train_val, y_test_final = train_test_split(
      data_x, data_y, test_size=0.2, stratify=data_y, random_state=42)

  print(f"    Total: {len(data_x)} | Train+Val (80%): {len(x_train_val)} | "
        f"Final Test (20%): {len(x_test_final)}")

  if FLAGS.run_hpo:
    print(f"\n>>> [Step 1] Optuna HPO ({ens_tag}, n_trials={FLAGS.n_optuna_trials}) ...")
    study = optuna.create_study(
        study_name=f'NAM_Wildfire_HPO_{ens_tag}',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
    study.optimize(
        lambda trial: objective(trial, x_train_val, y_train_val, column_names),
        n_trials=FLAGS.n_optuna_trials,
        show_progress_bar=True)

    best_params    = study.best_params
    best_val_score = study.best_value
    print(f"\n>>> HPO complete. Best Val Macro F1: {best_val_score:.4f}")
    print(f"    Best Params: {best_params}")

    hpo_logdir = os.path.join(FLAGS.logdir, 'optuna')
    os.makedirs(hpo_logdir, exist_ok=True)
    pd.DataFrame([best_params]).to_csv(os.path.join(hpo_logdir, 'best_params.csv'), index=False)
    study.trials_dataframe().to_csv(os.path.join(hpo_logdir, 'all_trials.csv'), index=False)

    _apply_best_params(best_params)
    beta = best_params.get('beta', 0.999)
  else:
    print("\n>>> Skipping HPO. Running final training with current FLAGS.")
    beta = 0.999

  print(f"\n>>> [Step 2] Final 5-Fold CV ({ens_tag}) ...")
  final_logroot = os.path.join(FLAGS.logdir, 'final_run')
  tf.io.gfile.makedirs(final_logroot)

  all_fold_results = []
  for current_fold in range(1, _N_FOLDS + 1):
    print(f"\n    [Final Run] Fold {current_fold} / {_N_FOLDS}")
    (x_train, y_train), (x_val, y_val), x_test_scaled = create_test_train_fold(
        x_train_val, y_train_val, current_fold, x_test_final)
    curr_logdir = os.path.join(final_logroot, f'fold_{current_fold}')
    train_score, val_score, test_score = training(
        x_train, y_train, x_val, y_val, x_test_scaled, y_test_final,
        logdir=curr_logdir, column_names=column_names,
        optuna_mode=False, beta=beta)
    all_fold_results.append((current_fold, train_score, val_score, test_score))
    tf.reset_default_graph()

  print(f"\n>>> [{ens_tag}] Final Results Summary (Macro F1)")
  print(f"    {'Fold':<8} {'Train':>10} {'Val':>10} {'Test':>10}")
  for fold_num, tr, va, te in all_fold_results:
    print(f"    {fold_num:<8} {tr:>10.4f} {va:>10.4f} {te:>10.4f}")
  print(f"    {'Average':<8} "
        f"{np.mean([r[1] for r in all_fold_results]):>10.4f} "
        f"{np.mean([r[2] for r in all_fold_results]):>10.4f} "
        f"{np.mean([r[3] for r in all_fold_results]):>10.4f}")


if __name__ == '__main__':
  app.run(main)
