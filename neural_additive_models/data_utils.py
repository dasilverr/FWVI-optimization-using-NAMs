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

"""Data readers for regression/classification datasets."""

import gzip
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow.compat.v1 as tf

gfile = tf.gfile
DATA_PATH = './dataset'
DatasetType = Tuple[np.ndarray, np.ndarray]


def save_array_to_disk(filename, np_arr, allow_pickle=False):
  """Saves a np.ndarray to disk as a gzipped binary file."""
  with gfile.Open(filename, 'wb') as f:
    with gzip.GzipFile(fileobj=f) as outfile:
      np.save(outfile, np_arr, allow_pickle=allow_pickle)


def load_wildfire_data():
  """Loads and preprocesses the wildfire CSV dataset.

  Handles '_none' missing values by creating binary indicator variables for
  numeric columns and treating them as an independent category for categorical
  columns. Labels are ordinally encoded as 0 (no damage), 1 (partial), 2 (total).

  Returns:
    dict with keys 'problem' (str), 'X' (DataFrame), 'y' (Series).
  """
  df = pd.read_csv('./dataset/wildfire.csv', encoding='cp949')

  column_mapping = {
      '\uc9c4\uc785 \ud0c8\uc244\ub85c \uac1c\uc218': 'Num_Exits',
      '\ucd5c\uc18c \ub3c4\ub85c\ud3ed': 'Min_Road_Width',
      '\uad50\ucc28\ub85c \uac1c\uc218': 'Num_Intersections',
      '\uce68\uc5fd\uc218\ub9bc \ube44\uc728': 'Conifer_Ratio',
      '\uc0b0\ub9bc\uae30\uc900 \uc774\uaca9 \uac70\ub9ac': 'Forest_Distance',
      '\ucd5c\ub300 \uacbd\uc0ac\ub3c4': 'Max_Slope',
      '\uc8fc\uac74\ubb3c \uc9c0\ubd95\ud2b9\uc131': 'Roof_Type',
      '\uc0b0\ubd88 \uc9c4\ud654\uc6a9\uc218 \uac70\ub9ac': 'Water_Distance',
      '\uc18c\ubc29\uc11c\uc640\uc758 \uac70\ub9ac': 'FireStation_Distance',
      '\ubd88\ub09c \ud6c4 \uc0c1\ud0dc': 'Label',
  }
  df = df.rename(columns=column_mapping)

  target_col = 'Label'
  cat_cols = ['Roof_Type']
  num_cols = [c for c in df.columns if c not in [target_col] + cat_cols]

  # Create binary missing indicators for numeric '_none' values
  missing_indicator_cols = []
  for col in num_cols:
    has_none = df[col].astype(str).str.strip() == '_none'
    if has_none.any():
      ind_col = f'{col}_is_missing'
      df[ind_col] = has_none.astype(float)
      missing_indicator_cols.append(ind_col)
      print(f"[INFO] '{col}': {has_none.sum()} missing -> created '{ind_col}'")

  for col in num_cols:
    df[col] = pd.to_numeric(df[col].replace('_none', np.nan), errors='coerce')
    if df[col].isna().any():
      df[col] = df[col].fillna(0.0)

  label_map = {'no_damage': 0, 'partial_burn': 1, 'total_burn': 2}
  df[target_col] = df[target_col].map(label_map)
  df = df.dropna(subset=[target_col])
  print(f"[INFO] Label mapping: {label_map}")

  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  df_cat_encoded = pd.DataFrame(
      ohe.fit_transform(df[cat_cols]),
      columns=ohe.get_feature_names_out(cat_cols),
      index=df.index)

  x_df = pd.concat(
      [df[num_cols],
       df[missing_indicator_cols] if missing_indicator_cols else pd.DataFrame(index=df.index),
       df_cat_encoded],
      axis=1)

  print(f"[INFO] {len(missing_indicator_cols)} missing indicator(s) added: {missing_indicator_cols}")
  return {'problem': 'classification', 'X': x_df, 'y': df[target_col]}


class CustomPipeline(Pipeline):
  """sklearn Pipeline subclass with a convenience transformation method."""

  def apply_transformation(self, x):
    """Applies all transforms except the final estimator."""
    xt = x
    for _, transform in self.steps[:-1]:
      xt = transform.fit_transform(xt)
    return xt


def load_dataset(dataset_name):
  """Loads a dataset by name and returns features, labels, and column names.

  Args:
    dataset_name: One of the supported dataset identifiers (e.g. 'Wildfire').

  Returns:
    data_x: np.ndarray of shape (n_samples, n_features).
    data_y: np.ndarray of shape (n_samples,).
    column_names: List of feature names.
  """
  if dataset_name == 'Wildfire':
    dataset = load_wildfire_data()
    data_x_df = dataset['X']
    data_y_df = dataset['y']
    column_names = data_x_df.columns.tolist()
    data_x = data_x_df.values.astype('float32')
    data_y = data_y_df.values.astype('float32')
    print(f"[INFO] Wildfire loaded. Features: {len(column_names)}")
    return data_x, data_y, column_names
  raise ValueError(f'{dataset_name} not found!')


def get_train_test_fold(data_x, data_y, fold_num, num_folds,
                        stratified=True, x_test_final=None, random_state=1337):
  """Returns a single K-fold split, scaled with MinMaxScaler.

  Args:
    data_x: Feature array of shape (n_samples, n_features).
    data_y: Label array of shape (n_samples,).
    fold_num: 1-indexed fold to use as the test set.
    num_folds: Total number of folds.
    stratified: Whether to use stratified splitting.
    x_test_final: Optional holdout set to scale using the training fold's scaler.
    random_state: Random seed.

  Returns:
    (x_train, y_train), (x_val, y_val) and optionally x_test_final_scaled.
  """
  assert 0 < fold_num <= num_folds, 'Pass a valid fold number.'

  kf = StratifiedKFold if stratified else KFold
  folds = list(kf(n_splits=num_folds, shuffle=True, random_state=random_state).split(data_x, data_y))
  train_index, test_index = folds[fold_num - 1]

  x_train, x_val = data_x[train_index], data_x[test_index]
  y_train, y_val = data_y[train_index], data_y[test_index]

  scaler = MinMaxScaler()
  x_train = scaler.fit_transform(x_train).astype(np.float32)
  x_val = scaler.transform(x_val).astype(np.float32)

  if x_test_final is not None:
    return (x_train, y_train), (x_val, y_val), scaler.transform(x_test_final).astype(np.float32)
  return (x_train, y_train), (x_val, y_val)


def split_training_dataset(data_x, data_y, n_splits, stratified=True,
                            test_size=0.125, random_state=1337):
  """Yields (train, validation) splits for early stopping.

  Args:
    data_x: Feature array of shape (n_samples, n_features).
    data_y: Label array of shape (n_samples,).
    n_splits: Number of random splits to generate.
    stratified: Whether to preserve class proportions in each split.
    test_size: Fraction of data to use as validation.
    random_state: Random seed.

  Yields:
    (x_train, y_train), (x_validation, y_validation)
  """
  splitter = StratifiedShuffleSplit if stratified else ShuffleSplit
  split_gen = splitter(n_splits=n_splits, test_size=test_size,
                       random_state=random_state).split(data_x, data_y)
  for train_idx, val_idx in split_gen:
    x_train, x_val = data_x[train_idx], data_x[val_idx]
    y_train, y_val = data_y[train_idx], data_y[val_idx]
    assert x_train.shape[0] == y_train.shape[0]
    yield (x_train, y_train), (x_val, y_val)
