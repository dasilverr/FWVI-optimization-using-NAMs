
# FWVI Optimization Using Neural Additive Models (NAMs)
This repository contains a **multi-class classification framework** built on [Neural Additive Models (NAMs)](https://arxiv.org/abs/2004.13912), developed for wildfire vulnerability index (FWVI) research. The codebase is dataset-agnostic and can be adapted to any tabular multi-class classification task.


## Overview
Standard deep learning models sacrifice interpretability for performance. NAMs address this by learning a separate neural network for each input feature and summing their outputs — making it possible to visualize exactly how each feature contributes to a prediction.

This project extends the original NAM implementation with:
- **Multi-class classification** support (3 damage levels)
- **Effective Number of Samples (ENS) class weighting** for handling imbalanced data
- **Optuna-based hyperparameter optimization (HPO)** with 5-fold cross-validation
- **Feature importance aggregation** including one-hot encoded and missing indicator variables
- **Macro F1** as the primary evaluation metric (more robust than accuracy on imbalanced data)


## Dataset
The dataset is **not included** in this repository and should be prepared separately.
Place your CSV file under `./dataset/` and update the column mapping and label encoding in `data_utils.py` to match your data. 

The loader expects:
- **Numeric features** — continuous or ordinal columns
- **Categorical features** — columns to be one-hot encoded (e.g. `Roof_Type`)
- **Target column** — an integer-encoded class label (e.g. `0`, `1`, `2`)

Missing values represented as `_none` strings are automatically handled: numeric columns receive a binary missing indicator variable, and categorical columns treat `_none` as its own category.
To use a different dataset, modify `load_wildfire_data()` in `data_utils.py` and register it in `load_dataset()` under a new `dataset_name`.

## Project Structure

```
├── neural_additive_models/
│   ├── train.py            # Entry point: HPO + final 5-fold CV training
│   ├── graph_builder.py    # TF computation graph, loss functions, evaluation metrics
│   ├── models.py           # NAM, FeatureNN, ActivationLayer, DNN definitions
│   ├── data_utils.py       # Dataset loading and K-fold split utilities
│   └── save_results.py     # Confusion matrix, classification report, feature importance
└── dataset/                # (not tracked — add your CSV here)
```

---

## Installation
### Python Version
Python **3.7 – 3.8** is recommended. TensorFlow 1.x does not support Python 3.9+.

### 1. Create a virtual environment
```bash
# Using venv
python3.8 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

```bash
# Using conda
conda create -n nam-env python=3.8
conda activate nam-env
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install tensorflow==2.11.0   # uses tensorflow.compat.v1 internally
pip install numpy pandas scikit-learn matplotlib seaborn optuna absl-py
```
> **Note:** Full TF1 (`tensorflow==1.x`) is no longer distributed on PyPI for Python 3.8+.  
> This project uses `tensorflow.compat.v1` via TF2 with `tf.disable_v2_behavior()`, which is fully supported on TF 2.x.

### Required packages summary
| Package | Recommended version |
|---|---|
| Python | 3.7 – 3.8 |
| tensorflow | ≥ 2.8, ≤ 2.13 |
| numpy | ≥ 1.21 |
| pandas | ≥ 1.3 |
| scikit-learn | ≥ 1.0 |
| matplotlib | ≥ 3.4 |
| seaborn | ≥ 0.11 |
| optuna | ≥ 3.0 |
| absl-py | ≥ 1.0 |


## Usage
### Basic training run (with HPO)
```bash
python -m neural_additive_models.train \
  --dataset_name=Wildfire \
  --logdir=logs/run_01 \
  --use_ens=True \
  --run_hpo=True \
  --n_optuna_trials=50 \
  --num_classes=3
```

### Without HPO
```bash
python -m neural_additive_models.train \
  --dataset_name=Wildfire \
  --logdir=logs/no_hpo \
  --use_ens=False \
  --run_hpo=False \
  --training_epochs=100 \
  --learning_rate=0.01
```

### ENS vs. no-ENS comparison
```bash
python -m neural_additive_models.train --use_ens=True  --logdir=logs/ens
python -m neural_additive_models.train --use_ens=False --logdir=logs/no_ens
```


## Key Flags
| Flag | Default | Description |
|---|---|---|
| `--training_epochs` | 10 | Number of training epochs |
| `--learning_rate` | 1e-2 | Initial learning rate |
| `--batch_size` | 64 | Batch size |
| `--dropout` | 0.5 | Dropout rate within FeatureNNs |
| `--feature_dropout` | 0.0 | Probability of dropping entire FeatureNNs |
| `--num_basis_functions` | 64 | Hidden units in the first FeatureNN layer |
| `--shallow` | False | Use a single hidden layer (True) or 3 layers (False) |
| `--activation` | `exu` | Hidden unit type: `exu` or `relu` |
| `--use_ens` | True | Apply ENS class weighting for imbalanced data |
| `--run_hpo` | True | Run Optuna HPO before final training |
| `--n_optuna_trials` | 50 | Number of Optuna trials |
| `--num_classes` | 3 | Number of output classes |
| `--logdir` | `logs` | Directory for saving checkpoints and results |


## Outputs
After training, each fold directory under `logdir/final_run/fold_N/` contains:
- `val_classification_report.txt` — Per-class Precision / Recall / F1
- `val_confusion_matrix_best.png` — Confusion matrix at best epoch
- `feature_importance.png` — Normalized feature importance bar chart
- `feature_importance.txt` — Ranked feature importance scores
- `val_y_true.npy`, `val_y_pred.npy`, `val_y_pred_probs.npy` — Raw prediction arrays

HPO results are saved under `logdir/optuna/`:
- `best_params.csv` — Best hyperparameters found
- `all_trials.csv` — Full Optuna trial history


## How ENS Weighting Works
Effective Number of Samples (ENS) computes per-class weights as:

$$w_c = \frac{1 - \beta}{1 - \beta^{n_c}}$$

where $n_c$ is the number of samples in class $c$ and $\beta \in \{0.9, 0.99, 0.999, 0.9999\}$. These weights are applied sample-wise to the cross-entropy loss, downweighting the majority class and improving minority class recall. The optimal $\beta$ is selected during HPO.


## Evaluation
The primary metric is **Macro F1**, which averages F1 score equally across all classes. This is preferred over accuracy for imbalanced multi-class problems, as it penalizes poor performance on minority classes.


## References
- Agarwal, R., et al. (2021). [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912). NeurIPS 2021.
- Cui, Y., et al. (2019). [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555). CVPR 2019.

