import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow.compat.v1 as tf
from neural_additive_models import graph_builder


def save_classification_metrics(sess, y_pred_tensor, init_op, y_true, logdir, epoch, prefix='val'):
  """Computes and saves a per-class classification report as a text file."""
  y_pred_probs = graph_builder.generate_predictions(y_pred_tensor, init_op, sess)
  y_pred_class = np.argmax(y_pred_probs, axis=1)
  y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true.astype(int)

  report = classification_report(y_true_class, y_pred_class, digits=4)
  save_path = os.path.join(logdir, f'{prefix}_classification_report.txt')
  with open(save_path, 'w', encoding='utf-8') as f:
    f.write(f'=== {prefix.upper()} Classification Report (Epoch {epoch}) ===\n')
    f.write(report)
    f.write('\n' + '=' * 60 + '\n')
  tf.logging.info(f'Saved {prefix} classification report to {save_path}')


def save_confusion_matrix(sess, y_pred_tensor, init_op, y_true, logdir, epoch, prefix='val'):
  """Saves a confusion matrix plot and raw prediction arrays."""
  y_pred_probs = graph_builder.generate_predictions(y_pred_tensor, init_op, sess)
  y_pred_class = np.argmax(y_pred_probs, axis=1)
  y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true.astype(int)

  np.save(os.path.join(logdir, f'{prefix}_y_true.npy'), y_true_class)
  np.save(os.path.join(logdir, f'{prefix}_y_pred.npy'), y_pred_class)
  np.save(os.path.join(logdir, f'{prefix}_y_pred_probs.npy'), y_pred_probs)

  cm = confusion_matrix(y_true_class, y_pred_class)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.title(f'Confusion Matrix — {prefix.upper()} (Epoch {epoch})')
  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  save_path = os.path.join(logdir, f'{prefix}_confusion_matrix_best.png')
  plt.savefig(save_path)
  plt.close()
  tf.logging.info(f'Saved {prefix} confusion matrix to {save_path}')


def save_feature_importance(sess, contribution_tensor, init_op, column_names, logdir, epoch):
  """Computes mean absolute feature contributions and saves a bar chart and text file.

  One-hot encoded Roof_Type columns and _is_missing indicators are merged back
  to their original feature names before ranking.
  """
  sess.run(init_op)
  all_contribs = []
  while True:
    try:
      batch = sess.run(contribution_tensor)
      all_contribs.append(np.stack(batch, axis=1))  # (batch, num_features, num_classes)
    except tf.errors.OutOfRangeError:
      break

  all_contributions = np.concatenate(all_contribs, axis=0)
  feature_importance = np.mean(np.mean(np.abs(all_contributions), axis=0), axis=1)

  # Merge split columns back to original feature names
  combined_scores = {}
  for name, score in zip(column_names, feature_importance):
    if '_is_missing' in name:
      base = name.replace('_is_missing', '')
    elif 'Roof_Type' in name:
      base = 'Roof_Type'
    else:
      base = name
    combined_scores[base] = combined_scores.get(base, 0) + score

  final_names = list(combined_scores.keys())
  final_scores = np.array(list(combined_scores.values()))
  total = final_scores.sum()
  normalized = final_scores / total if total > 0 else final_scores
  indices = np.argsort(normalized)[::-1]

  plt.figure(figsize=(10, 8))
  sns.barplot(x=normalized[indices], y=[final_names[i] for i in indices], palette='viridis')
  plt.title(f'Feature Importance (Epoch {epoch})')
  plt.xlabel('Importance Score (sum=1.0)')
  plt.ylabel('Feature')
  plt.tight_layout()
  save_path = os.path.join(logdir, 'feature_importance.png')
  plt.savefig(save_path)
  plt.close()

  txt_path = os.path.join(logdir, 'feature_importance.txt')
  with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f'=== Feature Importance (Epoch {epoch}) ===\n')
    for i in indices:
      f.write(f'{final_names[i]:<25} : {normalized[i]:.4f}\n')
  tf.logging.info(f'Saved feature importance to {save_path}')
