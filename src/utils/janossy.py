from tensorflow.keras.utils import Sequence
import tensorflow as tf

from itertools import permutations
import random
import numpy as np
import matplotlib.pyplot as plt


class JanossyPermutedBatchSequence(Sequence): # non viene usato nel file di kan
    """
    Keras Sequence generator for Janossy pooling.
    Generates `k` permutations per sample in each batch.
    """

    def __init__(self, X, y, batch_size=32, num_permutations=6, shuffle_batch=True):
        self.X = np.array(X)  # shape: (num_samples, sequence_length, input_dim)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.num_permutations = num_permutations
        self.shuffle_batch = shuffle_batch
        self.indexes = np.arange(len(self.X))
        self.seq_len = self.X.shape[1]

        # Precompute all permutations if sequence length is small
        if self.seq_len <= 6:
            self.all_perms = list(permutations(range(self.seq_len)))
        else:
            self.all_perms = None  # fallback to random sampling

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]

        # Output shape: (batch_size, num_permutations, sequence_length, input_dim)
        X_batch_permuted = np.zeros((len(batch_indexes), self.num_permutations, self.seq_len, self.X.shape[2]))

        for i, sample in enumerate(X_batch):
            if self.all_perms and len(self.all_perms) >= self.num_permutations:
                selected_perms = random.sample(self.all_perms, self.num_permutations)
            else:
                selected_perms = [np.random.permutation(self.seq_len) for _ in range(self.num_permutations)]

            for j, perm in enumerate(selected_perms):
                X_batch_permuted[i, j] = sample[list(perm)]

        return X_batch_permuted, y_batch

    def on_epoch_end(self):
        if self.shuffle_batch:
            np.random.shuffle(self.indexes)


def prepare_janossy_input(X, Y, num_permutations=6):
  """
  Transforms test data to Janossy-pooling-compatible input shape:
  (num_samples, num_permutations, sequence_length, input_dim)

  Args:
      X: shape (num_samples, sequence_length, input_dim)
      num_permutations: number of permutations to generate per sample
  Returns:
      X_prepared: shape (num_samples, num_permutations, sequence_length, input_dim)
  """
  # X
  X_prepared = prepare_janossy_test_input(X, num_permutations)
  # Y
  Y_reg, Y_cls = Y[:, :-1], Y[:, -1]
  Y_prepared = {"fn_0": Y_reg, "fn_1": Y_cls}
  return X_prepared, Y_prepared


def prepare_janossy_test_input(X, num_permutations=6): # nel file di kan è chiamata prepare_janossy_input 
  """
  Transforms test data to Janossy-pooling-compatible input shape:
  (num_samples, num_permutations, sequence_length, input_dim)

  Args:
      X: shape (num_samples, sequence_length, input_dim)
      num_permutations: number of permutations to generate per sample
  Returns:
      X_prepared: shape (num_samples, num_permutations, sequence_length, input_dim)
  """
  X = np.array(X)
  num_samples, seq_len, feat_dim = X.shape
  # -- try to precompute all permutations if small enough
  if seq_len <= 6:
    all_perms = list(permutations(range(seq_len)))
  else:
    all_perms = None
  X_prepared = np.zeros((num_samples, num_permutations, seq_len, feat_dim))
  for i in range(num_samples):
    if all_perms and len(all_perms) >= num_permutations:
      selected_perms = random.sample(all_perms, num_permutations)
    else:
      selected_perms = [np.random.permutation(seq_len) for _ in range(num_permutations)]
    for j, perm in enumerate(selected_perms):
      X_prepared[i, j] = X[i, list(perm)]
  return X_prepared

class UncertaintyTrackingCallback(tf.keras.callbacks.Callback):
  """
  Tracks total and per-head losses (train + validation),
  learned log_vars (uncertainty), and deltas (Huber thresholds)
  during multitask training with uncertainty weighting.
  """
  def __init__(self, plot_interval=1, save_path=None):
    super().__init__()
    self.plot_interval = plot_interval
    self.save_path = save_path
    self.history = {
      "loss": [],
      "val_loss": [],
      "per_head": {},        # training losses
      "val_per_head": {},    # validation losses
      "log_vars": {},
      "deltas": {},
      "loss_moving_averages": {}
    }

  def on_train_begin(self, logs=None):
    # Set up dicts for each variable present in the model
    if hasattr(self.model, "log_vars"):
        for k in self.model.log_vars:
            self.history["log_vars"][k] = []

    if hasattr(self.model, "deltas"):
        for k in self.model.deltas:
            self.history["deltas"][k] = []
    if hasattr(self.model, "loss_moving_averages"):
        for k in self.model.loss_moving_averages:
            self.history["loss_moving_averages"][k] = []

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    # Record total train/val losses
    self.history["loss"].append(logs.get("loss"))
    self.history["val_loss"].append(logs.get("val_loss"))
    # --- Per-head losses (train) ---
    for key, val in logs.items():
      if key.endswith("_loss") and not key.startswith("val_"):
        if key not in self.history["per_head"]:
          self.history["per_head"][key] = []
        self.history["per_head"][key].append(val)
    # --- Per-head losses (validation) ---
    for key, val in logs.items():
      if key.startswith("val_") and key.endswith("_loss"):
        clean_key = key.replace("val_", "")
        if clean_key not in self.history["val_per_head"]:
          self.history["val_per_head"][clean_key] = []
        self.history["val_per_head"][clean_key].append(val)
    # --- Record learned parameters ---
    if hasattr(self.model, "log_vars"):
        for k, v in self.model.log_vars.items():
            self.history["log_vars"][k].append(v.numpy())

    # if hasattr(self.model, "deltas"):
    #   for k, v in self.model.deltas.items():
    #     self.history["deltas"][k].append(v.numpy())
    if hasattr(self.model, "loss_moving_averages"):
        for k, v in self.model.loss_moving_averages.items():
            self.history["loss_moving_averages"][k].append(v.numpy())
    # --- Plot periodically ---
    if (epoch + 1) % self.plot_interval == 0:
        self._plot(epoch, save_all = False)

  def _plot(self, epoch, save_all = True):
    ncols = 4 if self.history["per_head"] else 3
    plt.figure(figsize=(4 * ncols, 4))
    # --- Total loss ---
    plt.subplot(1, ncols, 1)
    plt.plot(self.history["loss"], color='black', label="train")
    if any(self.history["val_loss"]):
      plt.plot(
        self.history["val_loss"], color='red', linestyle='--', label="val"
      )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    # --- Per-head losses (train + val) ---
    if self.history["per_head"]:
      plt.subplot(1, ncols, 2)
      for k, vals in self.history["per_head"].items():
        plt.plot(vals, label=f"{k} (train)")
        if k in self.history["val_per_head"]:
          plt.plot(
            self.history["val_per_head"][k], linestyle='--', label=f"{k} (val)"
          )
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.title("Per-head Losses")
      plt.legend(fontsize=8)
    # --- log_vars (uncertainty weights) ---
    if self.history["log_vars"]:
      plt.subplot(1, ncols, ncols - 1)
      for k, vals in self.history["log_vars"].items():
        plt.plot(vals, '.-', label=k)
      plt.xlabel("Epoch")
      plt.ylabel("log(σ²)")
      plt.title("Task Uncertainty (log_vars)")
      plt.legend(fontsize=8)
    # # --- deltas (Huber thresholds) ---
    # if self.history["deltas"]:
    #   plt.subplot(1, ncols, ncols)
    #   for k, vals in self.history["deltas"].items():
    #     plt.plot(vals, label=k)
    #   plt.xlabel("Epoch")
    #   plt.ylabel("δ (Huber threshold)")
    #   plt.title("Adaptive Robustness (δ per head)")
    #   plt.legend(fontsize=8)
    # --- deltas (Huber thresholds) ---
    if self.history["loss_moving_averages"]:
      plt.subplot(1, ncols, ncols)
      for k, vals in self.history["loss_moving_averages"].items():
        plt.plot(vals, label=k)
      plt.xlabel("Epoch")
      plt.ylabel("loss_moving_averages")
      plt.title("loss_moving_averages")
      plt.legend(fontsize=8)
    plt.tight_layout()
    if self.save_path:
      fname = (
        f"training_curves_epoch_{epoch+1}" if save_all 
          else "training_curves_epoch"
      )
      plt.savefig(
        f"{self.save_path}/{fname}.png", dpi=150
      )
    plt.close()
