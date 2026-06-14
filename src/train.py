from utils.model_creators import build_janossy_rnn_model
from gossiplearning.config import Config

from utils.janossy import prepare_janossy_input, prepare_janossy_test_input, UncertaintyTrackingCallback
from utils.data import load_dataset, load_tasks

from statsmodels.graphics.mosaicplot import mosaic
from keras.api.callbacks import ModelCheckpoint
from tensorflow.python.keras import Model
from matplotlib import colors as mcolors
from keras.api.models import load_model
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from collections import deque
from sklearn import metrics
from typing import Tuple
import tensorflow as tf
import pandas as pd
import numpy as np
import functools
import itertools
import argparse
import random
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Train models for Azure function traces"
  )
  parser.add_argument(
    "-f", "--base_folder", 
    help="Paths to the base folder", 
    type=str,
    required=True
  )
  parser.add_argument(
    "-m", "--modes", 
    help="Training mode(s)", 
    type = str,
    nargs = "+",
    choices = ["centralized", "local"],
    required=True
  )
  parser.add_argument(
    "-c", "--config_file", 
    help="Config file", 
    type=str,
    default="config.json"
  )
  parser.add_argument(
    "--seed", 
    help="Seed for random number generation", 
    type=int,
    default=4850
  )
  parser.add_argument(
    "--networks", 
    help="Network indices", 
    nargs="+",
    default=[0]
  )
  parser.add_argument(
    "--n_sim_per_network", 
    help="Number of simulations to run for each network", 
    type=int,
    default=1
  )
  parser.add_argument(
    "--plot_single_history", 
    help="True to plot the history of each model", 
    default=False,
    action="store_true"
  )
  args, _ = parser.parse_known_args()
  return args


def already_trained(
    output_folder: str, model_keyword: str
  ) -> Tuple[bool, str, str]:
  history_file = os.path.join(output_folder, f"{model_keyword}_history.csv")
  model_file = os.path.join(output_folder, f"{model_keyword}.keras")
  at = os.path.exists(history_file) and os.path.exists(model_file)
  return at, history_file, model_file


def compute_and_plot_predictions(
    X_Y_data: dict, 
    model,
    output_folder: str,
    node: str,
    targets: list
  ) -> dict:
  # compute predictions
  predictions = {}
  for data_key, (X_data, _) in X_Y_data.items():
    X_test_prepared = prepare_janossy_test_input(X_data, num_permutations = 6)
    predictions[data_key] = model.predict(X_test_prepared)
  # prepare dictionary to store confusion matrices
  cmatrices = {}
  # plot
  fontsize = 14
  nrows = len(X_Y_data)
  ncols = next(iter(X_Y_data.values()))[1].shape[1] - 1
  colors = list(mcolors.TABLEAU_COLORS.values())
  _, axs = plt.subplots(
    nrows = nrows, 
    ncols = ncols,
    figsize = (28, 3 * nrows),
    gridspec_kw = {"wspace": 0.1}
  )
  idx = 0
  for data_key in predictions:
    ax = axs if nrows == 1 else axs[idx,:]
    Y_pred = predictions[data_key][0]
    Y_pred_cls = np.array(
      [0 if p < 0.5 else 1 for p in predictions[data_key][1]], 
      dtype = int
    )
    Y_data = np.array(X_Y_data[data_key][1])[:,:-1]
    Y_data_cls = np.array(X_Y_data[data_key][1])[:,-1]
    # loop over predicted values
    nr, nc = Y_data.shape
    ymin = [2] * nc
    ymax = [1] * nc
    for i in range(nr):
      for j in range(nc):
        ax[j].plot(
          Y_data[i,j],
          Y_pred[i,j],
          color = colors[j % len(colors)],
          marker = ".",
          label = targets[j] if i == 0 else None
        )
        ymin[j] = min(ymin[j], Y_data[i,j])
        ymax[j] = max(ymax[j], Y_data[i,j])
    # add reference lines
    for j in range(nc):
      ax[j].plot(
        np.linspace(ymin[j], ymax[j], 1000),
        np.linspace(ymin[j], ymax[j], 1000),
        "k--"
      )
      # axis properties
      ax[j].set_ylabel(data_key, fontsize = fontsize)
      ax[j].legend(loc = "upper left", fontsize = fontsize)
      ax[j].grid(True)
    # compute confusion matrix classification results
    cmatrix = metrics.confusion_matrix(Y_data_cls, Y_pred_cls)
    cmatrices[data_key] = cmatrix
    idx += 1
  # save figure
  if output_folder is not None:
    plt.savefig(
      os.path.join(output_folder, f"predictions_{node}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()
  # plot confusion matrices
  ncols = len(cmatrices)
  _, axs = plt.subplots(
    nrows = 2, 
    ncols = ncols, 
    figsize=(13 * ncols, 20)
  )
  idx = 0
  for data_key, cmatrix in cmatrices.items():
    ax = axs if ncols == 1 else axs[:,idx]
    cm_display = metrics.ConfusionMatrixDisplay(
      confusion_matrix = cmatrix, display_labels = [0, 1]
    )
    cm_display.plot(ax = ax[0], text_kw = {"fontsize": fontsize})
    #
    nclass_classification_mosaic_plot(
      cmatrix.shape[0], 
      cmatrix,
      ax = ax[1]
      # None if output_folder is None else os.path.join(
      #   output_folder, f"mosaic_plot_{node}_{data_key}.png"
      # )
    )
    idx += 1
  plt.savefig(
    os.path.join(
      output_folder, f"cmatrix_{node}.png"
    ),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  return predictions


def nclass_classification_mosaic_plot(
    n_classes: int, results: list, ax = None, plot_path = None
  ):
  """
  build a mosaic plot from the results of a classification
  
  parameters:
  n_classes: number of classes
  results: results of the prediction in form of an array of arrays
  
  In case of 3 classes the prdiction could look like
  [[10, 2, 4],
    [1, 12, 3],
    [2, 2, 9]
  ]
  where there is one array for each class and each array holds the
  predictions for each class [class 1, class 2, class 3].
  
  This is just a prototype including colors for 6 classes.
  """
  # reshape data
  class_lists = [range(n_classes)] * 2
  mosaic_tuples = tuple(itertools.product(*class_lists))
  res_list = results[0].tolist()
  for i, l in enumerate(results):
    if i == 0:
      pass
    else:
      tmp = deque(l)
      tmp.rotate(-i)
      res_list.extend(tmp)
  data = {t: res_list[i] for i,t in enumerate(mosaic_tuples)}
  # define color palette
  font_color = '#2c3e50'
  pallet = list(mcolors.TABLEAU_COLORS.values())
  # [
  #   '#6a89cc', 
  #   '#4a69bd', 
  #   '#1e3799', 
  #   '#0c2461',
  #   '#82ccdd',
  #   '#60a3bc',
  # ]
  colors = deque(pallet[:n_classes])
  all_colors = []
  for i in range(n_classes):
    if i > 0:
      colors.rotate(-1)
    all_colors.extend(colors)
  props = {
    (str(a), str(b)): {
      'color': all_colors[i]
    } for i,(a, b) in enumerate(mosaic_tuples)
  }
  # plot
  standalone = False
  if ax is None:
    _, ax = plt.subplots(figsize=(11, 10))
    plt.rcParams.update({'font.size': 16})
    standalone = True
  labelizer = lambda k: ''
  p = mosaic(data, labelizer=labelizer, properties=props, ax=ax)
  axis_label_font_dict = {
    'fontsize': 16,
    'color' : font_color,
  }
  ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
  ax.axes.yaxis.set_ticks([])
  ax.tick_params(axis='x', which='major', labelsize=14)
  ax.set_xlabel('Observed Class', fontdict=axis_label_font_dict, labelpad=10)
  ax.set_ylabel('Predicted Class', fontdict=axis_label_font_dict, labelpad=35)
  legend_elements = [
    Patch(
      facecolor = all_colors[i], 
      label = 'Class {}'.format(i)
    ) for i in range(n_classes)
  ]
  ax.legend(
    handles = legend_elements, 
    bbox_to_anchor = (1,1.018), 
    fontsize = 16
  )
  if standalone:
    # set title
    title_font_dict = {
      'fontsize': 20 if ax is None else 10,
      'color' : font_color,
    }
    ax.set_title('Classification Report', fontdict=title_font_dict, pad=25)
    # save or show
    plt.tight_layout()
    if plot_path is not None:
      plt.savefig(plot_path, dpi = 300, format = "png", bbox_inches = "tight")
      plt.close()
    else:
      plt.show()


def plot_history(history_df: pd.DataFrame, output_folder: str, node: str):
  # plot MAE history
  history_df[["fn_0_masked_mae", "val_fn_0_masked_mae"]].plot(grid = True)
  plt.savefig(
    os.path.join(output_folder, f"mae_{node}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  # plot MSE history
  history_df[["fn_0_masked_mse", "val_fn_0_masked_mse"]].plot(grid = True)
  plt.savefig(
    os.path.join(output_folder, f"mse_{node}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  # plot MAPE history
  history_df[["fn_0_masked_mape", "val_fn_0_masked_mape"]].plot(grid = True)
  plt.savefig(
    os.path.join(output_folder, f"mape_{node}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  # plot accuracy history
  history_df[["fn_1_accuracy", "val_fn_1_accuracy"]].plot(grid = True)
  plt.savefig(
    os.path.join(output_folder, f"accuracy_{node}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


def plot_multinode_history(
    histories: pd.DataFrame, output_folder: str, keyword: str
  ):
  colors = list(mcolors.TABLEAU_COLORS.values())
  centralized_exists = "centralized" in histories["node"].unique()
  _, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (16,16), sharex = True)
  for node, history_df in histories.groupby("node"):
    color = "k" if node == "centralized" else (
      colors[0] if centralized_exists else colors[int(node) % len(colors)]
    )
    linewidth = 2 if node == "centralized" else 1
    linestyle = "solid" if node == "centralized" else "dashed"
    label = node
    # -- MAE
    history_df.plot(
      x = "iter",
      y = "fn_0_masked_mae", 
      ax = axs[0,0],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    history_df.plot(
      x = "iter",
      y = "val_fn_0_masked_mae", 
      ax = axs[0,1],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    # # -- MSE
    history_df.plot(
      x = "iter",
      y = "fn_0_masked_mse", 
      ax = axs[1,0],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    history_df.plot(
      x = "iter",
      y = "val_fn_0_masked_mse", 
      ax = axs[1,1],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    # # -- MAPE
    history_df.plot(
      x = "iter",
      y = "fn_0_masked_mape", 
      ax = axs[2,0],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    history_df.plot(
      x = "iter",
      y = "val_fn_0_masked_mape", 
      ax = axs[2,1],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    # -- accuracy
    history_df.plot(
      x = "iter",
      y = "fn_1_accuracy", 
      ax = axs[3,0],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    history_df.plot(
      x = "iter",
      y = "val_fn_1_accuracy", 
      ax = axs[3,1],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
  # plot average over local models
  local = histories[histories["node"]!="centralized"].groupby("iter").mean(
    numeric_only = True
  )
  color = colors[0] if "centralized" in histories["node"].unique() else "r"
  linewidth = 2
  linestyle = "solid"
  # -- MAE
  local.plot(
    y = "fn_0_masked_mae", 
    ax = axs[0,0],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  local.plot(
    y = "val_fn_0_masked_mae", 
    ax = axs[0,1],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  # # -- MSE
  local.plot(
    y = "fn_0_masked_mse", 
    ax = axs[1,0],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  local.plot(
    y = "val_fn_0_masked_mse", 
    ax = axs[1,1],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  # # -- MAPE
  local.plot(
    y = "fn_0_masked_mape", 
    ax = axs[2,0],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  local.plot(
    y = "val_fn_0_masked_mape", 
    ax = axs[2,1],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  # -- accuracy
  local.plot(
    y = "fn_1_accuracy", 
    ax = axs[3,0],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  local.plot(
    y = "val_fn_1_accuracy", 
    ax = axs[3,1],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  # axis properties
  axs[0,0].set_ylabel("MAE", fontsize = 14)
  axs[1,0].set_ylabel("MSE", fontsize = 14)
  axs[2,0].set_ylabel("MAPE", fontsize = 14)
  axs[3,0].set_ylabel("Accuracy", fontsize = 14)
  axs[-1,0].set_xlabel("Epoch", fontsize = 14)
  axs[-1,1].set_xlabel("Epoch", fontsize = 14)
  axs[0,0].set_title("Training History", fontsize = 14)
  axs[0,1].set_title("Validation History", fontsize = 14)
  axs[1,1].legend(
    fontsize = 14, loc = "center left", bbox_to_anchor = (1, 0)
  )
  axs[0,0].get_legend().remove()
  axs[0,1].get_legend().remove()
  axs[1,0].get_legend().remove()
  axs[2,0].get_legend().remove()
  axs[2,1].get_legend().remove()
  axs[3,0].get_legend().remove()
  axs[3,1].get_legend().remove()
  plt.savefig(
    os.path.join(output_folder, f"{keyword}_history.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


def prepare_training(config_file: str) -> Tuple[Config, functools.partial]:
  # load and validate configuration
  config = None
  with open(config_file, "r") as f:
    config = Config.model_validate(json.load(f))
  # define model creator
  model_creator = functools.partial(
      build_janossy_rnn_model,
      config = config,
  )
  print(model_creator().summary())
  return config, model_creator


def train_one_model(
    config: Config, 
    model_creator: functools.partial, 
    dataset: list, 
    checkpoint_path: str
  ) -> Tuple[Model, dict]:
  # prepare model and output files
  model = model_creator()
  model_checkpoint = ModelCheckpoint(
    filepath = checkpoint_path,
    save_best_only = True,
    monitor = "val_loss",
    mode = "min",
  )

  basedir, fname = os.path.split(checkpoint_path)
  log_dir = os.path.join(
      basedir, "logs", str(fname.replace(".keras", "").split("_")[0])
  )
  os.makedirs(log_dir, exist_ok=True)

  # extract data
  train_data, validation_data, _ = dataset
  X_train = train_data[0]
  Y_train = np.array(train_data[1])
  X_val = validation_data[0]
  Y_val = np.array(validation_data[1])
  #
  X_train_prepared, Y_train_prepared = prepare_janossy_input(
    X_train, Y_train, num_permutations = 6
  )
  X_val_prepared, Y_val_prepared = prepare_janossy_input(
    X_val, Y_val, num_permutations = 6
  )
  
  # callbacks
  tracker = UncertaintyTrackingCallback(
      plot_interval=5,
      save_path=log_dir
  )

  # train
  history = model.fit(
    X_train_prepared,
    Y_train_prepared,
    validation_data = (
      X_val_prepared, Y_val_prepared
    ),
    validation_batch_size = config.training.batch_size,
    verbose = 1,
    callbacks = [
      # early_stopping,
      model_checkpoint,
      tracker
    ],
    epochs = config.training.epochs_per_update * config.training.fixed_updates,
    batch_size = config.training.batch_size,
    shuffle = config.training.shuffle_batch,
    # use_multiprocessing = False,
  ).history
  return model, history


def train_local_models(
    config: Config, 
    model_creator: functools.partial, 
    nodes_dataset: dict, 
    seed: int,
    output_folder: str,
    plot_single_history: bool,
    regression_targets: list
  ) -> Tuple[dict, pd.DataFrame]:
  # loop over nodes
  models = {}
  histories = pd.DataFrame()
  for node in nodes_dataset:
    # set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    # train
    model, history = train_one_model(
      config = config, 
      model_creator = model_creator, 
      dataset = nodes_dataset[node], 
      checkpoint_path = os.path.join(output_folder, f"{node}_single.keras")
    )
    # save model
    models[node] = model
    # plot and save history
    history_df = pd.DataFrame(history)
    history_df["iter"] = history_df.index
    history_df["node"] = [node] * len(history_df)
    history_df.to_csv(
      os.path.join(output_folder, f"{node}_single_history.csv"), index = False
    )
    if plot_single_history:
      plot_history(history_df, output_folder, f"{node}_single")
    histories = pd.concat([histories, history_df], ignore_index = True)
    # compute, plot and save predictions
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = nodes_dataset[
      node
    ]
    predictions = compute_and_plot_predictions(
      {
        "train": (X_train, Y_train), 
        "val": (X_val, Y_val), 
        "test": (X_test, Y_test)
      }, 
      models[node], 
      output_folder, 
      f"{node}_single",
      regression_targets
    )
    predictions_json = {
      "train": {k: v.tolist() for k,v in enumerate(predictions["train"])},
      "val": {k: v.tolist() for k,v in enumerate(predictions["val"])},
      "test": {k: v.tolist() for k,v in enumerate(predictions["test"])}
    }
    with open(
        os.path.join(output_folder, f"{node}_single_pred.json"), "w"
      ) as ost:
      ost.write(json.dumps(predictions_json, indent = 2))
  return models, histories


def train_centralized_model(
    config: Config, 
    model_creator: functools.partial, 
    centralized_dataset: list, 
    output_folder: str,
    plot_single_history: bool,
    regression_targets: list
  ) -> Tuple[dict, pd.DataFrame]:
  # train
  model, history = train_one_model(
    config = config,
    model_creator = model_creator,
    dataset = centralized_dataset,
    checkpoint_path = os.path.join(output_folder, "centralized.keras")
  )
  # plot and save history
  history_df = pd.DataFrame(history)
  history_df["node"] = ["centralized"] * len(history_df)
  history_df["iter"] = history_df.index
  history_df.to_csv(
    os.path.join(output_folder, "centralized_history.csv"), index = False
  )
  if plot_single_history:
    plot_history(history_df, output_folder, "centralized")
  # compute, plot and save predictions
  predictions = compute_and_plot_predictions(
    {
      "train": centralized_dataset[0],
      "val": centralized_dataset[1],
      "test": centralized_dataset[2]
    },
    model,
    output_folder, 
    "centralized",
    regression_targets
  )
  predictions_json = {
    "train": {k: v.tolist() for k,v in enumerate(predictions["train"])},
    "val": {k: v.tolist() for k,v in enumerate(predictions["val"])},
    "test": {k: v.tolist() for k,v in enumerate(predictions["test"])}
  }
  with open(os.path.join(output_folder, "centralized_pred.json"), "w") as ost:
    ost.write(json.dumps(predictions_json, indent = 2))
  # return
  return model, history_df


def run_single_training_experiment(
    base_folder: str,
    network: int,
    config: Config, 
    model_creator: functools.partial, 
    seed: int,
    mode: str = "centralized",
    plot_single_history: bool = False
  ) -> Tuple[dict, pd.DataFrame]:
  # build name of data folder
  data_folder = os.path.join(base_folder, str(network))
  # build output folder
  output_folder = os.path.join(data_folder, f"results_{seed}")
  os.makedirs(output_folder, exist_ok = True)
  # load data and train according to mode
  models = {}
  tasks = {}
  histories = pd.DataFrame()
  if mode == "centralized":
    # set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    # centralized training
    is_already_trained, history_file, model_file = already_trained(
      output_folder, "centralized"
    )
    history_df = None
    if not is_already_trained:
      dataset = load_dataset(data_folder, "centralized")
      tasks = load_tasks(data_folder)
      models["centralized"], history_df = train_centralized_model(
        config = config,
        model_creator = model_creator,
        centralized_dataset = dataset,
        output_folder = output_folder,
        plot_single_history = plot_single_history,
        regression_targets = tasks["Multi_Task_regression"]["targets"]
      )
    else:
      print(
        f"  {mode} results for network {network} (seed {seed}) already exist!"
      )
      models["centralized"] = load_model(model_file)
      history_df = pd.read_csv(history_file)
    histories = pd.concat([histories, history_df], ignore_index = True)
  elif mode == "local":
    # local training for all nodes
    dataset = {}
    for node in range(config.n_nodes):
      is_already_trained, history_file, model_file = already_trained(
        output_folder, f"{node}_single"
      )
      if not is_already_trained:
        dataset[node] = load_dataset(data_folder, node)
        if len(tasks) == 0:
          tasks = load_tasks(data_folder)
      else:
        print(
          f"  {mode}-{node} results for network "
          f"{network} (seed {seed}) already exist!"
        )
        models[node] = None#load_model(model_file)
        history_df = pd.read_csv(history_file)
        histories = pd.concat([histories, history_df], ignore_index = True)
    if len(dataset) > 0:
      l_models, l_histories = train_local_models(
        config = config,
        model_creator = model_creator,
        nodes_dataset = dataset,
        seed = seed,
        output_folder = output_folder,
        plot_single_history = plot_single_history,
        regression_targets = tasks["Multi_Task_regression"]["targets"]
      )
      models = {**models, **l_models}
      histories = pd.concat([histories, l_histories], ignore_index = True)
    # plot history of different nodes
    plot_multinode_history(histories, output_folder, "local")
  return models, histories


def train(
    config_file: str, 
    base_folder: str,
    modes: str,
    seed: int,
    networks: list,
    n_sim_per_network: int,
    plot_single_history: bool
  ) -> Tuple[dict, pd.DataFrame]:
  # load configuration and define model creator
  config, model_creator = prepare_training(config_file)
  # build base i/o folder
  n = config.n_nodes
  k = config.connectivity
  io_folder = os.path.join(
    base_folder,
    f"{n}n-{k}k/seed{seed}"
  )
  # prepare seeds
  np.random.seed(seed)
  seeds = np.random.randint(1000, 10000, n_sim_per_network)
  # loop over simulations
  models = {}
  histories = pd.DataFrame()
  for network in networks:
    for _, internal_seed in enumerate(seeds):
      sim_models = {}
      sim_histories = pd.DataFrame()
      for mode in modes:
        mode_models, mode_histories = run_single_training_experiment(
          base_folder = io_folder,
          network = network,
          config = config,
          model_creator = model_creator,
          seed = int(internal_seed),
          mode = mode,
          plot_single_history = plot_single_history
        )
        mode_histories["mode"] = [mode] * len(mode_histories)
        # save
        sim_models = {**sim_models, **mode_models}
        sim_histories = pd.concat(
          [sim_histories, mode_histories], 
          ignore_index = True
        )
      # plot simulation history
      plot_multinode_history(
        sim_histories, io_folder, f"{network}_{internal_seed}_all"
      )
  return models, histories


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  config_file = args.config_file
  base_folder = args.base_folder
  modes = args.modes
  seed = args.seed
  networks = args.networks
  n_sim_per_network = args.n_sim_per_network
  plot_single_history = args.plot_single_history
  if not isinstance(modes, list):
    modes = [modes]
  if networks[0] == "all":
    networks = list(range(10))
  # run
  _ = train(
    config_file, 
    base_folder, 
    modes, 
    seed, 
    networks, 
    n_sim_per_network, 
    plot_single_history
  )
