from train import compute_and_plot_predictions
from utils.data import load_dataset
from utils.metrics import compute_metrics

from matplotlib import colors as mcolors
from keras.api.models import load_model
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Postprocessing"
  )
  parser.add_argument(
    "-f", "--base_folder", 
    help="Paths to the base experiment folder", 
    type=str,
    required=True
  )
  parser.add_argument(
    "-m", "--merge_strategy", 
    help="Merge strategy adopted by GL experiments", 
    type=str,
    default=None
  )
  parser.add_argument(
    "--nodes", 
    help="Indices of nodes to consider (default: [0,10) + 'centralized')", 
    nargs="+",
    default="all"
  )
  parser.add_argument(
    "--networks", 
    help="Network indices (default: [0,10)", 
    nargs="+",
    default="all"
  )
  args, _ = parser.parse_known_args()
  return args


def compute_predictions(
    models: dict, all_X_Y_data: dict, plot_folders: str, tasks: dict
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # compute and plot predictions
  all_predictions = pd.DataFrame()
  all_metrics = {"key": [], "node": [], "seed": [], "metrics": []}
  for node, sim_models in models.items():
    for seed, model in sim_models.items():
      X_Y_data = all_X_Y_data[node][seed]
      predictions = compute_and_plot_predictions(
        X_Y_data, 
        model, 
        plot_folders[seed], 
        str(node), 
        tasks["Multi_Task_regression"]["targets"]
      )
      # get train, val, test
      for key, pred in predictions.items():
        Y_real = X_Y_data[key][1]
        # convert to dataframe and save
        df = get_predictions_df(Y_real, pred, tasks)
        df["key"] = key
        df["node"] = node
        df["seed"] = seed
        all_predictions = pd.concat([all_predictions, df], ignore_index = True)
        # compute metrics
        metrics = compute_metrics(Y_real, pred)
        all_metrics["key"].append(key)
        all_metrics["node"].append(node)
        all_metrics["seed"].append(seed)
        all_metrics["metrics"].append(metrics)
  # extract some surely-relevant metrics
  all_metrics = get_metrics_df(all_metrics)
  return all_predictions, all_metrics


def load_gossip_models_and_data(
    base_folder: str, network_index: int, merge_strategy: str
  ) -> Tuple[dict, dict, dict]:
  # data folder
  data_folder = os.path.join(base_folder, str(network_index))
  # loop over simulations
  models = {}
  all_X_Y_data = {}
  plot_folders = {}
  for foldername in os.listdir(base_folder):
    if foldername.startswith(f"gossip-MergeStrategy.{merge_strategy}"):
      # get seed
      internal_seed = int(foldername.removeprefix(
        f"gossip-MergeStrategy.{merge_strategy}_"
      ))
      # define folders and files
      models_folder = os.path.join(
        base_folder, foldername, str(network_index), "models"
      )
      common_test_file = os.path.join(
        base_folder, foldername, str(network_index), "common_test_set.json"
      )
      plot_folder = os.path.join(
        base_folder, foldername, str(network_index), "plots"
      )
      plot_folders[internal_seed] = plot_folder
      # load common test set
      X_test, Y_test = None, None
      with open(common_test_file, "r") as istream:
        common_test_set = json.load(istream)
        X_test = np.array(common_test_set["X_test"])
        Y_test = np.array(common_test_set["Y_test"])
      # load models and nodes-specific data
      for model_filename in os.listdir(models_folder):
        # get node id
        node = int(model_filename.split(".")[0])
        # load model
        model = load_model(os.path.join(models_folder, model_filename))
        if node not in models:
          models[node] = {}
        models[node][internal_seed] = model
        # load data
        train, val, test = load_dataset(data_folder, node)
        if node not in all_X_Y_data:
          all_X_Y_data[node] = {}
        all_X_Y_data[node][internal_seed] = {
          "train": train,
          "val": val,
          "test": test,
          "common_test": [X_test, Y_test]
        }
  return models, all_X_Y_data, plot_folders

def compute_common_test_predictions_by_node_type(
    models: dict,
    all_X_Y_data: dict,
    plot_folders: dict,
    tasks: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate every gossip model on the common test set,
    splitting common-test samples by node_type.
    """

    all_predictions = pd.DataFrame()

    all_metrics = {
        "key": [],
        "node": [],
        "common_test_node_type": [],
        "seed": [],
        "metrics": [],
    }

    for node, sim_models in models.items():
        for seed, model in sim_models.items():

            X_common, Y_common = (
                all_X_Y_data[node][seed]["common_test"]
            )

            X_common = np.asarray(X_common)
            Y_common = np.asarray(Y_common)

            if X_common.ndim != 3:
                raise RuntimeError(
                    "Common-test X must have 3 dimensions. "
                    f"Received shape: {X_common.shape}"
                )

            if len(X_common) != len(Y_common):
                raise RuntimeError(
                    "Common-test X and Y have different sample counts: "
                    f"{len(X_common)} != {len(Y_common)}"
                )

            # True for actual timesteps, False for zero-padding.
            valid_steps = np.any(
                X_common[:, :, :-1] != 0,
                axis=2,
            )

            sample_node_types = np.full(
                len(X_common),
                -1,
                dtype=int,
            )

            for sample_index in range(len(X_common)):
                node_types_in_sample = X_common[
                    sample_index,
                    :,
                    -1,
                ][valid_steps[sample_index]]

                if len(node_types_in_sample) == 0:
                    raise RuntimeError(
                        "Cannot determine node_type for common-test "
                        f"sample {sample_index}: no valid timestep."
                    )

                unique_node_types = np.unique(
                    node_types_in_sample
                )

                if len(unique_node_types) != 1:
                    raise RuntimeError(
                        "A common-test sample contains multiple "
                        f"node types: sample={sample_index}, "
                        f"types={unique_node_types.tolist()}"
                    )

                sample_node_types[sample_index] = int(
                    unique_node_types[0]
                )

            node_plot_folder = os.path.join(
                plot_folders[seed],
                "common_test_by_node_type",
                f"node_{node}",
            )

            os.makedirs(
                node_plot_folder,
                exist_ok=True,
            )

            for common_test_node_type in (0, 1, 2):

                type_mask = (
                    sample_node_types
                    == common_test_node_type
                )

                n_samples = int(type_mask.sum())

                if n_samples == 0:
                    print(
                        f"[COMMON_TEST_BY_TYPE] "
                        f"model_node={node}, seed={seed}, "
                        f"type={common_test_node_type}: "
                        "no samples"
                    )
                    continue

                X_type = X_common[type_mask]
                Y_type = Y_common[type_mask]

                dataset_key = (
                    f"common_test_node_type_"
                    f"{common_test_node_type}"
                )

                predictions = compute_and_plot_predictions(
                    {
                        dataset_key: (
                            X_type,
                            Y_type,
                        )
                    },
                    model,
                    node_plot_folder,
                    (
                        f"node_{node}_"
                        f"common_test_type_"
                        f"{common_test_node_type}"
                    ),
                    tasks[
                        "Multi_Task_regression"
                    ]["targets"],
                )

                pred = predictions[dataset_key]

                prediction_df = get_predictions_df(
                    Y_type,
                    pred,
                    tasks,
                )

                prediction_df["key"] = dataset_key
                prediction_df["node"] = node
                prediction_df[
                    "common_test_node_type"
                ] = common_test_node_type
                prediction_df["seed"] = seed

                all_predictions = pd.concat(
                    [
                        all_predictions,
                        prediction_df,
                    ],
                    ignore_index=True,
                )

                calculated_metrics = compute_metrics(
                    Y_type,
                    pred,
                )

                all_metrics["key"].append(dataset_key)
                all_metrics["node"].append(node)
                all_metrics[
                    "common_test_node_type"
                ].append(common_test_node_type)
                all_metrics["seed"].append(seed)
                all_metrics["metrics"].append(
                    calculated_metrics
                )

                class_values, class_counts = np.unique(
                    Y_type[:, -1].astype(int),
                    return_counts=True,
                )

                class_distribution = dict(
                    zip(
                        class_values.tolist(),
                        class_counts.tolist(),
                    )
                )

                print(
                    f"[COMMON_TEST_BY_TYPE] "
                    f"model_node={node}, "
                    f"seed={seed}, "
                    f"type={common_test_node_type}, "
                    f"samples={n_samples}, "
                    f"class_distribution="
                    f"{class_distribution}"
                )

    if len(all_metrics["metrics"]) == 0:
        return all_predictions, pd.DataFrame()

    return (
        all_predictions,
        get_metrics_df(all_metrics),
    )


def get_metrics_df(metrics_data: pd.DataFrame) -> pd.DataFrame:
  all_metrics = pd.DataFrame(metrics_data)
  all_metrics["mape"] = [m.mape * 100 for m in all_metrics["metrics"]]
  all_metrics["mse"] = [m.mse for m in all_metrics["metrics"]]
  all_metrics["r2"] = [m.r2 for m in all_metrics["metrics"]]
  all_metrics["accuracy"] = [
    m.cls_report["accuracy"] for m in all_metrics["metrics"]
  ]
  return all_metrics


def get_predictions_df(
    Y_real: np.array, pred, tasks: dict
  ) -> pd.DataFrame:
  real_df_reg = pd.DataFrame(
    Y_real[:,:-1], 
    columns = tasks["Multi_Task_regression"]["targets"]
  )
  real_df_cls = pd.DataFrame(
    Y_real[:,-1], 
    columns = tasks["Multi_Task_classification"]["targets"]
  )
  pred_df_reg = pd.DataFrame(
    pred[0], 
    columns = tasks["Multi_Task_regression"]["targets"]
  )
  pred_df_cls = pd.DataFrame(
    (pred[1] > 0.5).astype(int), 
    columns = tasks["Multi_Task_classification"]["targets"]
  )
  # save
  df = pd.concat(
    [
      real_df_reg.join(
        pred_df_reg, lsuffix = "_real", rsuffix = "_pred"
      ),
      real_df_cls.join(
        pred_df_cls, lsuffix = "_real", rsuffix = "_pred"
      )
    ],
    axis = "columns"
  )
  return df


def load_existing_predictions(
    data_folder: str, nodes: list, tasks: dict
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # loop over nodes
  all_predictions = pd.DataFrame()
  all_metrics = {"key": [], "node": [], "seed": [], "metrics": []}
  for node in nodes:
    # load data
    data = load_dataset(data_folder, node)
    # load predictions
    for foldername in os.listdir(data_folder):
      # loop over simulations
      if foldername.startswith("results_"):
        internal_seed = int(foldername.replace("results_", ""))
        # load
        pred_file = ""
        if node != "centralized":
          pred_file = os.path.join(
            data_folder, foldername, f"{node}_single_pred.json"
          )
        else:
          pred_file = os.path.join(
            data_folder, foldername, "centralized_pred.json"
          )
        predictions = {}
        with open(pred_file, "r") as istream:
          predictions = json.load(istream)
        # get train, val, test
        for idx, (key, pred_dict) in enumerate(predictions.items()):
          Y_real = data[idx][1]
          pred = {int(k): np.array(v) for k, v in pred_dict.items()}
          # convert to dataframe and save
          df = get_predictions_df(Y_real, pred, tasks)
          df["key"] = key
          df["node"] = node
          df["seed"] = internal_seed
          all_predictions = pd.concat(
            [all_predictions, df], ignore_index = True
          )
          # compute metrics
          metrics = compute_metrics(Y_real, pred)
          all_metrics["key"].append(key)
          all_metrics["node"].append(node)
          all_metrics["seed"].append(internal_seed)
          all_metrics["metrics"].append(metrics)
  # extract some surely-relevant metrics
  all_metrics = get_metrics_df(all_metrics)
  return all_predictions, all_metrics


def load_single_centralized_models(data_folder: str, nodes: list) -> dict:
  models = {}
  for node in nodes:
    # loop over simulations
    models[node] = {}
    for foldername in os.listdir(data_folder):
      if foldername.startswith("results_"):
        seed = int(foldername.replace("results_", ""))
        # get model filename
        model_filename = None
        if str(node) != "centralized":
          model_filename = os.path.join(
            data_folder, foldername, f"{node}_single.keras"
          )
        else:
          model_filename = os.path.join(
            data_folder, foldername, "centralized.keras"
          )
        # load model
        model = load_model(model_filename)
        models[node][seed] = model
  return models


def load_single_data(data_folder: str, nodes: list) -> dict:
  single_X_Y_data = {}
  for node in nodes:
    _, _, test = load_dataset(data_folder, node)
    single_X_Y_data[node] = {"test": test}
  return single_X_Y_data


def plot_metrics(
    all_metrics: pd.DataFrame, 
    experiment_idx: int, 
    relevant_metrics: list,
    output_folder: str = None
  ):
  # rename 'centralized' to 'C' (if available)
  if (all_metrics["node"] == "centralized").any():
    all_metrics.replace("centralized", "C", inplace = True)
  # get maximum and minimum number of nodes
  minnode = min([int(n) for n in all_metrics["node"].unique() if n != "C"])
  maxnode = max([int(n) for n in all_metrics["node"].unique() if n != "C"])
  # plot
  nrows = len(relevant_metrics)
  ncols = len(all_metrics["key"].unique())
  _, axs = plt.subplots(
    nrows = nrows,
    ncols = ncols,
    figsize = (8 * ncols, 3 * nrows),
    sharex = True,
    sharey = "row"
  )
  cidx = 0
  for key, metrics in all_metrics.groupby("key"):
    cax = axs if ncols == 1 else axs[:,cidx]
    ridx = 0
    for metric_name in relevant_metrics:
      rax = cax if nrows == 1 else cax[ridx]
      metrics.plot.bar(
        x = "node",
        y = metric_name,
        rot = 0,
        fontsize = 14,
        grid = True,
        label = None,
        ax = rax
      )
      rax.hlines(
        xmin = minnode,
        xmax = maxnode,
        y = float(metrics[metrics["node"] != "C"][metric_name].mean()),
        color = mcolors.TABLEAU_COLORS["tab:red"],
        linewidth = 2
      )
      if ridx == 0:
        rax.set_title(key, fontsize = 14)
      if cidx == 0:
        rax.set_ylabel(metric_name, fontsize = 14)
      if ridx == nrows - 1:
        rax.set_xlabel("Node", fontsize = 14)
      ridx += 1
    cidx += 1
  if output_folder is not None:
    plt.savefig(
      os.path.join(
        output_folder, f"{'_'.join(relevant_metrics)}_{experiment_idx}.png"
      ),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()


def plot_predictions(
    all_predictions: pd.DataFrame, 
    tasks: dict, 
    output_folder: str = None
  ):
  for (node, key), full_pred in all_predictions.groupby(["node", "key"]):
    # -- regression
    for target in tasks["Multi_Task_regression"]["targets"]:
      real = full_pred[f"{target}_real"]
      pred = full_pred[f"{target}_pred"]
      _, ax = plt.subplots(figsize = (7,3))
      ax.plot(
        real, pred, '.', label = None
      )
      ymin = float(real.min())
      ymax = float(real.max())
      ax.plot(
        np.linspace(ymin,ymax,1000), np.linspace(ymin,ymax,1000), "k--"
      )
      plt.title(f"{node}, {key}, {target}")
      if output_folder is not None:
        plt.savefig(
          os.path.join(
            output_folder, 
            f"predictions_{node}_{key}_{target}.png"
          ),
          dpi = 300,
          format = "png",
          bbox_inches = "tight"
        )
        plt.close()
      else:
        plt.show()


if __name__ == "__main__":
  # # parse arguments
  # args = parse_arguments()
  # base_folder = args.base_folder
  # merge_strategy = args.merge_strategy
  # nodes = args.nodes
  # networks = args.networks
  base_folder = "../experiments/18n-17k/seed4850"
  merge_strategy = "NODE_TYPE_MERGE"
  nodes = "all"
  networks = [0, 1, 2, 3, 4]#"all"
  recompute_all = True
  if not isinstance(nodes, list):
    if str(nodes) != "all":
      nodes = [nodes]
    else:
      nodes = list(range(18)) + ["centralized"]
  if not isinstance(networks, list):
    if str(networks) != "all":
      networks = [networks]
    else:
      networks = list(range(10))
  # run
  all_test_avg_metrics = pd.DataFrame()
  for idx in networks:
    # single/centralized predictions
    data_folder = os.path.join(
      base_folder, str(idx)
    )
    tasks = {}
    with open(os.path.join(data_folder, "tasks.json"), "r") as istream:
      tasks = json.load(istream)
    sc_metrics = None
    if not recompute_all and os.path.exists(
        os.path.join(data_folder, "sc_metrics.csv")
      ):
      sc_metrics = pd.read_csv(os.path.join(data_folder, "sc_metrics.csv"))
    else:
      sc_predictions, sc_metrics = load_existing_predictions(
        data_folder, nodes, tasks
      )
      sc_metrics.to_csv(
        os.path.join(data_folder, "sc_metrics.csv"), index = False
      )
      # plot_predictions_with_average(sc_predictions, 4, data_folder)
      plot_metrics(
        sc_metrics.groupby(
          ["node","key"]
        ).mean(numeric_only = True).reset_index().drop(
          "seed", axis = "columns"
        ), 
        idx, 
        ["mse", "mape", "r2", "accuracy"], 
        data_folder
      )
    # gossip predictions and metrics (if any)
    gossip_predictions, gossip_metrics, sc_generalized_metrics = [None] * 3
    sc_models = load_single_centralized_models(data_folder, nodes)
    if merge_strategy is not None:
      # -- load data
      g_models, g_X_Y_data, gplot_folders = load_gossip_models_and_data(
        base_folder, idx, merge_strategy
      )

      print("NETWORK:", idx)
      print("MODELLI GOSSIP TROVATI:", g_models.keys())
      print("SEED TROVATI:", gplot_folders.keys())

      # -- compute predictions and metrics
      if not recompute_all and os.path.exists(
          os.path.join(
            base_folder, f"{idx}_gossip_metrics_{merge_strategy}.csv"
          )
        ):
        gossip_metrics = pd.read_csv(
          os.path.join(
            base_folder, f"{idx}_gossip_metrics_{merge_strategy}.csv"
          )
        )
      else:
        gossip_predictions, gossip_metrics = compute_predictions(
          g_models, g_X_Y_data, gplot_folders, tasks
        )
        common_type_predictions, common_type_metrics = (
            compute_common_test_predictions_by_node_type(
                models=g_models,
                all_X_Y_data=g_X_Y_data,
                plot_folders=gplot_folders,
                tasks=tasks,
            )
        )

        if not common_type_metrics.empty:
            common_type_metrics.to_csv(
                os.path.join(
                    base_folder,
                    (
                        f"{idx}_common_test_by_node_type_"
                        f"{merge_strategy}.csv"
                    ),
                ),
                index=False,
            )

        for seed, plot_folder in gplot_folders.items():
          plot_predictions(
            gossip_predictions[gossip_predictions["seed"] == seed], 
            tasks, 
            plot_folder
          )
          plot_metrics(
            gossip_metrics[gossip_metrics["seed"] == seed], 
            idx, 
            ["mse", "mape", "r2", "accuracy"], 
            plot_folder
          )
          if not common_type_metrics.empty:
              common_type_seed_metrics = (
                  common_type_metrics[
                      common_type_metrics["seed"] == seed
                  ]
              )

              if not common_type_seed_metrics.empty:
                  common_type_plot_folder = os.path.join(
                      plot_folder,
                      "common_test_by_node_type",
                  )

                  os.makedirs(
                      common_type_plot_folder,
                      exist_ok=True,
                  )

                  plot_metrics(
                      common_type_seed_metrics,
                      idx,
                      ["mse", "mape", "r2", "accuracy"],
                      common_type_plot_folder,
                  )
        gossip_metrics.to_csv(
          os.path.join(
            base_folder, f"{idx}_gossip_metrics_{merge_strategy}.csv"
          ),
          index = False
        )
      # compute common-test predictions with single/centralized models
      # -- compute predictions and metrics with all models
      sc_generalized_metrics = None
      if not recompute_all and os.path.exists(
          os.path.join(data_folder, "sc_generalized_metrics.csv")
        ):
        sc_generalized_metrics = pd.read_csv(
          os.path.join(data_folder, "sc_generalized_metrics.csv")
        )
      else:
        sc_generalized_metrics = {"node": [], "seed": [], "metrics": []}
        for node, sim_models in sc_models.items():
          for seed, model in sim_models.items():
            if seed in g_X_Y_data[0]:
              # common test set
              common_test = {
                "common_test": g_X_Y_data[0][seed]["common_test"]
              }
              # build output folder
              plot_folder = os.path.join(
                data_folder, f"results_{seed}", "common_test_predictions"
              )
              os.makedirs(plot_folder, exist_ok = True)
              # compute  
              predictions = compute_and_plot_predictions(
                common_test, 
                model, 
                plot_folder, 
                str(node), 
                tasks["Multi_Task_regression"]["targets"]
              )
              metrics = compute_metrics(
                common_test["common_test"][1], predictions["common_test"]
              )
              sc_generalized_metrics["node"].append(node)
              sc_generalized_metrics["seed"].append(seed)
              sc_generalized_metrics["metrics"].append(metrics)
        sc_generalized_metrics = pd.DataFrame(sc_generalized_metrics)
        sc_generalized_metrics["mape"] = [
          m.mape * 100 for m in sc_generalized_metrics["metrics"]
        ]
        sc_generalized_metrics["mse"] = [
          m.mse for m in sc_generalized_metrics["metrics"]
        ]
        sc_generalized_metrics["r2"] = [
          m.r2 for m in sc_generalized_metrics["metrics"]
        ]
        sc_generalized_metrics["accuracy"] = [
          m.cls_report["accuracy"] for m in sc_generalized_metrics["metrics"]
        ]
        sc_generalized_metrics.to_csv(
          os.path.join(data_folder, "sc_generalized_metrics.csv"), 
          index = False
        )
    # compute local predictions with centralized models
    c_local_metrics = None
    if not recompute_all and os.path.exists(
        os.path.join(data_folder, "c_local_metrics.csv")
      ):
      c_local_metrics = pd.read_csv(
        os.path.join(data_folder, "c_local_metrics.csv")
      )
    else:
      # -- load data
      local_X_Y_data = load_single_data(data_folder, nodes)
      # -- loop over simulations
      c_local_metrics = {"node": [], "seed": [], "metrics": []}
      for seed, model in sc_models["centralized"].items():
        plot_folder = os.path.join(
          data_folder, f"results_{seed}", "local_centralized_predictions"
        )
        os.makedirs(plot_folder, exist_ok = True)
        for node, X_Y_data in local_X_Y_data.items():
          pred = compute_and_plot_predictions(
            X_Y_data, 
            model, 
            plot_folder, 
            node, 
            tasks["Multi_Task_regression"]["targets"]
          )
          metrics = compute_metrics(X_Y_data["test"][1], pred["test"])
          c_local_metrics["node"].append(node)
          c_local_metrics["seed"].append(seed)
          c_local_metrics["metrics"].append(metrics)
      c_local_metrics = pd.DataFrame(c_local_metrics)
      c_local_metrics["mape"] = [
        m.mape * 100 for m in c_local_metrics["metrics"]
      ]
      c_local_metrics["mse"] = [
        m.mse for m in c_local_metrics["metrics"]
      ]
      c_local_metrics["r2"] = [
        m.r2 for m in c_local_metrics["metrics"]
      ]
      c_local_metrics["accuracy"] = [
        m.cls_report["accuracy"] for m in c_local_metrics["metrics"]
      ]
      c_local_metrics.to_csv(
        os.path.join(data_folder, "c_local_metrics.csv"), index = False
      )
    # save test metrics
    # -- average over simulations
    sc_metrics = sc_metrics.groupby(
      ["node", "key"]
    ).mean(numeric_only = True).reset_index().drop("seed", axis = "columns")
    c_local_metrics = c_local_metrics.groupby("node").mean(
      numeric_only = True
    ).reset_index().drop("seed", axis = "columns")
    sc_generalized_metrics = sc_generalized_metrics.groupby("node").mean(
      numeric_only = True
    ).reset_index().drop("seed", axis = "columns")
    # -- concat
    test_avg_metrics = pd.concat(
      [
        # -- single
        pd.DataFrame(
          sc_metrics[
            (
              sc_metrics["key"] == "test"
            ) & (
              sc_metrics["node"] != "centralized"
            )
          ].mean(numeric_only = True),
          columns = ["single"]
        ).transpose(),
        # -- centralized (on local data)
        pd.DataFrame(
          c_local_metrics[
            c_local_metrics["node"] != "centralized"
          ][["mape", "mse", "r2", "accuracy"]].mean(numeric_only = True),
          columns = ["centralized"]
        ).transpose()
      ]
    )
    if merge_strategy is not None:
      test_avg_metrics = pd.concat(
        [
          test_avg_metrics,
          # -- gossip
          pd.DataFrame(
            gossip_metrics[gossip_metrics["key"] == "test"].mean(
              numeric_only = True
            ),
            columns = ["gossip"]
          ).transpose().drop("node", axis = "columns"),
          # -- single (generalized)
          pd.DataFrame(
            sc_generalized_metrics[
              sc_generalized_metrics["node"] != "centralized"
            ].mean(numeric_only = True),
            columns = ["single (generalized)"]
          ).transpose(),
          # -- centralized (generalized)
          sc_generalized_metrics[
            sc_generalized_metrics["node"] == "centralized"
          ][["node", "mape", "mse", "r2", "accuracy"]].replace(
            "centralized", "centralized (generalized)"
          ).set_index("node", drop = True),
          # -- gossip (generalized)
          pd.DataFrame(
            gossip_metrics[gossip_metrics["key"] == "common_test"].mean(
              numeric_only = True
            ),
            columns = ["gossip (generalized)"]
          ).transpose().drop("node", axis = "columns")
        ]
      )
    test_avg_metrics["idx"] = [idx] * len(test_avg_metrics)
    all_test_avg_metrics = pd.concat(
      [all_test_avg_metrics, test_avg_metrics]
    )
  all_test_avg_metrics.to_csv(
    os.path.join(
      base_folder, f"all_test_avg_metrics-MergeStrategy.{merge_strategy}.csv"
    )
  )
