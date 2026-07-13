from utils.centralized_training import aggregate_datasets
from utils.data import save_dataset

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import json
import os

import datetime
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedShuffleSplit
import gc
from math import ceil

node_types = ["LIGHT", "MID", "HEAVY"]
features = ['node_type',
 'rate_function_curl',
 'rate_function_eat_memory',
 'rate_function_env',
 'rate_function_figlet',
 'rate_function_nmap',
 'rate_function_shasum']# usata in preprocess_fold

VERSION = 'V1'
PATH_TO_AUGMENTED_ROWS = f"../data/augmented_rows/{VERSION}/" # usata in perform_custom_oversampling

# Function to import data
def import_data(path):
  df = pd.DataFrame()
  for node_type in node_types:
      # Retrieve all files in the data folder
      file_csv = [file for file in os.listdir(path + node_type) if file.endswith('.csv')]
      # Create the dataframe by concatenating all read files
      dataframes = []
      for file in file_csv:
          file_path = os.path.join(path + node_type, file)
          df_temp = pd.read_csv(file_path)
          # Remove the columns in the dataframe that begin with "function_"
          df_temp.drop(columns=[col for col in df_temp if col.startswith('function_')], inplace=True)
          # Add the column "node_type" and assign the value of 'type' to all rows
          if node_type == "HEAVY":
              df_temp["node_type"] = 0
          elif node_type == "MID":
              df_temp["node_type"] = 1
          else:
              df_temp["node_type"] = 2

          dataframes.append(df_temp)

      df = pd.concat([df, *dataframes], axis=0, ignore_index=True)

  return df


# Function used to fill NaN values within the dataframe X
def fill_NaN(X):
  for col in X:
    if(col.startswith('success_rate_')):
      X.loc[:, col] = X.loc[:, col].fillna(1)
    else:
      X.loc[:, col] = X.loc[:, col].fillna(0)
  return X

# Function to assign zero to noise metrics where requests rates are 0
def assign_zero_to_noise(df):
  functions = [col[14:] for col in df if col.startswith('rate')]

  for function in functions:
      df.loc[df['rate_function_' + function] == 0, ['cpu_usage_function_' + function, 'ram_usage_function_' + function, 'power_usage_function_' + function, 'replica_' + function]] = 0
  return df

# Function to select relevant columns
def select_columns(df):
  targets = [col for col in df if (col.startswith('cpu_usage_') or
                                   col.startswith('ram_usage_') or
                                   col.startswith('overloaded_node')
                                   or col.startswith('medium_latency')) and 'idle' not in col]# or col.startswith('replica')
  features = [col for col in df if col.startswith('rate_') or 'node_type' in col]
  features.sort()
  df = df[features+targets]
  return df

# Function to Rename Eat Memory column's Name
def rename_eat_memory(df):
  new_column_name = {i:i.replace("-","_") for i in df.keys() if "eat" in i}
  df.rename(columns=new_column_name, inplace=True)
  return df

# Function to add overload status and overload ratio.

# 0 = all 0, 1 = all 1, 2 = mix
def group_status(x):
    if (x == 0).all():
        return 0
    elif (x == 1).all():
        return 1
    else:
        return 2
    
def overload_status_ratio(df,features,target):
  grouped = df.groupby(features)

  df['overloaded_status'] = grouped[target].transform(group_status)

  # Ratio of 1’s in the group
  df['overloaded_ratio'] = grouped[target].transform('mean')

  return df

def get_node_capacity(node_id: int) -> int:
  # Lookup for node capacities
  GB = 1024 ** 3  # 1 GB in bytes
  capacities = {
      0: 24 * GB,   # Heavy
      1: 16 * GB,   # Mid
      2:  8 * GB    # Light
  }
  return capacities[node_id]

def ram_usage_to_percentage(ram_usage, node_type):
    """Return RAM utilization % given usage (bytes) and node type."""
    return (ram_usage / get_node_capacity(node_type)) * 100

# Function to compute theoretical RAM usage percentage.
def compute_ram_usage_percentage_theoretical(df, ram_col="ram_usage_node", node_col="node_type"):
    """Apply ram_usage_to_perc on the DataFrame."""
    # df["ram_usage_node_perc_theor"] = df.apply(
    #     lambda row: ram_usage_to_percentage(row[ram_col], row[node_col]),
    #     axis=1
    # )
    # Compute values
    perc_values = df.apply(
        lambda row: ram_usage_to_percentage(row[ram_col], row[node_col]),
        axis=1
    )

    # Find the index of ram_usage column
    insert_at = df.columns.get_loc("ram_usage_node_percentage") + 1

    # Insert new column at that position
    df.insert(insert_at, "ram_usage_node_perc_theor", perc_values)

    return df


#Remove rows where all specified columns have value 0.
def remove_baselines(df,features):
    df = df.copy()
    return df[~(df[features] == 0).all(axis=1)]

# Function to remove outliers
def remove_outliers(df):
  # Iterate over each target column and handle outliers
  functions_column = [col for col in df if col.startswith('rate')]
  targets = [col for col in df if (col.startswith('power_usage_') or col.startswith('cpu_usage_') or col.startswith('ram_usage_') or col.startswith('overloaded_node') or col.startswith('medium_latency')) and 'idle' not in col]
  grouped = df.groupby(functions_column + ['node_type'])
  threshold = 1
  for target in targets:
      print(target)
      if target != 'overloaded_node':
          mean = grouped[target].transform('mean')
          std = grouped[target].transform('std')
          outliers = (df[target] > mean + threshold * std) | (df[target] < mean - threshold * std)
          print(outliers.sum())
          df[target] = df[target].where(~outliers, mean)
      else:
          # Replace the overloaded value of the group by the mode.
          new_overloaded = grouped[target].transform(lambda x: x.mode().iloc[0])
          df['overloaded_node'] = new_overloaded
          print(df["overloaded_node"].value_counts())
  df_only_useful = df[functions_column + targets]
  return df

# creation of function to perform all the processes ---------------------------------------

# data preparation
def prepare_single_dataset(path_to_csvs: str, features: list) -> pd.DataFrame:
    df = import_data(path_to_csvs)
    df = fill_NaN(df)
    df = assign_zero_to_noise(df)
    df = select_columns(df)
    df = rename_eat_memory(df)
    df = overload_status_ratio(df, features, "overloaded_node")
    df = compute_ram_usage_percentage_theoretical(
        df,
        ram_col="ram_usage_node",
        node_col="node_type"
    )
    return df

def clean_for_model(df: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
    df = remove_baselines(df, numerical_features)
    df = remove_outliers(df)
    return df
# build task and dataset dictionaries
def build_tasks_and_datasets(df: pd.DataFrame):
    # target_prefixes = ("cpu_usage_", "ram_usage_", "overloaded_node", "medium_latency")
    # targets = [
    #     col for col in df.columns
    #     if col.startswith(target_prefixes) and "idle" not in col
    # ]

    regression_targets = [
    "cpu_usage_node",
    "ram_usage_node",
    "ram_usage_node_percentage"
    ]

    categorical_features = ["node_type"]
    numerical_features = [col for col in df.columns if col.startswith("rate_")]
    features = categorical_features + numerical_features

    categorical_targets = ["overloaded_node"]
    #numerical_targets = [col for col in targets if col not in categorical_targets]

    tasks = {
        "Multi_Task_regression": {
            "features": features,
            "targets": regression_targets
        },
        "Multi_Task_classification": {
            "features": features,
            "targets": categorical_targets
        }
    }

    tasks_unified = {
        "Multi_Task": {
            "features": features,
            "targets": regression_targets + categorical_targets,
            "regression_targets": regression_targets,
            "classification_targets": categorical_targets
        }
    }

    return features, numerical_features, tasks, tasks_unified

def prepare_feature_target_datasets_cv(df, tasks):
    """
    Prepare datasets for multiple targets regression and overloaded node classification
    with optional SMOTENC oversampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset containing both features and targets.
    features : list of str
        List of feature column names.
    targets : list of str
        List of target column names.
    categorical_features : list of str
        Categorical feature columns to be passed to SMOTENC.

    Returns
    -------
    features_datasets : dict
        Mapping target names to feature DataFrames (oversampled if needed).
    target_datasets : dict
        Mapping target names to target Series.
    """

    feature_dataset = {}
    target_dataset = {}
    df_augmented = None
    for task_name, task_info in tasks.items():
      X = df[task_info["features"]]
      y = df[task_info["targets"]]
      feature_dataset[task_name] = X
      target_dataset[task_name] = y

    return feature_dataset, target_dataset

# final data preparation function
def prepare_source_target_datasets(path_source: str, path_target: str):
    initial_features = ['node_type',
     'rate_function_curl',
     'rate_function_eat_memory',
     'rate_function_env',
     'rate_function_figlet',
     'rate_function_nmap',
     'rate_function_shasum']

    df_source = prepare_single_dataset(path_source, initial_features)
    df_target = prepare_single_dataset(path_target, initial_features)

    features, numerical_features, tasks, tasks_unified = build_tasks_and_datasets(df_source)

    df_source = clean_for_model(df_source, numerical_features)
    df_target = clean_for_model(df_target, numerical_features)

    feature_dataset_source, target_dataset_source = prepare_feature_target_datasets_cv(
        df_source, tasks_unified
    )
    feature_dataset_target, target_dataset_target = prepare_feature_target_datasets_cv(
        df_target, tasks_unified
    )

    return {
        "df_source": df_source,
        "df_target": df_target,
        "features": features,
        "numerical_features": numerical_features,
        "tasks": tasks,
        "tasks_unified": tasks_unified,
        "feature_dataset_source": feature_dataset_source,
        "target_dataset_source": target_dataset_source,
        "feature_dataset_target": feature_dataset_target,
        "target_dataset_target": target_dataset_target,
    }

#multi_target_x, multi_target_y, tasks, unified_tasks

# training time dataframe ----------------------------------------------------


# plot functions -------------------------------------------------------------------



# main function for testing ----------------------------------------


# if __name__ == "__main__":
#     path_to_csvs = "../data/raw/source_domain/"
#     path_to_csvs_target = "../data/raw/target_domain/"
#     node_types = ["LIGHT", "MID", "HEAVY"]

#     result = prepare_source_target_datasets(path_to_csvs, path_to_csvs_target)

#     df_source = result["df_source"]
#     # SORT 
#     df_sorted = (
#         df_source
#         .sort_values(by=df_source.columns.tolist())
#         .reset_index(drop=True)
#         .head(100)  # opzionale ma utile per confronto
#     )
#     #OUTPUT DIRECTORY
#     output_dir = "/Users/kingsley/Documents/TESI/TEST_RESULTS/TESTS"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "df_source_head100_sorted.csv")
#     # SAVE FOR CONRONTATION
#     df_sorted.to_csv(output_path, index=False)
#     print(f"✅ Salvato file ordinato in: {output_path}")



# functions to display results --------------------------------------------------------------

from IPython.display import display, HTML
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

def display_results(results_df):
  for task_name, df in results_df.items():
    printmd(f"# {task_name}")
    printmd("---")
    display(HTML(df.to_html()))



# network creaation and node distribution ------------------------------------------------------------------------------------------------------------------------------------



def convert_node_type(node_type: str) -> int:
  node_type_idx = -1
  if node_type == "HEAVY":
    node_type_idx = 0
  elif node_type == "MID":
    node_type_idx = 1
  elif node_type == "LIGHT":
    node_type_idx = 2
  else:
    raise RuntimeError(f"Node type `{node_type}` does not exist")
  return node_type_idx

def assign_node_type(
    towers: pd.DataFrame, node_types: list, rng: np.random.Generator
) -> pd.DataFrame:

    n_nodes = len(towers)
    n_types = len(node_types)

    base_count = n_nodes // n_types
    remainder = n_nodes % n_types

    assigned_types = []
    for node_type in node_types:
        assigned_types.extend([convert_node_type(node_type)] * base_count)
    if remainder > 0:
        extra_types = rng.choice(node_types, size=remainder, replace=False)
        for node_type in extra_types:
            assigned_types.append(convert_node_type(node_type))
    rng.shuffle(assigned_types)
    towers["node_type"] = assigned_types
    return towers

def assign_functions( # attualmente sembra non essere utilizzato i
    towers: pd.DataFrame, 
    functions: list, 
    n_functions_per_node: int,
    rng: np.random.Generator
  ) -> pd.DataFrame:
  functions_per_node = [
    [
      rng.choice(functions) for _ in range(n_functions_per_node)
    ] for _ in range(len(towers))
  ]
  towers["functions"] = functions_per_node
  return towers


def load_network(
    path_to_networks: str, 
    n: int, 
    k: int, 
    seed: int, 
    simulation: int, 
    rng: np.random.Generator, 
    node_types: list = None,
    functions: list = None,
    n_functions_per_node: int = None
  ):
  towers_file = os.path.join(
    path_to_networks, f"porto_{n}n_{k}k/seed{seed}/{simulation}/towers.csv"#path_to_networks, f"porto_{n}n_{k}k/{simulation}/towers.csv"#path_to_networks, f"porto_{n}n_{k}k/seed{seed}/{simulation}/towers.csv"
  )
  towers = pd.read_csv(towers_file)
  if node_types is not None:
    towers = assign_node_type(towers, node_types, rng)
  if functions is not None:
    towers = assign_functions(towers, functions, n_functions_per_node, rng)
  return towers

# funzione per distribuire i dati ai nodi

def build_nodes_dataframe(
    data_x: pd.DataFrame, data_y: pd.DataFrame, towers: pd.DataFrame
  ) -> dict:
  nodes_dataset = {}
  surely_test_set = {}
  for node_type, node_type_data in data_x.groupby("node_type"):
    # extract the corresponding y values
    node_type_targets = data_y.loc[node_type_data.index]
    # create groups based on identical feature vectors
    feature_cols = node_type_data.columns.tolist()
    df_tmp = pd.concat([node_type_data, node_type_targets], axis=1)
    df_tmp["_group_key"] = df_tmp[feature_cols].apply(
      lambda row: tuple(row.values),
      axis=1
    )
    # count nodes that have the required type
    nnodes = len(towers[towers["node_type"] == node_type])
    # count available groups for that node type (dividing overloaded and 
    # not overloaded)
    ngroups = len(df_tmp["_group_key"].unique())
    ngroups_overloaded = 0

    if "overloaded_node" in node_type_targets:
      ngroups_overloaded = len(
        df_tmp[
          df_tmp["overloaded_node"] == 1
        ]["_group_key"].unique()
      )

    nvpn, remainder = [None, None], [None, None]

    nvpn[0], remainder[0] = divmod(
      ngroups - ngroups_overloaded,
      nnodes
    )

    nvpn[1], remainder[1] = divmod(
      ngroups_overloaded,
      nnodes
    )

    # split equally
    for i, node_id in enumerate(
        towers[towers["node_type"] == node_type].index
      ):

      for overload_status in [0, 1]:

        selected_groups = (
          df_tmp[
            df_tmp["overloaded_node"] == overload_status
          ][["_group_key"]]
          .drop_duplicates()
          .iloc[
            i * nvpn[overload_status] : (i+1) * nvpn[overload_status]
          ]
        )

        idxs = df_tmp[
          df_tmp["_group_key"].isin(
            selected_groups["_group_key"]
          )
        ].index
        # add
        if node_id not in nodes_dataset:
          nodes_dataset[node_id] = {
            "x": node_type_data.loc[idxs,:],
            "y": node_type_targets.loc[idxs,:]
          }
        else:
          nodes_dataset[node_id]["x"] = pd.concat(
            [nodes_dataset[node_id]["x"], node_type_data.loc[idxs,:]]
          )
          nodes_dataset[node_id]["y"] = pd.concat(
            [nodes_dataset[node_id]["y"], node_type_targets.loc[idxs,:]]
          )
    # remainder data go into test
    for overload_status in [0, 1]:
      if remainder[overload_status] > 0:
        if node_type not in surely_test_set:
          surely_test_set[node_type] = {
            "x": pd.DataFrame(), "y": pd.DataFrame()
          }
        remainder_groups = (
          df_tmp[
            df_tmp["overloaded_node"] == overload_status
          ][["_group_key"]]
          .drop_duplicates()
          .iloc[-remainder[overload_status]:]
        )

        idxs = df_tmp[
          df_tmp["_group_key"].isin(
            remainder_groups["_group_key"]
          )
        ].index

        surely_test_set[node_type]["x"] = pd.concat(
          [surely_test_set[node_type]["x"], node_type_data.loc[idxs,:]]
        )
        surely_test_set[node_type]["y"] = pd.concat(
          [surely_test_set[node_type]["y"], node_type_targets.loc[idxs,:]]
        )
  # check coherence
  for node_id, node_data in nodes_dataset.items():
    if (node_data["x"].index != node_data["y"].index).any():
      raise RuntimeError(f"Incoherent dataset for node {node_id}")
  return nodes_dataset, surely_test_set

def prepare_network_and_nodes_dataset(# non usata (da deprecare)
    path_to_csvs_source: str,
    path_to_csvs_target: str,
    path_to_networks: str,
    n: int,
    k: int,
    simulation: int,
    rng: np.random.Generator,
    node_types: list
):
    """
    1. Carica dataset (source + target)
    2. Estrae data_x e data_y (Multi_Task)
    3. Costruisce il network
    4. Distribuisce i dati ai nodi

    Returns
    -------
    network : pd.DataFrame
    nodes_dataset : dict
    test_set : dict
    """

    # -------------------------
    # 1. Load dataset
    # -------------------------
    result = prepare_source_target_datasets(
        path_to_csvs_source,
        path_to_csvs_target
    )

    # -------------------------
    # 2. Extract Multi_Task
    # -------------------------
    data_x = result["feature_dataset_source"]["Multi_Task"]
    data_y = result["target_dataset_source"]["Multi_Task"]

    # -------------------------
    # 3. Load network
    # -------------------------
    network = load_network(
        path_to_networks,
        n,
        k,
        simulation,
        rng,
        node_types=node_types
    )

    # -------------------------
    # 4. Distribute data to nodes
    # -------------------------
    nodes_dataset, test_set = build_nodes_dataframe(
        data_x,
        data_y,
        network
    )

    return network, nodes_dataset, test_set


    


# FUTURE FUNCTIONS AFTER THE SPLIT ----------------------------------------------------------------------------------------------------------
# PREPROCESS FOLD
	# 1.	check_node_type_distribution
	# 2.	check_data_leakage
	# 3.	fit_minmax_scaler
	# 4.	save_scalers
	# 5.	transform_with_scalers
	# 6.	transform
	# 7.	preprocess_fold
#  PERFORM CUSTOM OVERSDAMPLING
	# 8.	limit_identical_combinations
	# 9.	generate_synthetic_overloaded
	# 10.	balance_df_nodewise
	# 11.	perform_custom_oversampling
# PREPROCESS_FOLD---------------------------------------------------------------------------------------------------------------
def check_node_type_distribution(X_train, X_val, X_test, node_col='node_type', normalize=True): # used in preprocessing(not used in kan) and preprocess_fold
    """
    Check and compare the distribution of `node_type` (or any categorical feature)
    across train, validation, and test splits.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        DataFrames for training, validation, and test sets.
    node_col : str, default='node_type'
        Name of the column representing node types (or any categorical variable).
    normalize : bool, default=True
        If True, show proportions (%) instead of raw counts.

    Returns
    -------
    pd.DataFrame
        Summary table comparing distributions across splits.
    """
    # Validate column existence
    for split_name, df in zip(['Train', 'Val', 'Test'], [X_train, X_val, X_test]):
        if node_col not in df.columns:
            raise ValueError(f"'{node_col}' not found in {split_name} set.")

    # Compute distributions
    def get_dist(df, name):
        return df[node_col].value_counts(normalize=normalize).rename(name)

    dist_train = get_dist(X_train, 'Train')
    dist_val   = get_dist(X_val, 'Val')
    dist_test  = get_dist(X_test, 'Test')

    # Combine into one comparison DataFrame
    dist_summary = pd.concat([dist_train, dist_val, dist_test], axis=1).fillna(0)
    if normalize:
        dist_summary = dist_summary * 100  # show as %

    # Add total rows for reference
    total_counts = {
        "Train Total": len(X_train),
        "Val Total": len(X_val),
        "Test Total": len(X_test)
    }
    totals_df = pd.DataFrame(total_counts, index=["Total Rows"])

    printmd("## Node Type Distribution Across Splits:")
    print(dist_summary.round(2))
    print("\n Dataset Sizes:")
    print(totals_df)

    return dist_summary



def check_data_leakage(X_train, X_val, X_test, group_cols=None): # used in preprocessing and preprocess_fold
    """
    Verify that there is no data leakage (i.e., no repeated feature combinations)
    across train, validation, and test splits.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        DataFrames for train, validation, and test sets.
    group_cols : list of str, optional
        Columns to consider for checking duplicates.
        If None, all columns of X_train are used.

    Returns
    -------
    None
        Prints leakage report. Raises AssertionError if leakage is detected.
    """


    if group_cols is None:
        group_cols = X_train.columns.tolist()

    # Create a unique signature for each feature combination
    def make_signatures(df):
        return set(df[group_cols].astype(str).agg('-'.join, axis=1))

    sig_train = make_signatures(X_train)
    sig_val   = make_signatures(X_val)
    sig_test  = make_signatures(X_test)

    # Check overlaps
    leak_train_val = sig_train.intersection(sig_val)
    leak_train_test = sig_train.intersection(sig_test)
    leak_val_test = sig_val.intersection(sig_test)

    total_leaks = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)

    printmd("## Data Leakage Check:")

    print(f"Train Samples: {len(X_train)}")
    print(f"Val Samples: {len(X_val)}")
    print(f"Test Samples: {len(X_test)}")

    print("\n")
    # print(f"- Train–Val overlaps: {len(leak_train_val)}")
    # print(f"- Train–Test overlaps: {len(leak_train_test)}")
    # print(f"- Val–Test overlaps: {len(leak_val_test)}")

    if total_leaks == 0:
        print("No data leakage detected. Splits are clean.")
    else:
        print("Data leakage detected!")
        if len(leak_train_val):
            print(f"  → {len(leak_train_val)} overlapping feature groups between Train and Val")
        if len(leak_train_test):
            print(f"  → {len(leak_train_test)} overlapping feature groups between Train and Test")
        if len(leak_val_test):
            print(f"  → {len(leak_val_test)} overlapping feature groups between Val and Test")
        #raise AssertionError("Data leakage detected between splits.")

def fit_minmax_scaler(X, y=None, regression_cols=None, classification_col=None): # used in preprocessing(not used in kan notebook) and preprocess_fold and prepare_cross_domain_evaluation_data(not used in the notebook)
    """
    Fit MinMaxScaler for features (and optionally regression targets).

    X:
      - node_type column is excluded from scaling

    y:
      - only regression columns are scaled
      - classification column remains unchanged
    """

    # ----------------------
    # 1. Handle feature scaler
    # ----------------------
    scaler_x = MinMaxScaler(feature_range=(1, 2))
    # Drop node_type if exists
    if "node_type" in X.columns:
        scaler_x.fit(X.drop(columns=["node_type"]))
    else:
        scaler_x.fit(X)

    # ----------------------
    # 2. Handle target scaler
    # ----------------------
    scaler_y = None
    if y is not None and regression_cols is not None:
        # Fit scaler on regression columns only
        scaler_y = MinMaxScaler(feature_range=(1, 2))
        scaler_y.fit(y[regression_cols])

    # ---- Logging ----
    print("[INFO] MinMaxScaler fitted:")
    print(f"  - Features shape: {X.shape}")
    if y is not None:
        print(f"  - Regression targets shape: {y[regression_cols].shape}")
        print(f"  - Classification column '{classification_col}' excluded from scaling")
    print()

    return scaler_x, scaler_y


def save_scalers(scaler_x, scaler_y, task_name, base_path='./scalers/functions/'): # used in preprocessing(not used in kan notebook) and preprocess_fold and prepare_cross_domain_evaluation_data(not used in the notebook)
    """
    Save fitted scalers for features and targets.

    Parameters
    ----------
    scaler_x : MinMaxScaler
        Fitted feature scaler.
    scaler_y : MinMaxScaler or None
        Fitted target scaler (if any).
    task_name : str
        Name of the task (e.g., 'Multi_Target_regression', 'overloaded_node_classification')
    base_path : str, default='./scalers/functions/'
        Base directory to save scalers.
    """
    # Create directories
    x_dir = os.path.join(base_path, 'scaler_x')
    y_dir = os.path.join(base_path, 'scaler_y')
    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    # Save X scaler
    joblib.dump(scaler_x, os.path.join(x_dir, f"{task_name}_x.joblib"))

    # Save Y scaler (only if provided)
    if scaler_y is not None:
        joblib.dump(scaler_y, os.path.join(y_dir, f"{task_name}_y.joblib"))

    print(f"[INFO] Scalers saved for target '{task_name}'")
    if scaler_y is None:
        print("  - No target scaler (classification or categorical target).")
    print()

def transform_with_scalers( # used in preprocessing and preprocess_fold 
    X_train, X_val, X_test, scaler_x,
    y_train=None, y_val=None, y_test=None,
    scaler_y=None, regression_cols=None, classification_col=None
):
    """
    Transform train/val/test using fitted scalers.

    X:
      - scales all columns except node_type
      - adds node_type back after scaling

    y:
      - scales only regression targets
      - keeps classification target unchanged
    """

    # =========================================
    # 1. Separate node_type from X
    # =========================================
    def split_node_type(df):
        if "node_type" in df.columns:
            node = df["node_type"].values.reshape(-1, 1)
            df_wo = df.drop(columns=["node_type"])
        else:
            node = None
            df_wo = df
        return df_wo, node


    X_train_noNT, node_train = split_node_type(X_train)
    X_val_noNT, node_val = split_node_type(X_val)
    X_test_noNT, node_test = split_node_type(X_test)

    # =========================================
    # 2. Scale numeric features only
    # =========================================
    X_train_scaled = scaler_x.transform(X_train_noNT)
    X_val_scaled   = scaler_x.transform(X_val_noNT)
    X_test_scaled  = scaler_x.transform(X_test_noNT)

    # Add back node_type (not scaled)
    if node_train is not None:
        X_train_scaled = np.hstack([node_train, X_train_scaled])
        X_val_scaled   = np.hstack([node_val, X_val_scaled ])
        X_test_scaled  = np.hstack([node_test, X_test_scaled])

    # =========================================
    # 3. Handle Y scaling
    # =========================================
    if y_train is not None and scaler_y is not None:

        # Extract and scale regression columns
        y_train_reg_s = scaler_y.transform(y_train[regression_cols])
        y_val_reg_s   = scaler_y.transform(y_val[regression_cols])
        y_test_reg_s  = scaler_y.transform(y_test[regression_cols])

        # Recombine regression + classification
        y_train_scaled = pd.DataFrame(y_train_reg_s, columns=regression_cols)
        y_val_scaled   = pd.DataFrame(y_val_reg_s, columns=regression_cols)
        y_test_scaled  = pd.DataFrame(y_test_reg_s, columns=regression_cols)

        # Add classification target unchanged
        y_train_scaled[classification_col] = y_train[classification_col].values
        y_val_scaled[classification_col]   = y_val[classification_col].values
        y_test_scaled[classification_col]  = y_test[classification_col].values

    else:
        y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

    # ---- Logging ----
    print("[INFO] Transformation complete.")
    print(f"  - X_train shape: {X_train_scaled.shape}")
    print(f"  - X_val shape: {X_val_scaled.shape}")
    print(f"  - X_test shape: {X_test_scaled.shape}\n")

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_scaled, y_val_scaled, y_test_scaled
    )

def transform(df, rate_columns=None, pad_value=0.0): # used in preprocessing and preprcess_fold and prepare_cross_domain_evaluation_data(not used in the notebook)
    """
    Transform the input dataframe into padded, variable-length sequences.

    Each sample becomes a sequence of steps where each step = [rate, method_onehot(6), node_type].
    Steps with zero rate are removed. Then sequences are padded to the same length.

    Returns:
        X_padded: np.ndarray, shape (num_samples, max_seq_len, 8)
        mask: np.ndarray, shape (num_samples, max_seq_len), 1 for valid, 0 for padded
    """

    if rate_columns is None:
        rate_columns = [
            'rate_function_env', 'rate_function_curl', 'rate_function_eat_memory',
            'rate_function_nmap', 'rate_function_shasum', 'rate_function_figlet'
        ]

    node_type = df['node_type'].values
    num_samples = df.shape[0]
    sequence_length = len(rate_columns)  # 6

    # One-hot encode method indices (0–5)
    method_ids = np.arange(sequence_length).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    method_onehot = encoder.fit_transform(method_ids)  # (6, 6)

    X_sequences = []

    for i in range(num_samples):
        node_val = node_type[i]
        rate_values = df.loc[i, rate_columns].values  # shape (6,)

        # Keep only non-zero steps
        nonzero_mask = rate_values != 1
        if not np.any(nonzero_mask):
            nonzero_mask = np.array([True])  # Keep one dummy step if all zero

        rate_seq = rate_values[nonzero_mask].reshape(-1, 1)
        method_seq = method_onehot[nonzero_mask]
        node_seq = np.full((np.sum(nonzero_mask), 1), node_val)

        seq = np.concatenate([rate_seq, method_seq, node_seq], axis=1)  # (L_i, 8)
        X_sequences.append(seq)

    # Pad all sequences to same length (post-padding with zeros)
    max_len = max(len(seq) for seq in X_sequences)
    X_padded = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=pad_value)

    # Create mask (1 for real steps, 0 for padded)
    mask = np.array([[1]*len(seq) + [0]*(max_len - len(seq)) for seq in X_sequences], dtype='float32')

    # Drop node_type column if you want to reuse df
    df = df.drop(columns=['node_type'])

    return X_padded

def preprocess_fold(X_train, X_val, X_test, y_train, y_val, y_test,tasks,base_path): # used in cross_validate
    """
    Perform full preprocessing pipeline:
    - Split data into train/validation/test sets
    - Fit MinMax scalers for features and (optionally) targets
    - Save scalers to disk
    - Transform datasets using fitted scalers
    - Return scaled datasets and scalers for all tasks
    """
    printmd("## Preprocessing")
    # -----------------------------------------------------
    # Initialize dictionaries to store splits and scalers
    # -----------------------------------------------------
    x_train_dict, x_val_dict, x_test_dict = {}, {}, {}
    y_train_dict, y_val_dict, y_test_dict = {}, {}, {}
    x_scalers, y_scalers = {}, {}

    # -----------------------------------------------------
    # Loop through each task for preprocessing
    # -----------------------------------------------------
    for task_name,task_info in tasks.items():
        printmd(f"# {task_name}")
        printmd("---")


        # -------------------------------------------------
        # Check for data leakage
        # -------------------------------------------------
        check_node_type_distribution(X_train,X_val,X_test)
        check_data_leakage(X_train,X_val,X_test)

        # -------------------------------------------------
        # Fit MinMaxScaler
        # Skip target scaling for classification tasks
        # -------------------------------------------------
        printmd("## Scalaing")
        if task_name == "overloaded_node_classification":
            scaler_x, scaler_y = fit_minmax_scaler(X_train)
        else:
            scaler_x, scaler_y = fit_minmax_scaler(X_train, y_train,
                                                   regression_cols=task_info["regression_targets"],
                                                  classification_col=task_info["classification_targets"])

        # -------------------------------------------------
        # Save fitted scalers to disk
        # -------------------------------------------------
        save_scalers(scaler_x, scaler_y, task_name, base_path)

        # -------------------------------------------------
        # Apply scaling to train/val/test sets
        # -------------------------------------------------
        (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled
        ) = transform_with_scalers(
            X_train, X_val, X_test,
            scaler_x,
            y_train, y_val, y_test,
            scaler_y,
            regression_cols=task_info["regression_targets"],
            classification_col=task_info["classification_targets"]
        )


        # -------------------------------------------------
        # Optional: Apply any custom transformation on scaled features
        # Skip this step for classification tasks
        # -------------------------------------------------
        print(f"Transforming for Janoosy: {datetime.datetime.now()}")
        if task_name != "overloaded_node_classification":
            X_train_scaled = transform(pd.DataFrame(X_train_scaled, columns=features))
            X_val_scaled = transform(pd.DataFrame(X_val_scaled, columns=features))
            X_test_scaled = transform(pd.DataFrame(X_test_scaled, columns=features))

        # -------------------------------------------------
        # Store scaled datasets and scalers in dictionaries
        # -------------------------------------------------
        x_train_dict[task_name] = X_train_scaled
        x_val_dict[task_name] = X_val_scaled
        x_test_dict[task_name] = X_test_scaled
        x_scalers[task_name] = scaler_x

        y_train_dict[task_name] = y_train_scaled
        y_val_dict[task_name] = y_val_scaled
        y_test_dict[task_name] = y_test_scaled
        y_scalers[task_name] = scaler_y

    # -----------------------------------------------------
    # Return all processed splits and scalers
    # -----------------------------------------------------
    return (
        x_train_dict, x_val_dict, x_test_dict, x_scalers,
        y_train_dict, y_val_dict, y_test_dict, y_scalers
    )


# perform_custom_oversampling (used in cross_validate) ----------------------------------------------------------------------------------------------------------
def limit_identical_combinations(df, features, target_cols, max_per_combination=1): # used in perform_custom_oversampling
    """
    Limit the dataset to at most `max_per_combination` identical feature combinations.

    Parameters
    ----------
    df : pd.DataFrame
    features : list
        Columns to group by when identifying identical combinations.
    target_cols : list
        Target column names.
    max_per_combination : int
        Maximum allowed identical combinations per group.

    Returns
    -------
    df_limited : pd.DataFrame
        Reduced datasets after limiting duplicates.
    """
    print(f"\n=== Limiting to max {max_per_combination} identical combinations ===")

    # df_combined = pd.concat([X, y], axis=1)
    df_limited = (
        df.groupby(features, as_index=False)
        .head(max_per_combination)
        .reset_index(drop=True)
    )
# old version
    #df_limited = (
        #df.groupby(features, group_keys=False)
        #.apply(lambda x: x.head(max_per_combination))
        #.reset_index(drop=True)
    #)

    X_limited = df_limited[features]
    y_limited = df_limited[target_cols]

    print(f"Original row count: {len(df)}")
    print(f"After limiting: {len(df_limited)} (removed {len(df) - len(df_limited)})")
    print("\nClass distribution after limiting:")
    print(y_limited.value_counts())

    return df_limited


def generate_synthetic_overloaded(df, features, target_col): # used in perform_custom_oversampling
    """
    Generate synthetic overloaded samples to balance classes per node_type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (after limiting).
    features : list
        Feature column names.
    target_col : str
        Binary target column name (0 = non-overloaded, 1 = overloaded).

    Returns
    -------
    df_augmented : pd.DataFrame
        Dataset containing newly generated synthetic rows.
    """
    print("\n=== Generating synthetic overloaded samples (Unique Features Balancing) ===")

    rate_cols = [col for col in df.columns if col.startswith("rate_function_")]
    node_types = df["node_type"].unique()
    augmented_rows = []
    generated_total = 0

    existing_keys = set(tuple(row[col] for col in features) for _, row in df.iterrows())

    for nt in node_types:
        subset = df[df["node_type"] == nt]
        count_1 = subset[subset[target_col] == 1].shape[0]
        count_0 = subset[subset[target_col] == 0].shape[0]
        needed = count_0 - count_1

        print(f"\n[Node Type {nt}] Overloaded: {count_1}, Non-overloaded: {count_0}")
        if needed <= 0:
            print(f"Already balanced or overloaded.")
            continue

        print(f"Generating {needed} synthetic samples for node_type {nt}")
        df_overloaded = subset[subset[target_col] == 1]
        generated = 0

        for _, row in df_overloaded.iterrows():
            overloaded_funcs = [col for col in rate_cols if row[col] > 0]
            if len(overloaded_funcs) != 3:
                continue

            fixed_rates = {col: row[col] for col in overloaded_funcs}

            for target_func in overloaded_funcs:
                start = int(fixed_rates[target_func])
                for val in range(start + 5, 201, 10):
                    if generated >= needed:
                        break

                    new_row = {col: 0.0 for col in rate_cols}
                    for col in overloaded_funcs:
                        new_row[col] = val if col == target_func else fixed_rates[col]

                    # Fill metadata and targets
                    new_row["node_type"] = nt
                    new_row[target_col] = 1

                    # Add other columns from original
                    for col in df.columns:
                        if col not in new_row and col not in rate_cols:
                            new_row[col] = row[col]

                    new_key = tuple(new_row[col] for col in features)
                    if new_key not in existing_keys:
                        augmented_rows.append(new_row)
                        existing_keys.add(new_key)
                        generated += 1
                        generated_total += 1

                if generated >= needed:
                    break
            if generated >= needed:
                break

        print(f"Generated {generated} samples for node_type {nt}")

    print(f"\n Total synthetic samples generated: {generated_total}")
    return pd.DataFrame(augmented_rows)


def balance_df_nodewise(df, df_augmented, node_col='node_type', target_col='overloaded_node'): # used in perform_custom_oversampling
    """
    Balance df within each node_type based on overloaded_node (target).
    df_augmented contains only rows with overloaded_node = 1.
    Includes detailed debug prints.
    """

    balanced_parts = []
    node_types = df[node_col].unique()

    print("\n========= ROWS BALANCING =========\n")
    print(f"Node types found: {list(node_types)}\n")

    for nt in node_types:
        print(f"\n--- Processing node_type: {nt} ---")

        subset = df[df[node_col] == nt].copy()

        # Count original samples
        count_1 = (subset[target_col] == 1).sum()
        count_0 = (subset[target_col] == 0).sum()

        print(f"Original counts → overloaded=0: {count_0}, overloaded=1: {count_1}")

        # Check if balance is needed
        if count_1 >= count_0:
            print("Already balanced or overloaded=1 is majority. No augmentation needed.")
            balanced_parts.append(subset)
            continue

        needed = count_0 - count_1
        print(f"Need to add {needed} more rows for overloaded=1")

        # Filter augmented rows for this node_type
        aug_rows = df_augmented[df_augmented[node_col] == nt]

        if aug_rows.empty:
            print(f"[WARNING] df_augmented has NO rows for node_type={nt}. Cannot balance this group!")
            balanced_parts.append(subset)
            continue

        print(f"Augmented rows available for this node_type={nt}: {len(aug_rows)}")

        # Calculate repeat factor
        repeats = ceil(needed / len(aug_rows))
        print(f"Repeating augmented rows {repeats} times")

        repeated = pd.concat([aug_rows] * repeats, ignore_index=True)

        rows_to_add = repeated.iloc[:needed]
        print(f"Rows actually added: {len(rows_to_add)}")

        # Append augmented rows
        subset = pd.concat([subset, rows_to_add], ignore_index=True)

        # Verify new counts
        new_count_1 = (subset[target_col] == 1).sum()
        new_count_0 = (subset[target_col] == 0).sum()
        print(f"After augmentation → overloaded=0: {new_count_0}, overloaded=1: {new_count_1}")

        balanced_parts.append(subset)

    print("\n========= BALANCING COMPLETED =========\n")
    final_df = pd.concat(balanced_parts, ignore_index=True)
    return final_df


def perform_custom_oversampling(df, features, reg_target, class_target): # used in prepare_feature_target_datasets(not in the notebook) and cross_validate
    """
    Custom domain-aware oversampling replacing SMOTENC.

    Steps:
    1. Limit identical feature combinations.
    2. Generate synthetic overloaded samples per node_type to balance unique features(Group level balance).
    3. Row-level Balance by duplicating augmented rows and returning a balanced dataset.

    Parameters
    ----------
    df : pd.DataFrame
    features : list
        Feature columns used for grouping and synthesis.
    reg_target : list
        Regression Target column(s).
    class_target : list
        Classification Target column(s).

    Returns
    -------
    X_resampled, y_resampled : pd.DataFrame, pd.DataFrame
        Balanced feature and target datasets.
    """
    printmd("## Oversampling")
    # Step 1: Apply limiting rule
    df_class = df[features + class_target]
    df_limited = limit_identical_combinations(df_class, features, class_target)

    # Step 2: Combine for rebalancing
    # df_limited = pd.concat([X_limited, y_limited], axis=1)
    target_col = class_target[0]

    # Step 3: Generate new samples to balance limited dataset
    df_augmented = generate_synthetic_overloaded(df_limited, features, target_col)

    # Optionally save the augmented rows
    os.makedirs(PATH_TO_AUGMENTED_ROWS, exist_ok=True)
    df_augmented.to_csv(PATH_TO_AUGMENTED_ROWS + "augmented_overloaded_samples.csv", index=False)
    print("Augmented rows saved to 'augmented_overloaded_samples.csv'")

    # Step 4: Final overall balance dataset (repeated features)
    df_balanced = balance_df_nodewise(df, df_augmented)
    X_res = df_balanced[features]
    y_res = df_balanced[reg_target + class_target]

    print("\nFinal class distribution after rebalancing:")
    print(y_res[class_target].value_counts())
    print("\nFinal class distribution after rebalancing (node_type-wise):")
    print(df_balanced.groupby("node_type")["overloaded_node"].value_counts().unstack(fill_value=0))

    print("\nFinal node_type distribution:")
    print(X_res["node_type"].value_counts())

    print("\n========= FINAL UNIQUE FEATURES SUMMARY =========")

    # Convert to tuples for unique combinations
    all_unique = df_balanced[features].drop_duplicates()
    unique_0 = df_balanced[df_balanced[target_col] == 0][features].drop_duplicates()
    unique_1 = df_balanced[df_balanced[target_col] == 1][features].drop_duplicates()

    print(f"Total unique feature combinations       : {len(all_unique)}")
    print(f"Unique feature combinations for 0       : {len(unique_0)}")
    print(f"Unique feature combinations for 1       : {len(unique_1)}")

    print("==========================================\n")

    return X_res, y_res, df_augmented, df_balanced

# single_split_preprocessing -------------------------------------------------------------------------------------------------

def stratified_group_shuffle_split(X, y, groups=None, test_size=0.2, random_state=42):
    """
    Performs a stratified shuffle split on groups.
    Optimized for performance and includes memory cleanup.
    """
    if groups is None:
        # Generate unique hash for each row to represent groups.
        # This is generally faster than X.apply(tuple, axis=1) for DataFrames.
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        groups = pd.util.hash_pandas_object(X_df, index=False)
        # Ensure 'groups' is a Series with the correct index for subsequent operations
        groups = pd.Series(groups, index=X.index)

    rng = check_random_state(random_state)

    # 1. group-level table
    df_group_y = pd.DataFrame({"group": groups, "y": y}, index=X.index)
    # Ensure y is numeric for mode calculation or handle non-numeric cases if necessary
    group_labels = df_group_y.groupby("group")["y"].agg(lambda s: s.mode()[0])

    unique_groups = group_labels.index.values
    strat_labels = group_labels.values

    # 2. stratified split on groups
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    (train_groups_idx, test_groups_idx) = next(sss.split(unique_groups, strat_labels))

    train_groups = unique_groups[train_groups_idx]
    test_groups = unique_groups[test_groups_idx]

    # 3. map back to row indices using boolean indexing for efficiency
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)

    train_idx = X.index[train_mask].values
    test_idx = X.index[test_mask].values

    # Memory cleanup
    del df_group_y, group_labels, unique_groups, strat_labels, train_mask, test_mask
    if groups is not None and isinstance(groups, pd.Series):
        del groups
    del rng
    gc.collect()

    return train_idx, test_idx

def get_folds(X, y, n_splits=5, random_state=42):

    # overload is last column
    y_overload = y.values[:, -1].astype(int)

    # group on identical feature vectors
    groups = X.apply(lambda row: tuple(row.values), axis=1)

    skf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    return skf.split(X.values, y_overload, groups)

def save_xy_split(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    save_dir,
    index=False
):
    """
    Save X and y train/val/test splits separately.

    Files saved:
    - X_train.csv, y_train.csv
    - X_val.csv, y_val.csv
    - X_test.csv, y_test.csv
    """

    os.makedirs(save_dir, exist_ok=True)

    X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=index)
    y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=index)

    X_val.to_csv(os.path.join(save_dir, "X_val.csv"), index=index)
    y_val.to_csv(os.path.join(save_dir, "y_val.csv"), index=index)

    X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=index)
    y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=index)

def single_split_preprocessing(
    features_dict,
    targets_dict,
    tasks_unified,
    output_path,
    scaler_path,
    n_splits=5,
    random_state=42
):
    task_name = "Multi_Task"
    task_info = tasks_unified[task_name]

    printmd(f"# Task: {task_name}")
    printmd("---")

    X = features_dict
    y = targets_dict

    # =====================================
    # STESSO SPLIT DELLA CROSS_VALIDATE
    # ma uso solo il primo fold
    # =====================================
    print(f"Start CV Split (single fold only): {datetime.datetime.now()}")
    splits = get_folds(X, y, n_splits=n_splits, random_state=random_state)

    train_idx, test_idx = next(splits)

    # Outer split: train/test
    X_train_outer, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # =====================================
    # STESSO SPLIT TRAIN/VAL DELLA CROSS_VALIDATE
    # =====================================
    print(f"Start Train/Val Split: {datetime.datetime.now()}")
    inner_train_idx, inner_val_idx = stratified_group_shuffle_split(
        X_train_outer,
        y_train_outer.values[:, -1].astype(int),
        random_state=random_state
    )

    X_train = X_train_outer.loc[inner_train_idx]
    y_train = y_train_outer.loc[inner_train_idx]
    X_val = X_train_outer.loc[inner_val_idx]
    y_val = y_train_outer.loc[inner_val_idx]

    # =====================================
    # OVERSAMPLING SOLO TRAIN
    # =====================================
    print(f"Start Oversampling: {datetime.datetime.now()}")
    df_train = pd.concat([X_train, y_train], axis=1)

    X_train, y_train, df_augmented, df_balanced = perform_custom_oversampling(
        df_train,
        task_info["features"],
        task_info["regression_targets"],
        task_info["classification_targets"]
    )

    # =====================================
    # SALVA SPLIT RAW
    # =====================================
    save_xy_split(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        output_path
    )

    # =====================================
    # PREPROCESSING
    # =====================================
    print(f"Start Preprocessing: {datetime.datetime.now()}")

    (
        x_train_dict, x_val_dict, x_test_dict, x_scalers,
        y_train_dict, y_val_dict, y_test_dict, y_scalers
    ) = preprocess_fold(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        tasks_unified,
        scaler_path
    )

    all_outputs = {
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "y_train_raw": y_train,
        "y_val_raw": y_val,
        "y_test_raw": y_test,
        "df_augmented": df_augmented,
        "df_balanced": df_balanced,
        "x_train_dict": x_train_dict,
        "x_val_dict": x_val_dict,
        "x_test_dict": x_test_dict,
        "y_train_dict": y_train_dict,
        "y_val_dict": y_val_dict,
        "y_test_dict": y_test_dict,
        "x_scalers": x_scalers,
        "y_scalers": y_scalers
    }

    return all_outputs



def prepare_data(
    base_output_folder,
    path_to_csvs,
    path_to_csvs_target,
    path_to_networks,
    PATH_TO_SPLITS,
    PATH_TO_SCALERS,
    n,
    k,
    seed,
    simulation,
    node_types,
    split_seed
):

    # ===============================
    # OUTPUT FOLDER
    # ===============================
    output_folder = os.path.join(
        base_output_folder, f"{n}n-{k}k", f"seed{seed}", str(simulation)
    )

    if os.path.exists(output_folder):
        print(f"WARNING: Output folder {output_folder} already exists!")
        return output_folder

    os.makedirs(output_folder)

    # ===============================
    # LOAD DATA
    # ===============================
    data = prepare_source_target_datasets(path_to_csvs, path_to_csvs_target)

    multi_target_x = data["feature_dataset_source"]["Multi_Task"]
    multi_target_y = data["target_dataset_source"]["Multi_Task"]
    tasks = data["tasks"]
    unified_tasks = data["tasks_unified"]

    # ===============================
    # LOAD NETWORK
    # ===============================
    print("load nodes network")
    rng = np.random.default_rng(seed)

    network = load_network(
        path_to_networks,
        n,
        k,
        seed,
        simulation,
        rng,
        node_types
    )
    print("...done")

    # ===============================
    # BUILD NODE DATASETS
    # ===============================
    print("build nodes dataframes")

    nodes_dataframe, test_set = build_nodes_dataframe(
        multi_target_x,
        multi_target_y,
        network
    )

    print("...done")

    # ===============================
    # SPLIT & SAVE
    # ===============================
    print("split and save")

    prepared_nodes_dataset = {}
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for node, node_data in nodes_dataframe.items():

        print(f"processing node {node}")

        node_split_path = os.path.join(PATH_TO_SPLITS, f"node_{node}")
        node_scaler_path = os.path.join(PATH_TO_SCALERS, f"node_{node}")

        outputs = single_split_preprocessing(
            node_data["x"],
            node_data["y"],
            unified_tasks,
            node_split_path,
            node_scaler_path,
            n_splits=5,
            random_state=split_seed
        )

        X_train = outputs["x_train_dict"]["Multi_Task"]
        Y_train = outputs["y_train_dict"]["Multi_Task"].to_numpy()

        X_val = outputs["x_val_dict"]["Multi_Task"]
        Y_val = outputs["y_val_dict"]["Multi_Task"].to_numpy()

        X_test = outputs["x_test_dict"]["Multi_Task"]
        Y_test = outputs["y_test_dict"]["Multi_Task"].to_numpy()

        # Save in memory
        prepared_nodes_dataset[node] = (
            X_train, Y_train, X_val, Y_val, X_test, Y_test
        )

        # Save per node
        save_dataset(
            X_train,
            Y_train,
            X_val,
            Y_val,
            X_test,
            Y_test,
            output_folder,
            node
        )

        # For centralized dataset
        train_datasets.append((X_train, Y_train))
        val_datasets.append((X_val, Y_val))
        test_datasets.append((X_test, Y_test))

    # ===============================
    # CENTRALIZED DATASET
    # ===============================
    centralized_train_data = aggregate_datasets(train_datasets)
    centralized_val_data = aggregate_datasets(val_datasets)
    centralized_test_data = aggregate_datasets(test_datasets)

    save_dataset(
        *centralized_train_data,
        *centralized_val_data,
        *centralized_test_data,
        output_folder,
        "centralized"
    )

    # ===============================
    # SAVE TASK METADATA
    # ===============================
    with open(os.path.join(output_folder, "tasks.json"), "w") as ost:
        ost.write(json.dumps(tasks, indent=2))

    with open(os.path.join(output_folder, "unified_tasks.json"), "w") as ost:
        ost.write(json.dumps(unified_tasks, indent=2))

    print("...done")

    return output_folder
