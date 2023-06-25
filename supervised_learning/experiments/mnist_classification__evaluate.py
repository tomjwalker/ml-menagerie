import pickle

import pandas as pd

from supervised_learning.low_level_implementations.feedforward_nn.models import load_checkpoint
from supervised_learning.datasets.mnist.data_utils import show_digit_samples

import matplotlib.pyplot as plt

import numpy as np


def reorder_run_sweep(run_sweep_dict):
    """
    Run sweep is a nested dictionary, currently in the order:
        run_sweep_dict[run_suffix]["training_logs"/"evaluation_logs"][metric_name]: metric_log

    This function returns a pandas DataFrame. The first three columns correspond to the levels of the nested dictionary:
    - "run_config": run_suffix
    - "metric_type": "training_logs" or "evaluation_logs"
    - "metric_name": metric_name
    The remaining columns house each element of the corresponding metric log.
    """

    # Create empty pandas DataFrame
    df = pd.DataFrame()

    # Loop over all run configs
    for run_config, run_config_dict in run_sweep_dict.items():
        # Loop over all metric types
        for metric_type, metric_type_dict in run_config_dict.items():
            # Loop over all metric names
            for metric_name, metric_log in metric_type_dict.items():
                # Determine the maximum length of metric log
                max_length = max([len(log) for log in metric_type_dict.values()])

                # Create a dictionary to hold the column values
                data_dict = {
                    "run_config": [run_config],
                    "metric_type": [metric_type],
                    "metric_name": [metric_name],
                }

                # Add each element of metric log as a separate column
                for i in range(max_length):
                    column_name = f"metric_log_{i}"
                    if i < len(metric_log):
                        data_dict[column_name] = [metric_log[i]]
                    else:
                        data_dict[column_name] = [np.nan]

                # Append to DataFrame
                df = df.append(pd.DataFrame(data_dict))

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ========================================
# MAIN
# ========================================

# Parameters and architecture names. These are used both as parameters into the model/loop, and as filenames for
# saving the outputs of the training loop
DATA_CACHE_DIR = "./data_cache"
PLOTS_DIR = "./plots"
MODEL_CHECKPOINTS_DIR = "./model_checkpoints"

# List of dict of run settings. Each dict specifies a run, and the list of dicts specifies a set of runs to compare
RUN_SETTINGS = [
    {
        "model_name": "mnist_ffnn_dense_50_batchnorm",
        "num_epochs": 10,
        "train_abs_samples": 200,
        "clip_grads_norm": True,
    },
]

# Loop over all specified runs
run_sweep = {}
for run_settings in RUN_SETTINGS:
    # Append run suffix to each run settings dict
    run_suffix = "__".join([f"{key}_{str(value)}" for key, value in run_settings.items()])
    run_suffix = run_suffix.replace(".", "_")  # If any values are floats, replace "." with "_" for filename
    run_settings["run_suffix"] = run_suffix

    # ========================================
    # Load loop
    # ========================================

    with open(f"{MODEL_CHECKPOINTS_DIR}/{run_suffix}__loop.pkl", "rb") as f:
        loop = pickle.load(f)

    # Collect validation set features and labels
    features_validation = loop.features_val
    labels_validation = loop.labels_val

    # ========================================
    # Collect training and evaluation metrics
    # ========================================
    training_logs = loop.training_task.metric_log
    evaluation_logs = loop.evaluation_task.metric_log
    run_sweep[run_suffix] = {
        "training_logs": training_logs,
        "evaluation_logs": evaluation_logs,
    }

    # =============================================
    # Show samples of digits from the MNIST dataset
    # =============================================

    # Load trained models
    best_model = load_checkpoint(f"{MODEL_CHECKPOINTS_DIR}/{run_suffix}.pkl")
    last_model = load_checkpoint(f"{MODEL_CHECKPOINTS_DIR}/{run_suffix}__last_run.pkl")

    for model in [best_model, last_model]:
        predictions = model.forward_pass(features_validation, mode="infer")
        show_digit_samples(features_validation, labels_validation, predictions, m_samples=10)


run_sweep_df = reorder_run_sweep(run_sweep)
print(run_sweep_df)
