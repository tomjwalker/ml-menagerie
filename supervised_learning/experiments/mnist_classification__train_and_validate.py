import os

import numpy as np
import pandas as pd

from supervised_learning.datasets.mnist.data_utils import load_mnist, preprocess_mnist, show_digit_samples
from supervised_learning.low_level_implementations.feedforward_nn.costs_and_metrics import (
    CategoricalCrossentropyCost, AccuracyMetric)
from supervised_learning.low_level_implementations.feedforward_nn.layers import Dense, Relu, Softmax, BatchNorm
from supervised_learning.low_level_implementations.feedforward_nn.models import SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser
from supervised_learning.low_level_implementations.feedforward_nn.tasks import (TrainingTask, EvaluationTask,
                                                                                ModelSaveTask, Loop)

import matplotlib.pyplot as plt


def summarise_grads_logs(loop, norm_type=2):
    """
    If specified in loop.run, the loop will log the gradient norms for each layer. These are stored as a list of
    dicts of dicts:
    a. List: Each element corresponds to an iteration.
    b. Dict: Each key corresponds to a layer.
    c. Dict: Each key corresponds to either a layer's "weights" or "bias".

    This function calculates the L2 norm of the gradients for each layer, across all iterations, and returns them as a
    Pandas DataFrame where:
    a. Rows are iteration numbers
    b. Columns are a combination of layer and weight/bias (e.g. "layer_1_weights", "layer_2_bias", etc.)
    """

    grads_log = loop.grads_log

    # Loop over internal dictionaries, and calculate norm of each weight array
    grads_log_norms = {}
    for k, v in grads_log.items():
        grads_log_norms[k] = {arr_name: np.linalg.norm(arr, ord=norm_type) for arr_name, arr in v.items()}

    # <>.T ensures ordering of `n_iterations` rows x `n_weight_arrays` columns
    norm_df = pd.DataFrame(grads_log_norms).T

    return norm_df


def plot_metric_logs_vs_gradient_norms(metric_log_training, metric_log_evaluation, metric_name, loop,
                                       plot_dir="./plots",
                                       run_config_suffix=""):
    """
    Function creates a 2x1 subplot, where the top plot is the plot from `plot_metric_logs`, and the bottom plot is
    the plot from `plot_gradient_norms`.
    """

    # Create directory for plots if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot metric logs
    axs[0].plot(metric_log_training[metric_name], label="Training")
    axs[0].plot(metric_log_evaluation[metric_name], label="Evaluation")
    axs[0].set_xlabel("Iteration number")
    axs[0].set_ylabel(metric_name)
    axs[0].legend()

    # Plot gradient norms
    norm_df = summarise_grads_logs(loop)
    axs[1].plot(norm_df)
    axs[1].set_xlabel("Iteration number")
    axs[1].set_ylabel("Gradient norm")
    axs[1].legend(norm_df.columns, loc="upper right")

    # Draw a vertical line where metric_log_training is at its maximum - on both plots
    max_metric_log_training = np.argmax(metric_log_training[metric_name])
    axs[0].axvline(max_metric_log_training, color="black", linestyle="--")
    axs[1].axvline(max_metric_log_training, color="black", linestyle="--")

    # Get the gradient norm values at the maximum of the metric_log_training.
    # Display these values on the lower axes
    max_norms = norm_df.iloc[max_metric_log_training]
    for i, norm in enumerate(max_norms):
        axs[1].text(
            max_metric_log_training, norm, f"{norm:.2f}", horizontalalignment="right", verticalalignment="bottom"
        )

    # Display text for the maximum of the metric_log_training
    axs[0].text(
        max_metric_log_training, metric_log_training[metric_name][max_metric_log_training],  # x, y
        f"{metric_log_training[metric_name][max_metric_log_training]:.2f}",
        horizontalalignment="right", verticalalignment="bottom"
    )

    # Display also the equivalent value from the metric_log_evaluation at the same x-coordinate
    axs[0].text(
        max_metric_log_training, metric_log_evaluation[metric_name][max_metric_log_training],  # x, y
        f"{metric_log_evaluation[metric_name][max_metric_log_training]:.2f}",
        horizontalalignment="right", verticalalignment="top"
    )

    plt.show()

    # Save plot
    fig.savefig(f"{plot_dir}/{metric_name}_vs_gradient_norms__{run_config_suffix}.png")


def save_metric_logs(metric_log_training, metric_log_evaluation, metric_name, metric_log_dir="./data_cache",
                     run_config_suffix=""):
    """
    Saves outputs of the training and evaluation metric logs during the training loop to a subdirectory within the
    current working directory.
    """

    # If data_cache directory does not exist, create it
    if not os.path.exists(metric_log_dir):
        os.makedirs(metric_log_dir)

    np.save(
        f"{metric_log_dir}/{metric_name}_training_log__{run_config_suffix}.npy", metric_log_training[metric_name]
    )
    np.save(
        f"{metric_log_dir}/{metric_name}_evaluation_log__{run_config_suffix}.npy", metric_log_evaluation[metric_name]
    )


# ========================================
# Main script
# ========================================

# Parameters and architecture names. These are used both as parameters into the model/loop, and as filenames for
# saving the outputs of the training loop
DATA_CACHE_DIR = "./data_cache"
PLOTS_DIR = "./plots"
MODEL_CHECKPOINTS_DIR = "./model_checkpoints"

# RUN_SETTINGS = {
#     "model_name": "mnist_ffnn_dense_50_batchnorm",
#     "num_epochs": 10,  # 60
#     "train_abs_samples": None,  # TODO: remove this after debugging (set to None) - then full training set is used
#     "clip_grads_norm": True,
# }

RUN_SETTINGS = {
    "model_name": "mnist_ffnn_dense_50_batchnorm",
    "num_epochs": 2,  # 60
    "train_abs_samples": 200,  # TODO: remove this after debugging (set to None) - then full training set is used
    "clip_grads_norm": True,
}

# Filepath prefix specifies the run settings as defined above
run_suffix = "__".join([f"{key}_{str(value)}" for key, value in RUN_SETTINGS.items()])
run_suffix = run_suffix.replace(".", "_")  # If any values are floats, replace "." with "_" for filename

# Load MNIST dataset
features, labels = load_mnist()

# Preprocess MNIST dataset
features, labels = preprocess_mnist(features, labels)

# Define network architecture as a series of layers
# architecture = [
#         Dense(50),
#         Relu(),
#         Dense(10),
#         Softmax(),
#     ]
architecture = [
    Dense(50),
    BatchNorm(),
    Relu(),
    Dense(10),
    BatchNorm(),
    Softmax(),
]

# Define training task
training_task = TrainingTask(
    optimiser=GradientDescentOptimiser(learning_rate=0.01),
    cost=CategoricalCrossentropyCost(),
    metrics=[CategoricalCrossentropyCost(), AccuracyMetric()],
    clip_grads_norm=True,
)

# Initialise model
# TODO: want to separate out coupling between model and training task? Where best to instantiate clip_grads_norm?
model = SeriesModel(
    layers=architecture,
    clip_grads_norm=training_task.clip_grads_norm,
)

# Define evaluation task
evaluation_task = EvaluationTask(
    metrics=[CategoricalCrossentropyCost(), AccuracyMetric()],
)

# Instantiate a model saver
# TODO: replace attribute store of metrics within train/eval task with dicts, rather than lists. Then can access with
#   names rather than indices
model_saver = ModelSaveTask(
    monitoring_task=evaluation_task,
    metric_type=CategoricalCrossentropyCost(),
    save_every_n_iters=10,
    save_dir=MODEL_CHECKPOINTS_DIR,
    save_filename=run_suffix,
)

# Run training loop
loop = Loop(
    dataset=(features, labels),
    model=model,
    training_task=training_task,
    evaluation_task=evaluation_task,
    model_save_task=model_saver,
)

loop.run(
    n_epochs=RUN_SETTINGS["num_epochs"],
    batch_size=32,
    train_abs_samples=RUN_SETTINGS["train_abs_samples"],
    verbose=1,
    log_grads=True,
)

# Plot training and evaluation metric logs
for metric in loop.training_task.metric_log.keys():
    save_metric_logs(
        loop.training_task.metric_log,
        loop.evaluation_task.metric_log,
        metric,
        metric_log_dir=DATA_CACHE_DIR,
        run_config_suffix=run_suffix
    )

    plot_metric_logs_vs_gradient_norms(
        loop.training_task.metric_log,
        loop.evaluation_task.metric_log,
        metric,
        loop,
        plot_dir=PLOTS_DIR,
        run_config_suffix=run_suffix,
    )

# Look at some predictions on the evaluation set
predictions = loop.model.forward_pass(loop.features_val, mode="infer")
save_filepath = f"{PLOTS_DIR}/sampled_predictions__{run_suffix}.png"
show_digit_samples(loop.features_val, loop.labels_val, predictions, m_samples=10, save_filepath=save_filepath)
