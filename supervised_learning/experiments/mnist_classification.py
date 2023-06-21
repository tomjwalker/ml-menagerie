import os

import numpy as np

from supervised_learning.datasets.mnist.data_utils import load_mnist, preprocess_mnist, show_digit_samples
from supervised_learning.low_level_implementations.feedforward_nn.costs_and_metrics import (
    CategoricalCrossentropyCost, AccuracyMetric)
from supervised_learning.low_level_implementations.feedforward_nn.layers import Dense, Relu, Softmax
from supervised_learning.low_level_implementations.feedforward_nn.models import SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser
from supervised_learning.low_level_implementations.feedforward_nn.tasks import (TrainingTask, EvaluationTask, Loop,
                                                                                train_val_split)
import matplotlib.pyplot as plt


def plot_metric_logs(metric_log_training, metric_log_evaluation, metric_name):
    """
    Plots outputs of the training and evaluation metric logs during the training loop.
    """
    plt.plot(metric_log_training[metric_name], label='Training')
    plt.plot(metric_log_evaluation[metric_name], label='Evaluation')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


def save_metric_logs(metric_log_training, metric_log_evaluation, metric_name, metric_log_dir="./data_cache"):
    """
    Saves outputs of the training and evaluation metric logs during the training loop to a subdirectory within the
    current working directory.
    """

    # If data_cache directory does not exist, create it
    if not os.path.exists(metric_log_dir):
        os.makedirs(metric_log_dir)

    np.save(f'{metric_log_dir}/{metric_name}_training_log.npy', metric_log_training[metric_name])
    np.save(f'{metric_log_dir}/{metric_name}_evaluation_log.npy', metric_log_evaluation[metric_name])


# ========================================
# Main script
# ========================================

# Load MNIST dataset
features, labels = load_mnist()

# Preprocess MNIST dataset
features, labels = preprocess_mnist(features, labels)

# Show a few samples
# TODO: uncomment after debugging
# show_digit_samples(features, labels, m_samples=10)

# Define network architecture as a series of layers
architecture = [
        Dense(50),
        Relu(),
        Dense(10),
        Softmax(),
    ]
# Initialise model
model = SeriesModel(
    layers=architecture,
)

print(model)

# Define training task
training_task = TrainingTask(
    optimiser=GradientDescentOptimiser(learning_rate=0.01),
    cost=CategoricalCrossentropyCost(),
    metrics=[CategoricalCrossentropyCost(), AccuracyMetric()],
)

# Define evaluation task
evaluation_task = EvaluationTask(
    metrics=[CategoricalCrossentropyCost(), AccuracyMetric()],
)

# Run training loop
loop = Loop(
    dataset=(features, labels),
    model=model,
    training_task=training_task,
    evaluation_task=evaluation_task,
)

# TODO: remove train_abs_samples after debugging
# loop.run(n_epochs=3, batch_size=32)
loop.run(n_epochs=50, batch_size=16, train_abs_samples=100, verbose=1)

print(loop.training_task.metric_log)
print(loop.evaluation_task.metric_log)


# Plot training and evaluation metric logs
for metric in loop.training_task.metric_log.keys():
    plot_metric_logs(loop.training_task.metric_log, loop.evaluation_task.metric_log, metric)
    save_metric_logs(loop.training_task.metric_log, loop.evaluation_task.metric_log, metric)


