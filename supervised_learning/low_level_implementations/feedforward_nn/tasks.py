"""

Maths/literature to script nomenclature:
========================================

X ---> features
Y ---> labels
"""
import numpy as np
from typing import Tuple, Union, List
from tqdm import tqdm

from supervised_learning.low_level_implementations.feedforward_nn.costs_and_metrics import \
    (BaseMetric, CategoricalCrossentropyCost, AccuracyMetric)
from supervised_learning.low_level_implementations.feedforward_nn.models import SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser


def train_val_split(features, labels, train_fraction=0.8):

    # Expecting X to be of shape (n_features x m_samples) and Y to be of shape (1 x m_samples)
    m_total_samples = features.shape[1]
    idx_train_cutoff = int(m_total_samples * train_fraction)

    features_train = features[:, :idx_train_cutoff]
    features_val = features[:, idx_train_cutoff:]
    labels_train = labels[:, :idx_train_cutoff]
    labels_val = labels[:, idx_train_cutoff:]

    return features_train, features_val, labels_train, labels_val


def batch_generator(features_train_or_val, labels_train_or_val, batch_size=32, shuffle=True):

    # Get number of samples of a whole epoch. If specified, shuffle the indices of the whole epoch.
    m_samples = features_train_or_val.shape[1]
    sample_indices = [*range(m_samples)]
    if shuffle:
        np.random.shuffle(sample_indices)

    for i in range(0, m_samples, batch_size):
        batch_indices = sample_indices[i:i + batch_size]
        features_batch = features_train_or_val[:, batch_indices]
        labels_batch = labels_train_or_val[:, batch_indices]
        yield features_batch, labels_batch


class TrainingTask:
    """
    A training task defines the training architecture.

    Parameters:
        - optimiser: The optimiser to be used.
        - cost: The cost function to be used.
        - metrics: The metrics to be used and logged.

    Attributes:
        - metric_log: A dictionary of metric names and their corresponding logs.
    """

    def __init__(
            self,
            optimiser: GradientDescentOptimiser,
            cost: CategoricalCrossentropyCost,
            metrics: Union[None, List[BaseMetric]] = None,
    ):
        """
        Initialises a training task.
        """

        self.optimiser = optimiser
        self.cost = cost
        self.metrics = metrics

        self.metric_log = {metric.name: [] for metric in metrics}


class EvaluationTask:
    """
    An evaluation task defines the evaluation architecture.

    Parameters:
        - metrics: The metrics to be used and logged.

    Attributes:
        - metric_log: A dictionary of metric names and their corresponding logs.
    """

    def __init__(
            self,
            metrics: Union[None, List[BaseMetric]] = None,
    ):

        self.metrics = metrics

        self.metric_log = {metric.name: [] for metric in metrics}


class Loop:
    """
    A loop is a runs a training task and an (optional) evaluation task for a given number of epochs.

    The loop has a run method that takes in a number of epochs and runs the training and evaluation tasks for that
    number of epochs.

    Parameters:
        - dataset: A tuple of (features, labels) of the training data.
        - model: The model to be trained.
        - training_task: The training task to be run.
        - evaluation_task: The evaluation task to be run.

    Methods:
        - run: Runs the training and evaluation tasks for a given number of epochs.
    """

    def __init__(
            self,
            dataset: Tuple[np.ndarray, np.ndarray],
            model: SeriesModel,
            training_task: TrainingTask,
            evaluation_task: EvaluationTask = None,
            ):

        self.features, self.labels = dataset
        self.model = model
        self.training_task = training_task
        self.evaluation_task = evaluation_task

    def run(self, n_epochs=1, batch_size=32):

        # Split dataset into training and validation sets
        features_train, features_val, labels_train, labels_val = train_val_split(self.features, self.labels)

        # Instantiate batch generator object
        generator = batch_generator(features_train, labels_train, batch_size=batch_size)

        # Initialise epoch cost log
        cost_log = []

        # Iterate over epochs
        for epoch in tqdm(range(n_epochs), desc="Training epochs"):

            # Initialise batch cost log
            batch_cost_log = []

            # Iterate over batches
            for batch_X, batch_Y in generator:

                # =====================
                # Train
                # =====================

                # Forward pass
                predictions = self.model.forward_pass(batch_X)

                # Calculate cost
                batch_cost = self.training_task.cost(batch_Y, predictions)
                batch_cost_log.append(batch_cost)

                # Calculate dYhat
                grad_predictions = self.training_task.cost.compute_gradient(batch_Y, predictions)

                # Backward pass
                self.model.backward_pass(grad_predictions)

                # Update weights
                self.model.update_weights_biases(self.training_task.optimiser)

                # Update training task metrics
                if self.training_task.metrics is not None:
                    for metric in self.training_task.metrics:
                        metric_train = metric(batch_Y, predictions)
                        self.training_task.metric_log[metric.name].append(metric_train)

                        # Update tqdm progress bar with training metrics
                        metric_string = " | ".join(
                            [f"{metric.name}: {metric_train:.4f}" for metric in self.training_task.metrics]
                        )
                        tqdm.write(metric_string)

                # =====================
                # Evaluate
                # =====================

                if self.evaluation_task is not None:

                    # Forward pass
                    predictions_val = self.model.forward_pass(features_val)

                    # Update validation task metrics
                    if self.evaluation_task.metrics is not None:
                        for metric in self.evaluation_task.metrics:
                            metric_val = metric(labels_val, predictions_val)
                            self.evaluation_task.metric_log[metric.name].append(metric_val)

                            # Update tqdm progress bar with validation metrics
                            metric_string = " | ".join(
                                [f"{metric.name}: {metric_val:.4f}" for metric in self.evaluation_task.metrics]
                            )
                            tqdm.write(metric_string)

