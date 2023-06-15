"""

Maths/literature to script nomenclature:
========================================

X ---> features
Y ---> labels
"""
import numpy as np
from typing import Tuple, Union, List

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

    It collects together:
    - The training data
    - the loss function to be used
    - the optimiser to be used
    - optional metrics to be used
    - when to log checkpoints

    Parameters:
        - training_data: A tuple of (features, labels) of the training data.
        - model: The model to be trained.
        - optimiser: The optimiser to be used for training.
        - cost: The cost function to be used for training.
        - metrics (optional): A list of metrics to be used for training.

    Attributes:
        - features: The features of the training data.
        - labels: The labels of the training data.

    Methods:
        - train: Trains the model on a given dataset.

    """

    def __init__(
            self,
            training_data: Tuple[np.array, np.array],
            model: SeriesModel,
            optimiser: GradientDescentOptimiser,
            cost: CategoricalCrossentropyCost,
            metrics: Union[None, List[BaseMetric]] = None,
    ):
        """
        Initialises a training task.

        Args:
            training_data:
            model:
            optimiser:
            cost:
            metrics:
        """

        self.features = training_data[0]
        self.labels = training_data[1]
        self.model = model
        self.optimiser = optimiser
        self.cost = cost
        self.metrics = metrics

    # def train(self, features_train, labels_train, n_epochs=1, batch_size=32):
    #
    #     # Instantiate batch generator object
    #     generator = batch_generator(features_train, labels_train, batch_size=batch_size)
    #
    #     # Initialise epoch cost log
    #     cost_log = []
    #
    #     # Iterate over epochs
    #     for epoch in range(n_epochs):
    #
    #         # Initialise batch cost log
    #         batch_cost_log = []
    #
    #         # Iterate over batches
    #         for batch_X, batch_Y in generator:
    #
    #             # Forward pass
    #             predictions = self.model.forward_pass(batch_X)
    #
    #             # Calculate cost
    #             batch_cost = self.cost(batch_Y, predictions)
    #             batch_cost_log.append(batch_cost)
    #
    #             # Calculate dYhat
    #             grad_predictions = self.cost.compute_gradient(batch_Y, predictions)
    #
    #             # Backward pass
    #             self.model.backward_pass(grad_predictions)
    #
    #             # Update weights
    #             self.model.update_weights_biases(self.optimiser)
    #
    #         cost_log.append(batch_cost_log)


class EvaluationTask:
    """
    An evaluation task defines the evaluation architecture.

    It collects together:
    - The evaluation data
    - metrics to be used and logged

    """

    def __init__(
            self,
            validation_data: Tuple[np.array, np.array],
            model: SeriesModel,
            metrics: Union[None, List[BaseMetric]] = None,
    ):

        self.features = validation_data[0]
        self.labels = validation_data[1]

        self.model = model
        self.metrics = metrics

        self.metric_log = {metric.__name__: [] for metric in metrics}
    #
    # def evaluate(self, features, labels):
    #
    #     # Forward pass
    #     predictions = self.model.forward_pass(features)
    #
    #     #


class Loop:
    """
    A loop is a runs a training task and an (optional) evaluation task for a given number of epochs.

    The loop has a run method that takes in a number of epochs and runs the training and evaluation tasks for that
    number of epochs.

    Parameters:
        - model: The model to be trained.
        - training_task: The training task to be run.
        - evaluation_task: The evaluation task to be run.

    Methods:
        - run: Runs the training and evaluation tasks for a given number of epochs.
    """

    def __init__(
            self,
            model: SeriesModel,
            training_task: TrainingTask,
            evaluation_task: EvaluationTask = None,
            ):

        self.model = model
        self.training_task = training_task
        self.evaluation_task = evaluation_task

    def run(self, n_epochs=1, batch_size=32):

        # Split dataset into training and validation sets
        features_train, features_val, labels_train, labels_val = train_val_split(self.features, self.labels)

        # Run training task
        self.training_task.train(features_train, labels_train, n_epochs=n_epochs, batch_size=batch_size)

        # Run evaluation task
        if self.evaluation_task is not None:
            self.evaluation_task.evaluate(features_val, labels_val)