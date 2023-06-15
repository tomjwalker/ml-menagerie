"""

Maths/literature to script nomenclature:
========================================

X ---> features
Y ---> labels
"""
import numpy as np

from supervised_learning.low_level_implementations.feedforward_nn.costs import \
    CategoricalCrossentropyCost
from supervised_learning.low_level_implementations.feedforward_nn.models import \
    SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import \
    GradientDescentOptimiser


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
    A training task is a task that trains a model on a given dataset.

    The training task is initialised with a model, an optimiser and a cost function.

    The training task has a train method that takes in a dataset and trains the model on it.

    Parameters:
        - model: The model to be trained.
        - optimiser: The optimiser to be used for training.
        - cost: The cost function to be used for training.

    Attributes:
        - model: The model to be trained.
        - optimiser: The optimiser to be used for training.
        - cost: The cost function to be used for training.

    Methods:
        - train: Trains the model on a given dataset.

    """

    def __init__(self, model: SeriesModel, optimiser: GradientDescentOptimiser, cost: CategoricalCrossentropyCost):

        self.model = model
        self.optimiser = optimiser
        self.cost = cost

    def train(self, features_train, labels_train, n_epochs=1, batch_size=32):

        # Instantiate batch generator object
        generator = batch_generator(features_train, labels_train, batch_size=batch_size)

        # Initialise epoch cost log
        cost_log = []

        # Iterate over epochs
        for epoch in range(n_epochs):

            # Initialise batch cost log
            batch_cost_log = []

            # Iterate over batches
            for batch_X, batch_Y in generator:

                # Forward pass
                predictions = self.model.forward_pass(batch_X)

                # Calculate cost
                batch_cost = self.cost.compute_cost(batch_Y, predictions)
                batch_cost_log.append(batch_cost)

                # Calculate dYhat
                grad_predictions = self.cost.compute_gradient(batch_Y, predictions)

                # Backward pass
                self.model.backward_pass(grad_predictions)

                # Update weights
                self.model.update_weights_biases(self.optimiser)

            cost_log.append(batch_cost)


class EvaluationTask:
    """
    An evaluation task is a task that evaluates a model on a given dataset.

    The evaluation task is initialised with a model and a cost function.

    The evaluation task has an evaluate method that takes in a dataset and evaluates the model on it.

    Parameters:
        - model: The model to be evaluated.
        - cost: The cost function to be used for evaluation.
    """

    def __init__(self, model: SeriesModel, cost: CategoricalCrossentropyCost):

        self.model = model
        self.cost = cost

    def evaluate(self, features, labels):

        # Forward pass
        predictions = self.model.forward_pass(features)


class Loop:
    """
    A loop is a loop that runs a training task and an evaluation task for a given number of epochs.

    The loop is initialised with a training task, an evaluation task and a dataset.

    The loop has a run method that takes in a number of epochs and runs the training and evaluation tasks for that
    number of epochs.

    Parameters:
        - training_task: The training task to be run.
        - evaluation_task: The evaluation task to be run.
        - features: The features of the dataset.
        - labels: The labels of the dataset.

    Attributes:
        - training_task: The training task to be run.
        - evaluation_task: The evaluation task to be run.
        - features: The features of the dataset.
        - labels: The labels of the dataset.

    Methods:
        - run: Runs the training and evaluation tasks for a given number of epochs.
    """

    def __init__(self, training_task: TrainingTask, evaluation_task: EvaluationTask, features, labels):

        self.training_task = training_task
        self.evaluation_task = evaluation_task
        self.features = features
        self.labels = labels

    def run(self, n_epochs=1, batch_size=32):

        # Split dataset into training and validation sets
        features_train, features_val, labels_train, labels_val = train_val_split(self.features, self.labels)

        # Run training task
        self.training_task.train(features_train, labels_train, n_epochs=n_epochs, batch_size=batch_size)

        # Run evaluation task
        self.evaluation_task.evaluate(features_val, labels_val)