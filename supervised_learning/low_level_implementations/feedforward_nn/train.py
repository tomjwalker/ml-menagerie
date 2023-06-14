"""

Maths/literature to script nomenclature:
========================================

X ---> features
Y ---> labels
"""
import numpy as np

from supervised_learning.low_level_implementations.feedforward_nn.models import SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.losses import CategoricalCrossentropyCost
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser


def train_val_split(features, labels, val_fraction=0.2):

    # Expecting X to be of shape (n_features x m_samples) and Y to be of shape (1 x m_samples)
    m_total_samples = features.shape[1]
    idx_train_cutoff = int(m_total_samples * (1 - val_fraction))

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
