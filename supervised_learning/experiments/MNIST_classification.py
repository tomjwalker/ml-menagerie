import numpy as np

from supervised_learning.datasets.mnist.data_utils import load_mnist, preprocess_mnist, show_digit_samples
from supervised_learning.low_level_implementations.feedforward_nn.costs import CategoricalCrossentropyCost
from supervised_learning.low_level_implementations.feedforward_nn.layers import Dense, Input, Relu, Softmax
from supervised_learning.low_level_implementations.feedforward_nn.models import SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser
from supervised_learning.low_level_implementations.feedforward_nn.tasks import (TrainingTask, batch_generator,
                                                                                train_val_split)

# Load MNIST dataset
features, labels = load_mnist()

# Preprocess MNIST dataset
features, labels = preprocess_mnist(features, labels)

# Show a few samples
show_digit_samples(features, labels, m_samples=10)

# Split into train and validation sets
features_train, features_val, labels_train, labels_val = train_val_split(features, labels, train_fraction=0.8)

# Define network architecture as a series of layers
architecture = [
        Input(features_train),
        Dense(5),
        Relu(),
        Dense(10),
        Softmax(),
    ]

# Initialise model
model = SeriesModel(
    layers=architecture,
)

print(model)

