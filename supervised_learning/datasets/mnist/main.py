"""
A demonstration of the MNIST dataset and how to load it.
"""


import numpy as np

from supervised_learning.datasets.mnist.data_utils import load_mnist, preprocess_mnist, show_digit_samples

# Load MNIST dataset
features, labels = load_mnist()

# Preprocess MNIST dataset
features, labels = preprocess_mnist(features, labels)

# Generate dummy predictions for the next step. They are the output of a softmax layer, so they are normalised and
# same shape as labels
predictions = np.random.rand(*labels.shape)

# Show some samples
show_digit_samples(features, labels, predictions=predictions, m_samples=10)
