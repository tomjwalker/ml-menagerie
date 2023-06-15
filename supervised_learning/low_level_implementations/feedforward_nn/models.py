"""
This file contains the SeriesModel class which is a class that represents a neural network. It is a series of layers
"""
import numpy as np

from supervised_learning.low_level_implementations.feedforward_nn.layers import (
    Dense, Input)
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import \
    GradientDescentOptimiser


class SeriesModel:

    def __init__(self, layers=None):
        """

        Args:
            layers (List or None): optional argument to specify network in list form.
            e.g. [Dense(2), Relu(), Dense(2), Softmax()]
        """

        # Avoid having empty list in initialisation signature
        if layers is None:
            layers = []
        self.layers = layers

        # Initialise weights
        self.initialise_weights()

    def __repr__(self):

        # Iterate through all __repr__ methods of layers within the network and concatenate. Add a newline after each
        # layer
        repr_str = ""
        for layer in self.layers:
            repr_str += layer.__repr__()
        return repr_str

    def add(self, layer):
        self.layers.append(layer)

    def initialise_weights(self):
        prev_neurons = None
        for layer in self.layers:
            if isinstance(layer, Input):
                prev_neurons = layer.network_input_x.shape[0]
            if isinstance(layer, Dense):
                layer.initialise_weights(prev_neurons)
                layer.initialise_bias()
                prev_neurons = layer.n_neurons

    def forward_pass(self, network_input_x: np.array):

        # Input to first layer is the network input
        activation = network_input_x

        # Iterate through layers and perform forward pass
        for layer in self.layers:
            activation = layer(activation, method="forward")

        return activation

    def backward_pass(self, grad_cost_dyhat):
        grad = grad_cost_dyhat
        for layer in reversed(self.layers):
            grad = layer(grad, method="backward")
        return grad

    def update_weights_biases(self, optimiser: GradientDescentOptimiser):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = optimiser.update(layer.weights, layer.grad_weights)
                layer.bias = optimiser.update(layer.bias, layer.grad_bias)
