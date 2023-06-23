"""
This file contains the SeriesModel class which is a class that represents a neural network. It is a series of layers
"""
from typing import Union

import numpy as np

from supervised_learning.low_level_implementations.feedforward_nn.layers import Dense
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser
from supervised_learning.low_level_implementations.feedforward_nn.utils import ClipNorm


class SeriesModel:

    def __init__(self, layers=None, clip_grads_norm: Union[None, ClipNorm] = None):
        """

        Args:
            layers (List or None): optional argument to specify network in list form.
            e.g. [Dense(2), Relu(), Dense(2), Softmax()]
        """

        # Avoid having empty list in initialisation signature
        if layers is None:
            layers = []
        self.layers = layers

        self.input_shape = None
        self.weights_initialised = False

        # Optional list for storing gradients of weights and biases, if specified in backward pass method
        self.grads = {}

        # Clip grad settings
        self.clip_grads_norm = clip_grads_norm

    def __repr__(self):

        # Iterate through all __repr__ methods of layers within the network and concatenate. Add a newline after each
        # layer
        repr_str = ""
        for layer in self.layers:
            repr_str += layer.__repr__()
        return repr_str

    def add(self, layer):
        self.layers.append(layer)

    def initialise_weights(self, num_units_prev_layer):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.initialise_weights(num_units_prev_layer)
                layer.initialise_bias()
                num_units_prev_layer = layer.n_neurons

    def forward_pass(self, network_input_x: np.array):

        # TODO: replace this with __call__?

        if not self.weights_initialised:
            # Pass feature dimension if not already initialised
            self.initialise_weights(network_input_x.shape[0])
            self.weights_initialised = True

        # Input to first layer is the network input
        activation = network_input_x

        # Iterate through layers and perform forward pass
        for layer in self.layers:
            activation = layer(activation, method="forward")

        return activation

    def backward_pass(self, grad_cost_dyhat, log_grads=False):
        # Initial gradient, coming from the cost function
        grad = grad_cost_dyhat

        # Initialise dense layer number as number of dense layers (iterating backwards). -1 to keep indexing Pythonic
        dense_layer_num = sum(isinstance(layer, Dense) for layer in self.layers) - 1

        for layer in reversed(self.layers):

            # Perform backward pass
            grad = layer(grad, method="backward", clip_grads_norm=self.clip_grads_norm)

            # Log gradients of model parameters (dW, db) if specified
            if log_grads and isinstance(layer, Dense):
                self.grads[f"Dense_{dense_layer_num}_weights"] = layer.grad_weights
                self.grads[f"Dense_{dense_layer_num}_bias"] = layer.grad_bias

                dense_layer_num -= 1

        return grad

    def update_weights_biases(self, optimiser: GradientDescentOptimiser):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = optimiser.update(layer.weights, layer.grad_weights)
                layer.bias = optimiser.update(layer.bias, layer.grad_bias)
