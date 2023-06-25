"""
This file contains the SeriesModel class which is a class that represents a neural network. It is a series of layers
"""
from typing import Union
import pickle

import numpy as np

from supervised_learning.low_level_implementations.feedforward_nn.layers import Dense, BatchNorm
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

    def forward_pass(self, network_input_x: np.array, mode: str = "train"):
        """

        Args:
            network_input_x: Input data to the network. Shape (n_features, n_examples)
            mode: One of {"train", "infer"}. Certain layers may behave differently depending on the mode. For example,
            batch normalisation layers will update their running mean and variance during training, but will use the
            previously calculated running mean and variance during inference.

        Returns:
            np.array: Output of the network. Shape (n_classes, n_examples)

        """

        # TODO: replace this with __call__?

        if not self.weights_initialised:
            # Pass feature dimension if not already initialised
            self.initialise_weights(network_input_x.shape[0])
            self.weights_initialised = True

        # Input to first layer is the network input
        activation = network_input_x

        # Iterate through layers and perform forward pass
        for layer in self.layers:
            activation = layer(activation, method="forward", mode=mode)

        return activation

    def backward_pass(self, grad_cost_dyhat, log_grads=False):
        # Initial gradient, coming from the cost function
        grad = grad_cost_dyhat

        # Initialise dense layer number as number of dense layers (iterating backwards). -1 to keep indexing Pythonic
        dense_layer_num = sum(isinstance(layer, Dense) for layer in self.layers) - 1
        batch_norm_layer_num = sum(isinstance(layer, BatchNorm) for layer in self.layers) - 1

        for layer in reversed(self.layers):

            # Perform backward pass
            grad = layer(grad, method="backward", clip_grads_norm=self.clip_grads_norm)

            # Log gradients of model parameters (dW, db) if specified
            if log_grads and isinstance(layer, Dense):
                self.grads[f"Dense_{dense_layer_num}_grad_weights"] = layer.grad_weights
                self.grads[f"Dense_{dense_layer_num}_grad_bias"] = layer.grad_bias
                dense_layer_num -= 1

            # Log gradients of model parameters (dgamma, dbeta) if BatchNorm layers are present
            if log_grads and isinstance(layer, BatchNorm):
                self.grads[f"BatchNorm_{batch_norm_layer_num}_grad_gamma"] = layer.grad_gamma
                self.grads[f"BatchNorm_{batch_norm_layer_num}_grad_beta"] = layer.grad_beta
                batch_norm_layer_num -= 1

        return grad

    def update_weights_biases(self, optimiser: GradientDescentOptimiser):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = optimiser.update(layer.weights, layer.grad_weights)
                layer.bias = optimiser.update(layer.bias, layer.grad_bias)

            if isinstance(layer, BatchNorm):
                layer.gamma = optimiser.update(layer.gamma, layer.grad_gamma)
                layer.beta = optimiser.update(layer.beta, layer.grad_beta)

    def save_checkpoint(self, path):

        # If path does not end with .pkl, add it
        if not path.endswith(".pkl"):
            path += ".pkl"

        # Save model
        with open(path, "wb") as f:
            pickle.dump(self, f)


def load_checkpoint(path):

    # Load model
    with open(path, "rb") as f:
        model = pickle.load(f)

    return model





