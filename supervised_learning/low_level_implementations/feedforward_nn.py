"""
A feedforward neural network implementation using numpy.



Maths to code
=============

X --> input_X
Z --> output_Z
A --> activation_A
Yhat -->
Y -->
W --> weights_W
B --> biases_B
"""

import numpy as np


class Layer:
    """

    """

    def __init__(self):

        # Layer (node) neighbours
        self.left_neighbour = None
        self.right_neighbour = None

        # Forward prop edges in and out
        self.layer_input = None
        self.layer_output = None

        # Back prop edges in and out
        self.grad_input_from_right = None
        self.grad_output_to_left = None

    def __repr__(self):
        """
        When print called on the object, return 'Layer: <class layer name>'.
        This can be added to in subclass (specific layer) representations
        """
        return f"Layer: {self.__class__.__name__}. "

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass


class Input(Layer):
    """Ensures all neighbours are the same type when constructing a Model"""

    def __init__(self, layer_input):
        """
        Args:
            layer_input (numpy.array): The input matrix to the neural network, of shape (n_features x m_samples)
        """

        super().__init__()

        # Forward prop edges in and out
        self.layer_input = layer_input
        self.layer_output = None


class Dense(Layer):

    def __init__(self, left_neighbour, right_neighbour, n_neurons, weight_init_scale=0.01):
        """
        Initialise layer

        Args:
            left_neighbour (Layer): upstream Layer neighbour (graph defined in the Series model constructor object)
            n_neurons (int): number of neurons in layer
            weight_init_scale (float): Sets magnitude of randomised weight initialisation
        """

        super().__init__()

        # Feedforward connections in and out
        self.layer_input = left_neighbour.layer_output
        self.layer_output = None

        # Gradient from right neighbour
        self.grad_input_from_right = right_neighbour.grad_output_to_left
        self.grad_output_to_left = None

        # Parameters passed in on instantiation
        self.n_neurons = n_neurons
        self.weight_init_scale = weight_init_scale

        # Initialise weights (functions do this in-place)
        self.weights = None
        self.bias = None
        self.initialise_weights()
        self.initialise_bias()

    def __repr__(self):
        # TODO
        pass

    def initialise_weights(self):
        """
        Random initialisation of weights to "break symmetry" between hidden units
        """

        self.weights = np.random.randn(self.n_neurons, self.layer_input) * self.weight_init_scale

    def initialise_bias(self):
        self.bias = np.zeros((self.n_neurons, 1))

    def forward_pass(self):
        self.layer_output = np.dot(self.weights, self.layer_input.shape[0]) + self.bias

    def backward_pass(self):
        # TODO
        pass


class Relu(Layer):
    """

    """

    def __init__(self):
        super().__init__()

# Softmax
class Softmax(Layer):
    """

    """

    def __init__(self):
        super().__init__()


class Series:

    def __init__(self):
        self.layers = []


    def add(self, layer):
        self.layers.append(layer)


    def forward_pass(self, inputs):
        activations = None    # TODO
        return activations

