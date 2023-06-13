"""
A feedforward neural network implementation using numpy.



Maths to code
=============

X --> input_X
Z --> preactivation_Z
A --> activation_A
Yhat --> prediction_Yhat
Y --> label_Y
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
        self.layer_input = np.array(None)
        self.layer_output = np.array(None)

        # Back prop edges in and out
        self.grad_input_from_right = np.array(None)
        self.grad_output_to_left = np.array(None)

    def __repr__(self):
        """
        When print called on the object, return 'Layer: <class layer name>'.
        This can be added to in subclass (specific layer) representations
        """
        return f"""
        ========================================================
        Layer: {self.__class__.__name__}
        Input shape: {self.layer_input.shape}. Output shape: {self.layer_output.shape}
        """

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
        self.layer_output = layer_input


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
        # self.layer_output = None

        # Gradient from right neighbour
        self.grad_input_from_right = right_neighbour.grad_output_to_left
        # self.grad_output_to_left = None

        # Parameters passed in on instantiation
        self.n_neurons = n_neurons
        self.weight_init_scale = weight_init_scale

        # Initialise weights (functions do this in-place)
        self.weights = None
        self.bias = None
        self.initialise_weights()
        self.initialise_bias()

        # Initialise grad of weights and bias vector. Same shape as weights and bias, so just copy initialised
        self.grad_weights = self.weights.copy()
        self.grad_bias = self.bias.copy()

    def __repr__(self):
        superclass_repr = super().__repr__()
        dense_stub = f"""
        Weights shape: ({self.weights.shape}). Bias shape: ({self.bias.shape})
        """
        return superclass_repr + dense_stub

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
        """
        This function takes as input dC/dZ_l

        Returns:

        """
        # TODO
        # dC/dA_l-1
        self.grad_output_to_left = None  # TODO
        # dC/dW_l
        self.grad_weights = None    # TODO
        # dC/db_l
        self.grad_bias = None  # TODO


class Relu(Layer):
    """

    """

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        # TODO
        pass

    def backward_pass(self):
        # TODO
        pass


# Softmax
class Softmax(Layer):
    """

    """

    def __init__(self):
        super().__init__()


class Series:

    def __init__(self, layers=None):
        """

        Args:
            layers (List or None): optional argument to specify network in list form.
            e.g. [Dense(2), Relu(), Dense(2), Softmax()]
        """
        if layers is None:
            layers = []
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward_pass(self):
        activations = None    # TODO
        return activations

    def connect_forward(self):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                continue
            layer.left_neighbour = self.layers[idx - 1]
            layer.layer_input = self.layers[idx - 1].layer_output

    def connect_backward(self):
        reversed_list = self.layers[::-1]    # [::-1] reverses list to start with final layer
        for idx, layer in enumerate(reversed_list):
            if idx == 0:
                continue
            layer.right_neighbour = reversed_list[idx - 1]
            layer.grad_input_from_right = reversed_list[idx - 1].grad_output_to_left

    def __call__(self, input_array):

        # Turn np.array `input_array` into layer object Input, and append to front of layer list
        self.layers = [Input(input_array)] + self.layers

        # Connect up network
        self.connect_forward()
        self.connect_backward()

