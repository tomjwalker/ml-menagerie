"""
A feedforward neural network implementation using numpy.


Nomenclature (use these symbols in comments to keep track of equations. Code uses the object parameters)
========================================================================================================

X --> input
Z --> preactivation
A --> activation
Yhat --> prediction
Y --> label
W --> weights
B --> biases


Gradients during backprop: following NN literature convention that e.g. dA_l represents del(Cost)/del(A_l)
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

        # Store this dimension (number of samples in (mini-) batch. Used in backprop calculations
        self.m_samples = self.layer_input.shape[-1]

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
        This function takes as input del(J)/del(Z_l) = dZ_l (cost w.r.t layer output) and updates grads for:
        - dA_l-1 --> Cost w.r.t. layer input
        - dW_l --> Cost w.r.t. layer weight matrix
        - db_l --> Cost w.r.t. layer bias vector
        """
        # del(J)/del(A_l-1) --> dA_1 = np.dot(W_2.T, dZ_2)
        # The order of matmul and inclusion of transpose can be determined by considering d<param> and <param> have same
        # shapes, and looking at shapes at both sides of equality
        self.grad_output_to_left = np.dot(self.weights.T, self.grad_input_from_right)
        assert self.grad_output_to_left == self.layer_input    # Check shape(dA_l-1) == shape(A_l-1)

        # del(J)/del(W_l) = dW_l = dZ_l . del(Z_l)/del(W_l) = (1/m_samples) * np.dot(dZ_2, A_1.T).
        # Matmul commuted with A transpose to ensure dW_l and W_l have same shape
        self.grad_weights = (1 / self.m_samples) * np.dot(self.grad_input_from_right, self.layer_input)
        assert self.grad_weights == self.weights  # Check shape(dW_l) == shape(W_l)

        # del(J)/del(b_l) =  del(J)/del(Z_l) . del(Z_l)/del(b_l) --> dZ_l . 1.
        # The sum(...axis=1) sums over the sample dimension for dZ_l. The (1 / m_samples) then ensures values are
        # average bias grads over the samples
        self.grad_bias = (1 / self.m_samples) * np.sum(self.grad_input_from_right, axis=1, keepdims=True)
        assert self.grad_bias == self.bias  # Check shape(dW_l) == shape(W_l)


class Relu(Layer):
    """

    """

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        # A = ReLU(Z)
        self.layer_output = np.maximum(0, self.layer_input)

    def backward_pass(self):
        # dZ = dA * ReLU'(Z). ReLU'(Z) becomes a binary mask: 1 where Z>0, 0 otherwise
        self.grad_output_to_left = self.grad_input_from_right * (self.layer_input > 0)


# Softmax
class Softmax(Layer):
    """

    """

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        # A = Softmax(Z)
        self.layer_output = np.exp(self.layer_input) / np.sum(np.exp(self.layer_input), axis=0, keepdims=True)

    def backward_pass(self):
        # dZ = dA * softmax'(Z) = = dA * A * (1 - A)
        # NB: '*' signifies elementwise multiplication
        self.grad_output_to_left = self.grad_output_from_right * \
                                   self.layer_output * (1 - self.layer_output)


# Use to calculate dJ/dA, to pass in as input to Softmax backprop
class Output(Layer):
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
        activations = None  # TODO
        return activations

    def connect_forward(self):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                continue
            layer.left_neighbour = self.layers[idx - 1]
            layer.layer_input = self.layers[idx - 1].layer_output

    def connect_backward(self):
        reversed_list = self.layers[::-1]  # [::-1] reverses list to start with final layer
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
