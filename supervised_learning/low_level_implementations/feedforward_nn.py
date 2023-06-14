"""
A feedforward neural network implementation using numpy.


Nomenclature (use these symbols in comments to keep track of equations. Code uses the object parameters)
========================================================================================================

X --> input
Z --> preactivation
A --> activation
A_(l-1) ---> prev_activation (w.r.t. Z_(l), A_(l) in a dense layer)
Yhat --> prediction
Y --> label
W --> weights
B --> biases


Gradients during backprop: following NN literature convention that e.g. dA_l represents del(Cost)/del(A_l)
dZ --> grad_preactivation
dA ---> grad_activation
...etc...

"""

import numpy as np


class Layer:
    """

    """

    def __init__(self):
        pass

    def __repr__(self):
        """
        When print called on the object, return 'Layer: <class layer name>'.
        This can be added to in subclass (specific layer) representations
        """
        return f"""
        ========================================================
        Layer: {self.__class__.__name__}
        """

    def __call__(self, input_activation_or_grad, method):
        if method == "forward":
            return self.forward_pass(input_activation_or_grad)
        if method == "backward":
            return self.backward_pass(input_activation_or_grad)
        raise ValueError("Invalid method; should be an element of {'forward', 'backward'}")

    def forward_pass(self, input_activation_from_left):
        pass

    def backward_pass(self, input_grad_from_right):
        pass


class Dense(Layer):

    def __init__(self, n_neurons, weight_init_scale=0.01):
        """
        Initialise layer

        Args:
            n_neurons (int): number of neurons in layer
            weight_init_scale (float): Sets magnitude of randomised weight initialisation
        """

        super().__init__()

        # Feedforward connections in and out
        self.layer_input = None

        # TODO: don't recalculate this on every pass
        self.m_samples = None

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
        num_params = self.weights.size + self.bias.size
        dense_stub = f"""
        Weights shape: ({self.weights.shape}). Bias shape: ({self.bias.shape}. # trainable params: {num_params})
        """
        return superclass_repr + dense_stub

    def initialise_weights(self):
        """
        Random initialisation of weights to "break symmetry" between hidden units
        """

        self.weights = np.random.randn(self.n_neurons, self.layer_input) * self.weight_init_scale

    def initialise_bias(self):
        self.bias = np.zeros((self.n_neurons, 1))

    def forward_pass(self, prev_layer_activation):
        """

        Args:
            prev_layer_activation: A_(l-1)

        Returns:
            layer_preactivation --> Z_(l)

        """

        layer_preactivation = np.dot(self.weights, prev_layer_activation) + self.bias

        # Cache input for backprop
        self.layer_input = prev_layer_activation

        # TODO: pass this in from Series
        # Store this dimension (number of samples in (mini-) batch. Used in backprop calculations
        self.m_samples = self.layer_input.shape[-1]

        return layer_preactivation

    def backward_pass(self, grad_layer_preactivation):
        """
        This function takes as input del(J)/del(Z_l) = dZ_l (cost w.r.t layer output) and updates grads for:
        - dA_l-1 --> Cost w.r.t. layer input
        - dW_l --> Cost w.r.t. layer weight matrix
        - db_l --> Cost w.r.t. layer bias vector
        """

        # Update weights and biases

        # del(J)/del(W_l) = dW_l = dZ_l . del(Z_l)/del(W_l) = (1/m_samples) * np.dot(dZ_2, A_1.T).
        # Matmul commuted with A transpose to ensure dW_l and W_l have same shape
        self.grad_weights = (1 / self.m_samples) * np.dot(grad_layer_preactivation, self.layer_input)
        assert self.grad_weights == self.weights  # Check shape(dW_l) == shape(W_l)

        # del(J)/del(b_l) =  del(J)/del(Z_l) . del(Z_l)/del(b_l) --> dZ_l . 1.
        # The sum(...axis=1) sums over the sample dimension for dZ_l. The (1 / m_samples) then ensures values are
        # average bias grads over the samples
        self.grad_bias = (1 / self.m_samples) * np.sum(grad_layer_preactivation, axis=1, keepdims=True)
        assert self.grad_bias == self.bias  # Check shape(dW_l) == shape(W_l)

        # Update grad for prev layer activation (output left for backprop)

        # del(J)/del(A_l-1) --> dA_1 = np.dot(W_2.T, dZ_2)
        # The order of matmul and inclusion of transpose can be determined by considering d<param> and <param> have same
        # shapes, and looking at shapes at both sides of equality
        grad_prev_layer_activation = np.dot(self.weights.T, grad_layer_preactivation)
        assert grad_prev_layer_activation == self.layer_input    # Check shape(dA_l-1) == shape(A_l-1)

        return grad_prev_layer_activation


class Relu(Layer):
    """

    """

    def __init__(self):
        super().__init__()
        self.layer_input = None

    def forward_pass(self, layer_preactivation):

        # A = ReLU(Z)
        layer_activation = np.maximum(0, layer_preactivation)

        # Cache for backprop
        self.layer_input = layer_activation

        return layer_activation

    def backward_pass(self, grad_layer_activation):

        # dZ = dA * ReLU'(Z). ReLU'(Z) becomes a binary mask: 1 where Z>0, 0 otherwise
        grad_layer_preactivation = grad_layer_activation * (self.layer_input > 0)

        return grad_layer_preactivation


# Softmax
class Softmax(Layer):
    """

    """

    def __init__(self):
        super().__init__()
        self.layer_output = None

    def forward_pass(self, preactivation):

        # A = Softmax(Z)
        activation = np.exp(preactivation) / np.sum(np.exp(preactivation), axis=0, keepdims=True)

        # Cache for backprop
        self.layer_output = activation

        return activation

    def backward_pass(self, grad_activation):

        # dZ = dA * softmax'(Z) = = dA * A * (1 - A)
        # NB: '*' signifies elementwise multiplication
        grad_preactivation = grad_activation * self.layer_output * (1 - self.layer_output)

        return grad_preactivation


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

    def forward_pass(self, network_input_x):
        activation = network_input_x
        for layer in self.layers:
            activation = layer(activation, method="forward")
        return activation

    def backward_pass(self, grad_cost_dyhat):
        grad = grad_cost_dyhat
        for layer in reversed(self.layers):
            grad = layer(grad, method="backward")
        return grad

