"""
Defines layers: feedforward neural network components. Each layer has a forward pass and a backward pass method.


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
from typing import Union

import numpy as np


class Layer:
    """
    Base class for layers in feedforward neural network

    Parameters:

    Attributes:

    Methods:
        __init__: initialise layer
        __repr__: print representation of layer
        __call__: call layer (forward or backward pass)
        forward_pass: forward pass through layer
        backward_pass: backward pass through layer
    """

    def __init__(self):

        # Flag which indicates whether layer has trainable parameters (weights, biases, BatchNorm params etc.)
        # Helps for methods e.g. model save/load
        self.trainable = False

    def __repr__(self):
        """
        When print called on the object, return 'Layer: <class layer name>'.
        This can be added to in subclass (specific layer) representations
        """
        return f"""
        ==========================================================================
        Layer: {self.__class__.__name__}\
        """

    def __call__(self, input_activation_or_grad, method, mode=None, **backprop_tools):
        if method == "forward":
            if mode is None:
                raise ValueError("Must specify mode for forward pass")
            return self.forward_pass(input_activation_or_grad, mode)
        if method == "backward":
            return self.backward_pass(input_activation_or_grad, **backprop_tools)
        raise ValueError("Invalid method; should be an element of {'forward', 'backward'}")

    def forward_pass(self, input_activation_from_left, mode):
        """
        Forward pass through layer
        """
        pass

    def backward_pass(self, input_grad_from_right, **backprop_tools):
        """
        Backward pass through layer
        """
        pass


class Dense(Layer):
    """
    Dense layer (fully connected layer) for feedforward neural network

    Parameters:
        n_neurons (int): number of neurons in layer
        weight_init_scale (float): Sets magnitude of randomised weight initialisation

    Attributes:
        layer_input (np.array): input to layer
        m_samples (int): number of samples in input
        n_neurons (int): number of neurons in layer
        weight_init_scale (float): Sets magnitude of randomised weight initialisation
        weights (np.array): weights for layer
        bias (np.array): bias vector for layer
        grad_weights (np.array): gradient of weights
        grad_bias (np.array): gradient of bias vector

    Methods:
        initialise_weights: randomise weights
        initialise_bias: randomise bias vector
        forward_pass: forward pass through layer
        backward_pass: backward pass through layer
    """

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

        self.trainable = True

        # TODO: don't recalculate this on every pass
        self.m_samples = None

        # Parameters passed in on instantiation
        self.n_neurons = n_neurons
        self.weight_init_scale = weight_init_scale

        # State weight and bias attributes. These are initialised in the SeriesModel class once neighbouring layers are
        # known
        self.weights = None
        self.bias = None
        self.grad_weights = None
        self.grad_bias = None

    def __repr__(self):
        superclass_repr = super().__repr__()
        if self.weights is None:
            return superclass_repr + "Weights and bias not initialised yet."
        num_params = self.weights.size + self.bias.size
        dense_stub = f"""
        Weights shape: ({self.weights.shape}). Bias shape: ({self.bias.shape}. # trainable params: {num_params})\
        """
        return superclass_repr + dense_stub

    def initialise_weights(self, prev_layer_neurons, random_seed: Union[None, int] = 42):
        """
        Random initialisation of weights to "break symmetry" between hidden units.
        Also initialises self.grad_weights to same shape as self.weights.

        Using He-initialisation to avoid "Dying ReLU problem" (see https://arxiv.org/pdf/1502.01852.pdf)

        Args:
            random_seed: If not None, sets random seed for reproducibility
            prev_layer_neurons (int): number of neurons in previous layer
        """

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # He-initialisation variance factor
        stdev_he = np.sqrt(2 / prev_layer_neurons)

        # Random initialisation of weights with He-initialisation variance factor
        self.weights = np.random.randn(self.n_neurons, prev_layer_neurons) * stdev_he
        self.grad_weights = np.zeros_like(self.weights)

    def initialise_bias(self):
        """
        Random initialisation of bias vector. Also initialises self.grad_bias to same shape as self.bias
        """
        self.bias = np.zeros((self.n_neurons, 1))
        self.grad_bias = np.zeros_like(self.bias)

    def forward_pass(self, prev_layer_activation, mode):
        """
        This function takes as input A_(l-1), the activation from a previous layer, and returns the preactivation
        Z_(l) = W_l . A_(l-1) + b_l
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

    def backward_pass(self, grad_layer_preactivation, **backprop_tools):
        """
        This function takes as input del(J)/del(Z_l) = dZ_l (cost w.r.t layer output) and updates grads for:
        - dA_l-1 --> Cost w.r.t. layer input. Code nomenclature: grad_prev_layer_activation.
            Shape: (n_neurons_prev_layer, 1)
        - dW_l --> Cost w.r.t. layer weight matrix. Code nomenclature: self.grad_weights.
            Shape: (n_neurons, n_neurons_prev_layer)
        - db_l --> Cost w.r.t. layer bias vector. Code nomenclature: self.grad_bias/.
            Shape: (n_neurons, 1)
        """

        # Determine if a ClipNorm object has been passed in. If so, clip gradients
        clip_norm = None
        if "clip_grads_norm" in backprop_tools:
            clip_norm = backprop_tools["clip_grads_norm"]

        # ==============================================================================================================
        # Update weights and biases. These are stored as attributes of the layer
        # ==============================================================================================================

        # del(J)/del(W_l) = dW_l = dZ_l . del(Z_l)/del(W_l) = (1/m_samples) * np.dot(dZ_out, A_in.T).
        # Matmul commuted with A transpose to ensure dW_l and W_l have same shape
        # Matmul sums products over the sample dimension. The (1 / m_samples) then ensures values are average weight
        self.grad_weights = (1 / self.m_samples) * np.dot(grad_layer_preactivation, self.layer_input.T)
        if clip_norm is not None:
            self.grad_weights = clip_norm(self.grad_weights)
        assert self.grad_weights.shape == self.weights.shape

        # del(J)/del(b_l) =  del(J)/del(Z_l) . del(Z_l)/del(b_l) --> dZ_l . 1.
        # The sum(...axis=1) sums over the sample dimension for dZ_l. The (1 / m_samples) then ensures values are
        # average bias grads over the samples
        self.grad_bias = (1 / self.m_samples) * np.sum(grad_layer_preactivation, axis=1, keepdims=True)
        if clip_norm is not None:
            self.grad_bias = clip_norm(self.grad_bias)
        assert self.grad_bias.shape == self.bias.shape

        # ==============================================================================================================
        # Update grad for prev layer activation (output left for backprop). This is returned to the previous layer
        # ==============================================================================================================

        # del(J)/del(A_l-1) --> dA_1 = np.dot(W_2.T, dZ_2)
        # The order of matmul and inclusion of transpose can be determined by considering d<param> and <param> have same
        # shapes, and looking at shapes at both sides of equality
        grad_prev_layer_activation = np.dot(self.weights.T, grad_layer_preactivation)
        if clip_norm is not None:
            grad_prev_layer_activation = clip_norm(grad_prev_layer_activation)
        assert grad_prev_layer_activation.shape == self.layer_input.shape

        return grad_prev_layer_activation


class Relu(Layer):
    """
    Relu activation function.

    Forward pass: A = ReLU(Z)
    Backward pass: dZ = dA * ReLU'(Z). ReLU'(Z) becomes a binary mask: 1 where Z>0, 0 otherwise
    """

    def __init__(self):
        super().__init__()
        self.layer_input = None
        self.trainable = False

    def forward_pass(self, layer_preactivation, mode):
        """
        Relu layer forward pass

        Args:
            layer_preactivation (np.array): Input to layer, Z_l

        Returns:
            layer_activation (np.array): Output of layer, A_l

        """

        # A = ReLU(Z)
        layer_activation = np.maximum(0, layer_preactivation)

        # Cache for backprop
        self.layer_input = layer_activation

        return layer_activation

    def backward_pass(self, grad_layer_activation, **backprop_tools):
        """
        Relu layer backprop

        Args:
            grad_layer_activation (np.array): del(J)/del(A_l) = dA_l (cost w.r.t layer output)

        Returns:
            grad_layer_preactivation (np.array): del(J)/del(Z_l) = dZ_l (cost w.r.t layer preactivation)

        """

        # dZ = dA * ReLU'(Z). ReLU'(Z) becomes a binary mask: 1 where Z>0, 0 otherwise
        grad_layer_preactivation = grad_layer_activation * (self.layer_input > 0)

        if "clip_grads_norm" in backprop_tools:
            clip_norm = backprop_tools["clip_grads_norm"]
            grad_layer_preactivation = clip_norm(grad_layer_preactivation)

        # Assert shape is same as input
        assert grad_layer_preactivation.shape == self.layer_input.shape

        return grad_layer_preactivation


# Softmax
class Softmax(Layer):
    """

    """

    def __init__(self):
        super().__init__()
        self.layer_output = None
        self.trainable = False

    def forward_pass(self, preactivation, mode):

        # Subtract max to avoid overflow
        preactivation -= np.max(preactivation, axis=0, keepdims=True)

        # Exponentiate the values
        exp_preactivation = np.exp(preactivation)

        # Calculate softmax probabilities
        activation = exp_preactivation / np.sum(exp_preactivation, axis=0, keepdims=True)

        # # Cache for backprop
        self.layer_output = activation

        return activation

    def backward_pass(self, grad_activation, **backprop_tools):
        """
        Backprop through softmax layer, as a function of layer cached attributes:
        - layer output A (self.layer_output)
        - layer input at backprop, dA (grad_activation)

        Differential dA/dZ is a Jacobian matrix, with shape (n_neurons, n_neurons). The diagonal elements are:
        dA_i/dZ_i = A_i * (1 - A_i). The off-diagonal elements are: dA_i/dZ_j = -A_i * A_j. The off-diagonal elements are calculated
        by multiplying the layer output by its transpose, and subtracting the diagonal elements (which are already
        calculated).

        Finally, dZ = dA * dA/dZ = dA * Jacobian.

        Shapes:
        - self.layer_output = A = (n_neurons, m_samples)
        - grad_activation = dA = (n_neurons, m_samples) (Grad loss w.r.t. layer output, calculated at loss layer,
        averaged over
            samples)
        - grad_preactivation = dZ = (n_neurons, m_samples) (Grad loss w.r.t. layer preactivation, calculated at loss
            layer, averaged over samples)
        """

        # Calculate the Jacobian matrix
        # # Get diagonal elements
        diag = self.layer_output * (1 - self.layer_output)
        # # Average over samples
        diag = np.mean(diag, axis=1, keepdims=True)
        # # Turn into diagonal matrix
        diag = np.diag(diag.flatten())
        # # Get off-diagonal elements
        off_diag = -np.matmul(self.layer_output, self.layer_output.T)
        # # Set diagonal elements to 0 (does this in-place)
        np.fill_diagonal(off_diag, 0)
        # # Add diagonal and off-diagonal elements
        jacobian = diag + off_diag

        # Calculate dZ = Jacobian . dA:
        # # dA is currently of shape (n_neurons, m_samples). Jacobian is of shape (n_neurons, n_neurons). Want dZ to be
        # # the same shape as Z (n_neurons, m_samples), so matrix multiplication of (Jacobian . dA) ensures this.
        grad_preactivation = np.matmul(jacobian, grad_activation)

        # Determine if a ClipNorm object has been passed in. If so, clip gradients
        clip_norm = None
        if "clip_grads_norm" in backprop_tools:
            clip_norm = backprop_tools["clip_grads_norm"]
            grad_preactivation = clip_norm(grad_preactivation)

        # Shapes of dA, dZ should be the same (n_neurons, m_samples)
        assert grad_preactivation.shape == grad_activation.shape

        return grad_preactivation


class BatchNorm(Layer):
    """
    Batch normalisation layer. Normalises the output of the previous layer, and scales and shifts it by learned
    parameters.

    Forward pass:
    - Calculate mean and variance of layer input (mu_Z_l, sigma_Z_l)
    - Normalise layer input (U_l = (Z_l - mu_Z_l) / sqrt(sigma_Z_l + epsilon))
    - Scale and shift normalised layer input by learned parameters (V_l = gamma_l * U_l + beta_l)

    Backward pass:
    - Calculate gradients of gamma and beta (dJ/dgamma_l, dJ/dbeta_l)
    - Calculate gradients of layer input (dJ/dU_l)
    - Calculate gradients of layer preactivation (dJ/dZ_l)

    Attributes:
    - gamma: scale parameter, shape (n_neurons, 1)
    - beta: shift parameter, shape (n_neurons, 1)
    - epsilon: small number to avoid division by zero
    """

    def __init__(self, epsilon=1e-8, momentum=0.9):

        super().__init__()

        self.trainable = True

        # Gamma and beta can't be initialised until the number of neurons in the layer is known, within the context
        # of the model. Therefore, initialise to None for now.
        self.gamma = None
        self.beta = None
        self.grad_gamma = None
        self.grad_beta = None

        self.epsilon = epsilon

        # Initialise running mean and variance to None. These will be updated during training.
        self.running_mean = None
        self.running_var = None

        # These values, calculated during forward pass, are cached for use during backprop
        self.input_normalised = None

        # Momentum for running mean and variance
        self.momentum = momentum

    def initialise_trainable_params(self, n_neurons):
        """
        Initialise trainable parameters gamma and beta
        """

        # Initialise gamma and beta
        gamma = np.ones((n_neurons, 1))    # Initialise with no scaling
        beta = np.zeros((n_neurons, 1))    # Initialise with no shift

        return gamma, beta

    def forward_pass(self, input_activation_from_left, mode):

        if self.gamma is None or self.beta is None:
            # Initialise gamma and beta
            self.gamma, self.beta = self.initialise_trainable_params(input_activation_from_left.shape[0])

        # Update running mean and variance, if mode == "train". Otherwise use previously calculated values if mode ==
        # "infer"
        if mode == "train":
            # Calculate mean and variance of layer input
            batch_mean = np.mean(input_activation_from_left, axis=1, keepdims=True)
            batch_var = np.var(input_activation_from_left, axis=1, keepdims=True)
            if self.running_mean is None or self.running_var is None:
                # Initialise running mean and variance, if this is the first forward pass
                self.running_mean = batch_mean
                self.running_var = batch_var
            else:
                # Exponential moving average of running mean and variance
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        elif mode == "infer":
            # (Use running mean and variance for inference)
            pass
        else:
            raise ValueError("Mode must be either 'train' or 'infer'")

        # Normalise layer input
        input_normalised = (input_activation_from_left - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift normalised layer input by learned parameters
        layer_activation = self.gamma * input_normalised + self.beta

        # Cache these values for backprop
        self.input_normalised = input_normalised

        return layer_activation

    def backward_pass(self, input_grad_from_right, **backprop_tools):
        """
        Backward pass for batch normalisation layer. Calculate gradients of gamma, beta, layer input and layer
        preactivation.

        Cutting the corner a bit here, by implementing via
        https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
        """

        # Determine if a ClipNorm object has been passed in. If so, clip gradients
        clip_norm = None
        if "clip_grads_norm" in backprop_tools:
            clip_norm = backprop_tools["clip_grads_norm"]

        # Get number of samples
        m = input_grad_from_right.shape[1]

        # Calculate gradients of gamma and beta
        self.grad_gamma = np.sum(input_grad_from_right * self.input_normalised, axis=1, keepdims=True)
        if clip_norm is not None:
            self.grad_gamma = clip_norm(self.grad_gamma)
        self.grad_beta = np.sum(input_grad_from_right, axis=1, keepdims=True)
        if clip_norm is not None:
            self.grad_beta = clip_norm(self.grad_beta)

        # Calculate gradients of for input_normalised

        grad_input_normalised = self.gamma * input_grad_from_right

        # Calculate gradients of layer preactivation (see del(J)/del(X) formula in link above)
        left_hand_term = (1 / m) * self.gamma * (self.running_var + self.epsilon) ** (-1/2)
        right_hand_term = (-(self.grad_gamma * self.input_normalised) + (m * input_grad_from_right) - self.grad_beta)
        grad_preactivation = left_hand_term * right_hand_term
        if clip_norm is not None:
            grad_preactivation = clip_norm(grad_preactivation)

        # Check shapes
        assert self.grad_gamma.shape == self.gamma.shape
        assert self.grad_beta.shape == self.beta.shape
        assert grad_input_normalised.shape == self.input_normalised.shape
        assert grad_preactivation.shape == input_grad_from_right.shape   # Input to layer is same shape as output

        return grad_preactivation
