"""
XXX.


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


class BaseMetric:
    def __init__(self):
        self.name = None

    def __call__(self, label, prediction):
        raise NotImplementedError


class CategoricalCrossentropyCost(BaseMetric):
    def __init__(self, epsilon=1e-15, clip_gradient=True):
        super().__init__()
        self.name = "categorical_crossentropy_cost"

        # Small constant to avoid div/0 errors
        self.epsilon = epsilon

        self.clip_gradient = clip_gradient

    def __call__(self, label, prediction):

        # Sum over axis=0 sums loss for each categorical class. Output is a vector containing these costs for all
        # samples
        total_costs_over_all_classes = -np.sum(label * np.log(prediction + self.epsilon), axis=0)

        # Mean cost over samples
        cost = np.mean(total_costs_over_all_classes)

        return cost

    def compute_gradient(self, label, prediction):

        # Get number of samples
        m_samples = label.shape[1]

        # Gradient of loss w.r.t. prediction. This is the gradient of the loss for each output node and each sample
        grad_prediction = -(1 / m_samples) * label / (prediction + self.epsilon)

        # Assert that the gradient is of the same shape as the prediction
        assert grad_prediction.shape == prediction.shape

        # If specified, clip the gradient
        if self.clip_gradient == True:
            grad_prediction = np.clip(grad_prediction, -1, 1)

        return grad_prediction


class AccuracyMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.name = "accuracy"

    def __call__(self, label, prediction):

        # Get number of samples
        m_samples = label.shape[1]

        # Get number of correct predictions
        label_categories = np.argmax(label, axis=0)
        prediction_categories = np.argmax(prediction, axis=0)
        n_correct = np.sum(label_categories == prediction_categories)

        # Compute accuracy
        accuracy_over_samples = n_correct / m_samples

        return accuracy_over_samples
