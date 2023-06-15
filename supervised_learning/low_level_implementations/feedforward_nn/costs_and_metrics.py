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
    def __init__(self, epsilon=1e-10):
        super().__init__()
        self.name = "categorical_crossentropy_loss"

        # Small constant to avoid div/0 errors
        self.epsilon = epsilon

    def __call__(self, label, prediction):

        # Sum over axis=0 sums loss for each categorical class. Output is a vector containing these costs for all
        # samples
        total_costs_over_all_classes = -np.sum(label * np.log(prediction + self.epsilon), axis=0)

        # Mean cost over samples
        cost = np.mean(total_costs_over_all_classes)

        return cost

    @staticmethod
    def compute_gradient(label, prediction):

        # Gradient of loss w.r.t. prediction. This is the gradient of the loss for each output node and each sample
        grad_prediction = -label / prediction

        # Mean gradient over samples. Collapses that dimension so that the output is a vector of gradients for each node
        grad_prediction = np.mean(grad_prediction, axis=1)

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
