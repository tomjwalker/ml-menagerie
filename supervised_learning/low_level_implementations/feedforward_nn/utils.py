import numpy as np


class ClipNorm:

    def __init__(self, max_norm=5.0, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, grad_weight_or_bias):
        return self.clip_norm(grad_weight_or_bias)

    def clip_norm(self, grad_weight_or_bias):
        """
        Clip the gradient to a maximum value.

        Args:
            grad_weight_or_bias (np.array): gradient of weights or biases

        Returns:
            np.array: clipped gradient
        """

        # Calculate norm of gradient
        grad_norm = np.linalg.norm(grad_weight_or_bias, ord=self.norm_type)

        # If norm is greater than max_norm, clip gradient
        if grad_norm > self.max_norm:
            grad_weight_or_bias = grad_weight_or_bias * self.max_norm / grad_norm

        return grad_weight_or_bias
