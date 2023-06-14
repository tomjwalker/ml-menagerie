class GradientDescentOptimiser:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, weight_or_bias_array, grad_weight_or_bias):
        return weight_or_bias_array - self.learning_rate * grad_weight_or_bias
