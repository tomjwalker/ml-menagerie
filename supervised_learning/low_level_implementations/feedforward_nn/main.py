import numpy as np

from supervised_learning.low_level_implementations.feedforward_nn.layers import (
    Dense, Input, Relu, Softmax)
from supervised_learning.low_level_implementations.feedforward_nn.models import \
    SeriesModel

# Give a demo input
x = np.random.randn(784, 10)

# Define network architecture as a series of layers
architecture = [
        Input(x),
        Dense(5),
        Relu(),
        Dense(10),
        Softmax(),
    ]

# Initialise model
model = SeriesModel(
    layers=architecture,
)

print(model)
