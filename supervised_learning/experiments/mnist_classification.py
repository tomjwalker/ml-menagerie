import numpy as np

from supervised_learning.datasets.mnist.data_utils import load_mnist, preprocess_mnist, show_digit_samples
from supervised_learning.low_level_implementations.feedforward_nn.costs_and_metrics import (
    CategoricalCrossentropyCost, AccuracyMetric)
from supervised_learning.low_level_implementations.feedforward_nn.layers import Dense, Input, Relu, Softmax
from supervised_learning.low_level_implementations.feedforward_nn.models import SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser
from supervised_learning.low_level_implementations.feedforward_nn.tasks import (TrainingTask, EvaluationTask, Loop,
                                                                                train_val_split)

# Load MNIST dataset
features, labels = load_mnist()

# Preprocess MNIST dataset
features, labels = preprocess_mnist(features, labels)

# Show a few samples
show_digit_samples(features, labels, m_samples=10)

# Define network architecture as a series of layers
architecture = [
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

# Define training task
training_task = TrainingTask(
    optimiser=GradientDescentOptimiser(learning_rate=0.01),
    cost=CategoricalCrossentropyCost(),
    metrics=[CategoricalCrossentropyCost(), AccuracyMetric()],
)

# Define evaluation task
evaluation_task = EvaluationTask(
    metrics=[CategoricalCrossentropyCost(), AccuracyMetric()],
)

# Run training loop
loop = Loop(
    dataset=(features, labels),
    model=model,
    training_task=training_task,
    evaluation_task=evaluation_task,
)
loop.run(n_epochs=1, batch_size=32)
