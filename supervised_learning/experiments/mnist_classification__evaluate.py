from supervised_learning.low_level_implementations.feedforward_nn.models import load_checkpoint
from supervised_learning.datasets.mnist.data_utils import load_mnist, preprocess_mnist, show_digit_samples

import matplotlib.pyplot as plt


# Parameters and architecture names. These are used both as parameters into the model/loop, and as filenames for
# saving the outputs of the training loop
DATA_CACHE_DIR = "./data_cache"
PLOTS_DIR = "./plots"
MODEL_CHECKPOINTS_DIR = "./model_checkpoints"

RUN_SETTINGS = {
    "model_name": "mnist_ffnn_dense_50_batchnorm",
    "num_epochs": 2,  # 60
    "train_abs_samples": 200,  # TODO: remove this after debugging (set to None) - then full training set is used
    "clip_grads_norm": True,
}

# Filepath prefix specifies the run settings as defined above
run_suffix = "__".join([f"{key}_{str(value)}" for key, value in RUN_SETTINGS.items()])
run_suffix = run_suffix.replace(".", "_")  # If any values are floats, replace "." with "_" for filename

best_model = load_checkpoint(f"{MODEL_CHECKPOINTS_DIR}/{run_suffix}.pkl")
last_model = load_checkpoint(f"{MODEL_CHECKPOINTS_DIR}/{run_suffix}__last_run.pkl")

# Load MNIST dataset
features, labels = load_mnist()

# Preprocess MNIST dataset
features, labels = preprocess_mnist(features, labels)

for model in [best_model, last_model]:
    predictions = model.forward_pass(features, mode="infer")
    show_digit_samples(features, labels, predictions, m_samples=10)
