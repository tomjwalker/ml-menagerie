from supervised_learning.datasets.mnist.data_utils import load_mnist
from supervised_learning.datasets.mnist.data_utils import preprocess_mnist
from supervised_learning.datasets.mnist.data_utils import show_digit_samples

# Load MNIST dataset
features, labels = load_mnist()

# Preprocess MNIST dataset
features, labels = preprocess_mnist(features, labels)

# Show some samples
show_digit_samples(features, labels, m_samples=10)
