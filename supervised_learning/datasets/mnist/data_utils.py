import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

########################################################################################################################
# Loading functions
########################################################################################################################

# Get the directory path of the current module
module_dir = os.path.dirname(os.path.abspath(__file__))

# Specify a cache directory within the module's directory
cache_dir = os.path.join(module_dir, "data_cache")
cache_filename = "mnist_data.cache"
cache_filepath = os.path.join(cache_dir, cache_filename)

def load_mnist():
    """
    Load MNIST dataset.

    Output array shapes are:
    - features: (m_samples, n_features)
    - labels: (m_samples,)
    where n_features = 784 and m_samples = 70000

    Returns: features (np.array), labels (np.array)

    """

    # Check if cache file exists
    if os.path.exists(cache_filepath):
        # Load from cache
        mnist = joblib.load(cache_filepath)

    else:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        # Fetch MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        # Save to cache
        joblib.dump(mnist, cache_filepath)

    # Extract features and labels
    features = mnist.data
    labels = mnist.target

    return features, labels


########################################################################################################################
# Preprocessing functions
########################################################################################################################

def preprocess_mnist(features, labels):
    """
    Preprocess MNIST dataset. Input features and labels are converted to the correct format for the neural network:
    - Features are normalised to be between 0 and 1
    - Labels are converted to one-hot encoding

    Output array shapes are:
    - features: (n_features, m_samples)
    - labels: (n_classes, m_samples)

    Args:
        features (np.array): raw features
        labels (np.array): raw labels

    Returns: features (np.array), labels (np.array). Preprocessed features and labels

    """
    # Convert labels to one-hot encoding
    labels = labels.astype(int)
    # Set up array of zeros with shape (m_samples, n_classes)
    labels_onehot = np.zeros((labels.size, labels.max() + 1))
    # Set the correct element to 1 for each sample. Original labels are used as indices for the correct column position
    labels_onehot[np.arange(labels.size), labels] = 1
    # Transpose so that shape is (n_classes, m_samples)
    labels_onehot = labels_onehot.T

    # Normalise features
    features = features / 255

    # features read in as a Pandas DataFrame, so convert to np.array
    features = np.array(features)
    # Transpose so that shape is (n_features, m_samples)
    features = features.T

    return features, labels_onehot


def sample_digit_to_image(digit):
    """
    Convert a sample digit to an image.

    Args:
        digit (np.array): sample digit

    Returns: image (np.array). Image of the sample digit

    """
    image = digit.reshape(28, 28)
    return image


def show_digit_samples(features, labels, predictions=None, m_samples=10, save_filepath=None):
    """
    Show sample digits from the MNIST dataset.

    Args:
        features (np.array): features
        labels (np.array): labels
        predictions (np.array): predictions (optional). If provided, predictions will be shown alongside labels
        m_samples (int): number of sample digits to show

    Returns: None

    """
    # Get sample indices
    sample_indices = np.random.choice(features.shape[1], size=m_samples, replace=False)

    # Get sample digits
    sample_digits = features[:, sample_indices]

    # Get sample labels
    sample_labels = labels[:, sample_indices]

    # If predictions are provided, get sample predictions
    sample_predictions = None
    if predictions is not None:
        sample_predictions = predictions[:, sample_indices]

    # Convert sample digits to images
    sample_images = [sample_digit_to_image(digit) for digit in sample_digits.T]

    # Set up the figure and axes
    n_samples = len(sample_images)
    n_columns = (n_samples + 1) // 2
    fig, axes = plt.subplots(2, n_columns, figsize=(n_columns * 4, 6))
    fig.suptitle('Sample Digits', fontsize=16)

    # Iterate over samples and plot
    for i, (image, label) in enumerate(zip(sample_images, sample_labels.T)):
        ax = axes[i // n_columns, i % n_columns]
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # Set the title and subtitle
        title = f'Label: {np.argmax(label)}'
        if predictions is not None:
            prediction = np.argmax(sample_predictions[:, i])
            subtitle = f'Prediction: {prediction}'
            ax.set_title(title, color='black')
            ax.text(0.5, -0.15, subtitle, transform=ax.transAxes, color='green', ha='center')
        else:
            ax.set_title(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust top margin for title

    if save_filepath is not None:
        plt.savefig(save_filepath)
    else:
        plt.show()
