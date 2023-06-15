import numpy as np


def accuracy(labels, predictions):
    """
    Compute accuracy.

    Args:
        labels (np.array): labels. Shape (n_classes, m_samples). One-hot encoded, so it is of dtype int
        predictions (np.array): predictions. Shape (n_classes, m_samples). Output of the final softmax layer, so it is a
            probability distribution of dtype float

    Returns: accuracy (float)

    """

    # Get number of samples
    m_samples = labels.shape[1]

    # Get number of correct predictions
    label_categories = np.argmax(labels, axis=0)
    prediction_categories = np.argmax(predictions, axis=0)
    n_correct = np.sum(label_categories == prediction_categories)

    # Compute accuracy
    accuracy_over_samples = n_correct / m_samples

    return accuracy_over_samples



