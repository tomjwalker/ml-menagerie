"""

Maths/literature to script nomenclature:
========================================

X ---> features
Y ---> labels
"""
import numpy as np
from typing import Tuple, Union, List
from tqdm.auto import tqdm

from supervised_learning.low_level_implementations.feedforward_nn.costs_and_metrics import \
    (BaseMetric, CategoricalCrossentropyCost, AccuracyMetric)
from supervised_learning.low_level_implementations.feedforward_nn.models import SeriesModel
from supervised_learning.low_level_implementations.feedforward_nn.optimisers import GradientDescentOptimiser
from supervised_learning.low_level_implementations.feedforward_nn.utils import ClipNorm


def train_val_split(features, labels, train_fraction=0.8):

    # Expecting X to be of shape (n_features x m_samples) and Y to be of shape (1 x m_samples)
    m_total_samples = features.shape[1]
    idx_train_cutoff = int(m_total_samples * train_fraction)

    features_train = features[:, :idx_train_cutoff]
    features_val = features[:, idx_train_cutoff:]
    labels_train = labels[:, :idx_train_cutoff]
    labels_val = labels[:, idx_train_cutoff:]

    return features_train, features_val, labels_train, labels_val


def batch_generator(features_train_or_val, labels_train_or_val, batch_size=32, shuffle=True):

    # Get number of samples of a whole epoch. If specified, shuffle the indices of the whole epoch.
    m_samples = features_train_or_val.shape[1]
    sample_indices = [*range(m_samples)]
    if shuffle:
        np.random.shuffle(sample_indices)

    for i in range(0, m_samples, batch_size):
        batch_indices = sample_indices[i:i + batch_size]
        features_batch = features_train_or_val[:, batch_indices]
        labels_batch = labels_train_or_val[:, batch_indices]
        yield features_batch, labels_batch


class TrainingTask:
    """
    A training task defines the training architecture.

    Parameters:
        - optimiser: The optimiser to be used.
        - cost: The cost function to be used.
        - metrics: The metrics to be used and logged.
        - clip_grads_norm: Whether to clip the gradients norm or not.
        - clip_kwargs: Keyword arguments to be passed to the ClipNorm object.

    Attributes:
        - metric_log: A dictionary of metric names and their corresponding logs.
    """

    DEFAULT_CLIP_GRADS_NORM = {"max_norm": 5.0, "norm_type": 2}

    def __init__(
            self,
            optimiser: GradientDescentOptimiser,
            cost: CategoricalCrossentropyCost,
            metrics: Union[None, List[BaseMetric]] = None,
            clip_grads_norm: bool = False,
            **clip_kwargs,
    ):
        """
        Initialises a training task.
        """

        self.mode = "train"    # Passed into models - important for behaviour of certain layers which behave
        # differently in training and evaluation modes (e.g. batch norm)

        self.optimiser = optimiser
        self.cost = cost
        self.metrics = metrics

        self.metric_log = {metric.name: [] for metric in metrics}

        # If gradient norm clipping is specified, instantiate a ClipNorm object
        self.clip_grads_norm = None
        if clip_grads_norm:
            clip_params = {**TrainingTask.DEFAULT_CLIP_GRADS_NORM, **clip_kwargs}
            self.clip_grads_norm = ClipNorm(**clip_params)


class EvaluationTask:
    """
    An evaluation task defines the evaluation architecture.

    Parameters:
        - metrics: The metrics to be used and logged.

    Attributes:
        - metric_log: A dictionary of metric names and their corresponding logs.
    """

    def __init__(
            self,
            metrics: Union[None, List[BaseMetric]] = None,
    ):
        self.mode = "infer"  # Passed into models - important for behaviour of certain layers which behave
        # differently in training and evaluation modes (e.g. batch norm)

        self.metrics = metrics

        self.metric_log = {metric.name: [] for metric in metrics}


class Loop:
    """
    A loop is a runs a training task and an (optional) evaluation task for a given number of epochs.

    The loop has a run method that takes in a number of epochs and runs the training and evaluation tasks for that
    number of epochs.

    Parameters:
        - dataset: A tuple of (features, labels) of the training data.
        - model: The model to be trained.
        - training_task: The training task to be run.
        - evaluation_task: The evaluation task to be run.

    Methods:
        - run: Runs the training and evaluation tasks for a given number of epochs.
    """

    def __init__(
            self,
            dataset: Tuple[np.ndarray, np.ndarray],
            model: SeriesModel,
            training_task: TrainingTask,
            evaluation_task: EvaluationTask = None,
            ):

        self.features, self.labels = dataset
        self.model = model
        self.training_task = training_task
        self.evaluation_task = evaluation_task

        # If log_grads is True, this list will store the weight gradient logs over training iterations
        self.grads_log = {}

    def run(
            self,
            n_epochs: int = 1,
            batch_size: int = 32,
            train_fraction: float = 0.8,
            train_abs_samples: Union[None, int] = None,
            verbose: int = 1,    # 0: no progress bar, 1: simple print, 2: tqdm progress bar
            log_grads: bool = False,
    ):

        # Split dataset into training and validation sets
        features_train, features_val, labels_train, labels_val = train_val_split(self.features, self.labels,
                                                                                 train_fraction=train_fraction)

        # If train_abs samples is an int, use this to truncate the number of training samples
        if train_abs_samples is not None:
            features_train = features_train[:, :train_abs_samples]
            labels_train = labels_train[:, :train_abs_samples]

        # Initialise loop params
        cost_log = []
        n_batches = features_train.shape[1] // batch_size
        iteration_num = 0

        # Iterate over epochs
        for epoch in tqdm(range(n_epochs), desc="Training epochs", dynamic_ncols=True):

            # Instantiate batch generator object at start of each epoch
            generator = batch_generator(features_train, labels_train, batch_size=batch_size)

            # Initialise batch cost log
            batch_cost_log = []

            # Initialise a dictionary to store metric values
            metric_values = {}

            # Create a new tqdm progress bar for training batches
            # dynamic_ncols ensures that the progress bar doesn't get squashed when the terminal window is resized
            print("Number of batches:", n_batches)
            progress_bar = tqdm(generator, total=n_batches, desc="Training batches", dynamic_ncols=True, leave=False)

            # Iterate over batches
            for batch_idx, (batch_X, batch_Y) in enumerate(progress_bar):

                # =====================
                # Train
                # =====================

                # Forward pass
                predictions = self.model.forward_pass(batch_X, mode=self.training_task.mode)

                # Calculate cost
                batch_cost = self.training_task.cost(batch_Y, predictions)
                batch_cost_log.append(batch_cost)

                # Calculate dYhat
                grad_predictions = self.training_task.cost.compute_gradient(batch_Y, predictions)

                # Backward pass
                self.model.backward_pass(grad_predictions, log_grads=log_grads)

                # Update weights
                self.model.update_weights_biases(self.training_task.optimiser)

                # Update training task metrics
                if self.training_task.metrics is not None:
                    for metric in self.training_task.metrics:
                        # Calculate metric
                        metric_train = metric(batch_Y, predictions)
                        # Append metric to metric log
                        self.training_task.metric_log[metric.name].append(metric_train)
                        # Add metric to metric_values dictionary
                        metric_values[f"Training {metric.name}"] = metric_train

                # =====================
                # Evaluate
                # =====================

                if self.evaluation_task is not None:

                    # Forward pass
                    predictions_val = self.model.forward_pass(features_val, mode=self.evaluation_task.mode)

                    # Update validation task metrics
                    if self.evaluation_task.metrics is not None:
                        for metric in self.evaluation_task.metrics:
                            metric_val = metric(labels_val, predictions_val)
                            self.evaluation_task.metric_log[metric.name].append(metric_val)
                            metric_values[f"Evaluation {metric.name}"] = metric_val

                # =====================
                # Update progress bar
                # =====================

                # Update progress bar description
                progress_bar.set_description(f"Training batches (cost: {batch_cost:.4f})")

                # Update progress bar metrics
                progress_bar.set_postfix(metric_values)

                # =====================
                # Log gradients
                # =====================

                if log_grads:

                    # .copy() was necessary, otherwise as the model gradients were updated, the grads_log values from
                    # previous iterations were also updated
                    self.grads_log[iteration_num] = self.model.grads.copy()

                iteration_num += 1

            # =====================
            # Update epoch cost log
            # =====================

            # Calculate epoch cost
            epoch_cost = np.mean(batch_cost_log)

            # Append epoch cost to epoch cost log
            cost_log.append(epoch_cost)

            # =====================
            # Update progress bar
            # =====================

            # Update progress bar description
            progress_bar.set_description(f"Training epochs (cost: {epoch_cost:.4f})")

            # Update progress bar metrics
            progress_bar.set_postfix(metric_values)

