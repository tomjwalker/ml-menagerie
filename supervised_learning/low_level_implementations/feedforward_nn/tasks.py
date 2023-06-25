"""

Maths/literature to script nomenclature:
========================================

X ---> features
Y ---> labels
"""
import numpy as np
import pickle
from typing import Tuple, Union, List
from tqdm.auto import tqdm

import os

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
        self.metrics = {metric.name: metric for metric in metrics}

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

        self.metrics = {metric.name: metric for metric in metrics}

        self.metric_log = {metric.name: [] for metric in metrics}


class ModelSaveTask:
    """
    Function checks (at n_iters frequency) whether the current model is the best model so far. If so, it saves the
    model to a checkpoint file.
    """

    metric_initialisation = {
        "accuracy": {"init": 0.0, "how": "maximise"},
        "categorical_crossentropy_cost": {"init": np.inf, "how": "minimise"}
    }

    def __init__(
            self,
            monitoring_task: Union[TrainingTask, EvaluationTask],
            metric_type: Union[CategoricalCrossentropyCost, AccuracyMetric] = CategoricalCrossentropyCost(),
            save_every_n_iters: int = 10,
            save_dir: str = "./model_save_checkpoint/",
            save_filename: Union[str, None] = None,
    ):
        """
        Initialises a model save task.
        Args:
            monitoring_task: This has been previously initialised and is used to retrieve the metric logs.
            metric_type: This is used to retrieve the metric name.
            save_every_n_iters: Sets the frequency at which the model is saved.
            save_dir: Sets the directory in which the model is saved.
            save_filename: Sets the filename for model saving.
        """

        # Store a filepath for saving the model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Get the metric name from the provided metric type
        self.metric_name = metric_type.name
        self.save_filename = save_filename
        if self.save_filename is None:
            self.save_filename = f"{self.metric_name}_checkpoint"
        self.save_filepath = os.path.join(save_dir, self.save_filename)

        self.save_every_n_iters = save_every_n_iters

        self.monitoring_task = monitoring_task
        self.tracking_metric = monitoring_task.metrics[self.metric_name]

        self.best_so_far_value = self.metric_initialisation[self.metric_name]["init"]
        self.best_so_far_iteration = 0
        self.optimisation_direction = self.metric_initialisation[self.metric_name]["how"]

    def __call__(self, model: SeriesModel, n_iters: int):
        if n_iters % self.save_every_n_iters == 0:
            metric_value = self.monitoring_task.metric_log[self.metric_name][-1]
            if self.optimisation_direction == "maximise":
                if metric_value > self.best_so_far_value:
                    self.best_so_far_value = metric_value
                    self.best_so_far_iteration = n_iters
                    model.save_checkpoint(self.save_filepath)
            elif self.optimisation_direction == "minimise":
                if metric_value < self.best_so_far_value:
                    self.best_so_far_value = metric_value
                    self.best_so_far_iteration = n_iters
                    model.save_checkpoint(self.save_filepath)
            else:
                raise ValueError(f"Unknown optimisation direction: {self.optimisation_direction}")


class Loop:
    """
    A loop is a runs a training task and an (optional) evaluation task for a given number of epochs.

    The loop has a run method that takes in a number of epochs and runs the training and evaluation tasks for that
    number of epochs.

    Parameters:
        - dataset: A tuple of (features, labels) of the training data.
        - model: The model to be trained.
        - training_task: The training task to be run.
        - monitoring_task: The evaluation task to be run.

    Methods:
        - run: Runs the training and evaluation tasks for a given number of epochs.
    """

    def __init__(
            self,
            dataset: Tuple[np.ndarray, np.ndarray],
            model: SeriesModel,
            training_task: TrainingTask,
            evaluation_task: Union[None, EvaluationTask] = None,
            model_save_task: Union[None, ModelSaveTask] = None,
            ):

        self.features, self.labels = dataset
        self.model = model
        self.training_task = training_task
        self.evaluation_task = evaluation_task
        self.model_save_task = model_save_task

        # If log_grads is True, this list will store the weight gradient logs over training iterations
        self.grads_log = {}

        # Store training and validation sets after split (initialise as None, this will be set in run method)
        self.features_train = None
        self.features_val = None
        self.labels_train = None
        self.labels_val = None

    def run(
            self,
            n_epochs: int = 1,
            batch_size: int = 32,
            train_fraction: float = 0.8,
            train_abs_samples: Union[None, int] = None,
            verbose: int = 1,    # 0: no progress bar, 1: simple print, 2: tqdm progress bar
            log_grads: bool = False,
    ):
        """
        Runs the training and evaluation tasks for a given number of epochs.

        Args:
            n_epochs: Number of epochs to run.
            batch_size: Batch size to use.
            train_fraction: Fraction of the dataset to use for training.
            train_abs_samples: Number of training samples to use. If None, use all training samples.
            verbose: Verbosity level.
            log_grads: Whether to log the weight gradients or not.

        Returns:
            None. The training and evaluation tasks are run and the model is updated in place, and saved if a model
            save task is specified.
        """

        # Split dataset into training and validation sets
        features_train, features_val, labels_train, labels_val = train_val_split(self.features, self.labels,
                                                                                 train_fraction=train_fraction)

        # Store training and validation sets as attributes.
        self.features_train = features_train
        self.features_val = features_val
        self.labels_train = labels_train
        self.labels_val = labels_val

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
                    for metric_name, metric in self.training_task.metrics.items():
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
                        for metric_name, metric in self.evaluation_task.metrics.items():
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

                # =====================
                # Save model
                # =====================
                if self.model_save_task is not None:
                    self.model_save_task(model=self.model, n_iters=iteration_num)

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

        # Get the best run
        best_run_value = self.model_save_task.best_so_far_value
        best_run_iteration = self.model_save_task.best_so_far_iteration

        # Print best run
        print(f"Best saved run: {best_run_value:.4f} at iteration {best_run_iteration}")

        # Save these numbers as a log within model_checkpoints
        with open(f"{self.model_save_task.save_filepath}__best_run.txt", "w") as f:
            f.write(f"{best_run_value:.4f} at iteration {best_run_iteration}")

        # Save the last run's model
        self.model.save_checkpoint(f"{self.model_save_task.save_filepath}__last_run.pkl")

        # Save loop as a pickle file
        with open(f"{self.model_save_task.save_filepath}__loop.pkl", "wb") as f:
            pickle.dump(self, f)




