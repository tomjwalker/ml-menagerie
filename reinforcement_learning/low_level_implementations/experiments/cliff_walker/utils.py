import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import itertools
from scipy.stats import norm
import pandas as pd


def plot_q_table(q_table, action_num_to_str=None, episode_num=None, save_dir=None):

    if action_num_to_str is None:
        action_num_to_str = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        }

    # Create a list of action names as column labels
    actions = [action_num_to_str[i] for i in range(len(action_num_to_str))]

    # Transpose the Q-table to orient it the other way round
    q_table = q_table.T

    # Round down Q-values to 1 significant figure
    q_table = np.round(q_table, decimals=1)

    # Plot the Q-table
    fig, ax = plt.subplots()
    im = ax.imshow(q_table, cmap='plasma')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal")
    cbar.ax.set_xlabel('Q-value')

    # Set ticks and labels for x and y axes
    ax.set_xticks(np.arange(q_table.shape[1]))
    ax.set_yticks(np.arange(len(actions)))
    ax.set_xticklabels(np.arange(q_table.shape[1]))
    ax.set_yticklabels(actions)

    # Add text annotations inside the cells
    for i in range(len(actions)):
        for j in range(q_table.shape[1]):
            text = ax.text(j, i, str(q_table[i, j]), ha="center", va="center",
                           color="w" if abs(q_table[i, j]) < np.max(np.abs(q_table)) / 2 else "k")

    # Set title
    if episode_num is not None:
        ax.set_title(f"Q-table after {episode_num} episodes")
    else:
        ax.set_title("Q-table")

    # Rotate the x-axis tick labels for better visibility
    plt.xticks(rotation=90)

    # Adjust the layout to prevent overlap of annotations
    plt.tight_layout()

    # Save the plot
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"q_table_episode_{episode_num}.png"))
        plt.close()
    else:
        plt.show()


def plot_training_metrics_single_trial(training_metrics, window_size=100, metric_name=None, save_dir=None):
    """
    Takes a dictionary of training metrics (can be one, or a multiple of experiments) and plots the running average of
    each metric on top of a running variance fill between the upper and lower bounds.
    Args:
        training_metrics: dictionary of training metrics, where each key is the name of an experiment and each value
        is the experiment data for a single trial (so mean and varinance are calculated in a running window,
        rather than across multiple trials)
        window_size: int, controls the size of the rolling window used to calculate the running average and variance
        metric_name: str, for titles and save names
        save_dir: str or None, if None then the plot is shown, otherwise it is saved to the specified directory

    Returns:

    """
    colors = itertools.cycle(mcolors.TABLEAU_COLORS)

    for name, metric in training_metrics.items():
        # Calculate running average using a rolling window
        running_average = np.convolve(metric, (np.ones(window_size) / window_size), mode="valid")

        # Calculate running variance using a rolling window
        squared_diff = (metric - running_average.mean()) ** 2
        running_variance = np.convolve(squared_diff, (np.ones(window_size) / window_size), mode="valid")

        # Calculate the upper and lower bounds for the fill between
        upper_bound = running_average + np.sqrt(running_variance)
        lower_bound = running_average - np.sqrt(running_variance)

        # Plot the running average
        color = next(colors)
        plt.plot(running_average, label=name, color=color)

        # Plot the fill between the upper and lower bounds
        plt.fill_between(range(len(running_average)), upper_bound, lower_bound, alpha=0.3, color=color)

    # Set the plot title and labels
    if metric_name is None:
        title = "Training Metric per Episode"
    else:
        title = f"{metric_name}"
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Metric Value")

    # Show the legend
    plt.legend()

    # Save the plot
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{metric_name}_per_episode.png"))
        plt.close()
    else:
        plt.show()


def calculate_experiment_statistics(metric_over_multiple_trials, confidence_interval=0.95):

    experiment_statistics = pd.DataFrame(columns=["mean", "std", "upper_bound", "lower_bound"])

    experiment_mean = np.mean(metric_over_multiple_trials, axis=0)
    experiment_std = np.std(metric_over_multiple_trials, axis=0)
    z_value = norm.ppf((1 + confidence_interval) / 2)
    num_trials = len(metric_over_multiple_trials)
    upper_bound = experiment_mean + (z_value * (experiment_std / np.sqrt(num_trials)))
    lower_bound = experiment_mean - (z_value * (experiment_std / np.sqrt(num_trials)))

    experiment_statistics["mean"] = experiment_mean
    experiment_statistics["std"] = experiment_std
    experiment_statistics["upper_bound"] = upper_bound
    experiment_statistics["lower_bound"] = lower_bound

    return experiment_statistics


def plot_training_metrics_multiple_trials(
        metrics_over_multiple_trials,
        metric_name=None,
        save_dir=None,
        show_individual_trials=False
):
    """
    Takes the dataframe returned by `calculate_experiment_statistics` and plots the mean trace on top of a confidence
    interval fill between the upper and lower bounds. Optionally, individual trial traces can be shown as well.

    Args:
        metrics_over_multiple_trials: dict of data. The data is a numpy array of shape (num_trials, num_episodes) where each
        row is the data for a single trial. The dict keys are the names of the experiments.
        `calculate_experiment_statistics`
        metric_name: if None, set the title to "Training Metric per Episode"
        save_dir: if None, show the plot instead of saving it
        show_individual_trials: if True, individual trial traces will be plotted with alpha=0.1
    """

    colors = itertools.cycle(mcolors.TABLEAU_COLORS)

    for run_name, metric in metrics_over_multiple_trials.items():

        # Get training metric stats
        training_metric_stats = calculate_experiment_statistics(metric)

        # Plot the mean trace
        color = next(colors)
        plt.plot(training_metric_stats["mean"], label=run_name, color=color)

        # Plot the fill between the upper and lower bounds
        plt.fill_between(range(len(training_metric_stats["mean"])),
                         training_metric_stats["upper_bound"],
                         training_metric_stats["lower_bound"],
                         alpha=0.3,
                         color=color,
                         )

        # Plot the individual trial traces, if desired
        if show_individual_trials:
            for trial in metric:
                plt.plot(trial, alpha=0.05, color=color)

        # Set the plot title and labels
        if metric_name is None:
            title = "Training Metric per Episode"
        else:
            title = f"{metric_name}"
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Metric Value")

        # Show the legend
        plt.legend()

        # Save the plot
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{metric_name}_per_episode.png"))
            plt.close()
        else:
            plt.show()

