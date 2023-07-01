import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import itertools


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


def plot_training_metrics_per_step(training_metrics, window_size=100, metric_name=None, save_dir=None):
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
