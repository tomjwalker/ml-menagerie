import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import warnings    # There's an annoying warning in matplotlib to suppress
from enum import Enum
import os
import itertools

from reinforcement_learning.low_level_implementations.tabular_q_learning.agent import Agent


########################################################################################################################
# Helper functions
########################################################################################################################

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
        title = "Training Metrics per Episode"
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



########################################################################################################################
# Run parameters
########################################################################################################################

# Environment parameters
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 100
SAVE_FREQ = 100
RENDER_MODE = "none"    # "human" to render the environment/episode. "none" to turn off rendering
# Slippery version of environment has stochastic transitions: based on action, agent may move in a direction other
# than the one intended (with probability 1/3 for all directions except 180 degrees opposite action selection)
IS_SLIPPERY = False

# Agent parameters
AGENT_NAME = "tabular_q_learning__vanilla_epsilon_greedy"
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2


# Training artefact directories
class RunDirectories(Enum):
    Q_TABLE_DATA = f"./.cache/{AGENT_NAME}/data/q_table"
    METRIC_DATA = f"./.cache/{AGENT_NAME}/data/metrics"
    Q_TABLE_PLOTS = f"./.cache/{AGENT_NAME}/plots/q_table"
    METRIC_PLOTS = f"./.cache/{AGENT_NAME}/plots/metrics"


for directory in RunDirectories:
    if not os.path.exists(directory.value):
        os.makedirs(directory.value)


########################################################################################################################
# Training loop
########################################################################################################################

# Instantiate the environment and agent
env = gym.make('FrozenLake-v1', render_mode=RENDER_MODE, is_slippery=IS_SLIPPERY)
agent = Agent(
    gamma=DISCOUNT_FACTOR,
    alpha=LEARNING_RATE,
    epsilon=EXPLORATION_RATE,
    num_states=env.observation_space.n,
    num_actions=env.action_space.n
)

# Initialise performance metrics
episode_total_reward = []
episode_length = []
episode_discounted_return_per_step = []
for episode in range(NUM_EPISODES):

    # Reset the environment
    state, info = env.reset()

    # Run the episode to completion, or until the maximum number of steps is reached
    terminated = False
    truncated = False
    episode_rewards = []
    while not (terminated or truncated):

        # Agent action decision
        action = agent.choose_action(state)

        # Environment transition
        new_state, reward, terminated, truncated, info = env.step(action)

        # Agent learning
        agent.update_q_table(state, action, new_state, reward)

        # Update state
        state = new_state

        # Record the reward
        episode_rewards.append(reward)

        if RENDER_MODE == "human":
            env.render()

    # Record performance metrics
    episode_total_reward.append(sum(episode_rewards))
    episode_length.append(len(episode_rewards))
    episode_discounted_return_per_step.append(
        sum([agent.gamma ** i * episode_rewards[i] for i in range(len(episode_rewards))]) / len(episode_rewards)
    )

    if episode % SAVE_FREQ == 0:
        print(f"Episode {episode} of {NUM_EPISODES} completed")
        # Save the Q-table to a file
        agent.save_q_table(f"{RunDirectories.Q_TABLE_DATA.value}/q_table_episode_{episode}.npy")

        # Plot the Q-table while suppressing the MatplotlibDeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_q_table(agent.q_table, episode_num=episode, save_dir=RunDirectories.Q_TABLE_PLOTS.value)

# Save and plot performance metrics
np.save(f"{RunDirectories.METRIC_DATA.value}/episode_total_reward.npy", episode_total_reward)
np.save(f"{RunDirectories.METRIC_DATA.value}/episode_length.npy", episode_length)
np.save(
    f"{RunDirectories.METRIC_DATA.value}/episode_discounted_return_per_step.npy", episode_discounted_return_per_step
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plot_training_metrics_per_step(
        {AGENT_NAME: episode_total_reward}, window_size=100, metric_name="Total Reward",
        save_dir=RunDirectories.METRIC_PLOTS.value
    )
    plot_training_metrics_per_step(
        {AGENT_NAME: episode_length}, window_size=100, metric_name="Episode Length",
        save_dir=RunDirectories.METRIC_PLOTS.value
    )
    plot_training_metrics_per_step(
        {AGENT_NAME: episode_discounted_return_per_step}, window_size=100,
        metric_name="Discounted Return", save_dir=RunDirectories.METRIC_PLOTS.value
    )
