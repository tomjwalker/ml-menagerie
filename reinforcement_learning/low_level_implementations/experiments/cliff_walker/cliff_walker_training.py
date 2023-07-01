import gymnasium as gym
import numpy as np
import warnings    # There's an annoying warning in matplotlib to suppress
from enum import Enum
import os

from utils import plot_q_table, plot_training_metrics_per_step
from reinforcement_learning.low_level_implementations.tabular_q_learning.agent import Agent


########################################################################################################################
# Run parameters
########################################################################################################################

# Environment parameters
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 100
RENDER_MODE = "none"    # "human" to render the environment/episode. "none" to turn off rendering
# Slippery version of environment has stochastic transitions: based on action, agent may move in a direction other
# than the one intended (with probability 1/3 for all directions except 180 degrees opposite action selection)
IS_SLIPPERY = True
save_freq = NUM_EPISODES // 10

# Agent parameters
AGENT_NAME = "tabular_q_learning__vanilla_epsilon_greedy"
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1

run_name = f"{AGENT_NAME}__lr_{LEARNING_RATE}__df_{DISCOUNT_FACTOR}__er_{EXPLORATION_RATE}__episodes_{NUM_EPISODES}" \
           f"__is_slippery_{IS_SLIPPERY}"


# Training artefact directories
class RunDirectories(Enum):
    Q_TABLE_DATA = f"./.cache/{run_name}/data/q_table"
    METRIC_DATA = f"./.cache/{run_name}/data/metrics"
    Q_TABLE_PLOTS = f"./.cache/{run_name}/plots/q_table"
    METRIC_PLOTS = f"./.cache/{run_name}/plots/metrics"


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

    if episode % save_freq == 0:
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
        {run_name: episode_total_reward}, window_size=100, metric_name="Total Reward",
        save_dir=RunDirectories.METRIC_PLOTS.value
    )
    plot_training_metrics_per_step(
        {run_name: episode_length}, window_size=100, metric_name="Episode Length",
        save_dir=RunDirectories.METRIC_PLOTS.value
    )
    plot_training_metrics_per_step(
        {run_name: episode_discounted_return_per_step}, window_size=100,
        metric_name="Discounted Return", save_dir=RunDirectories.METRIC_PLOTS.value
    )
