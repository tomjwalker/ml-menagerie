import gymnasium as gym
import numpy as np
import warnings    # There's an annoying warning in matplotlib to suppress
from enum import Enum
import os
import pickle

from utils import plot_q_table, plot_training_metrics_per_step
from reinforcement_learning.low_level_implementations.tabular_q_learning.agents import Agent
from reinforcement_learning.low_level_implementations.tabular_q_learning.utils import EpsilonGreedySelector


########################################################################################################################
# Run parameters
########################################################################################################################

# Training configuration parameters
config = {
    # Environment parameters
    "NUM_EPISODES": 10000,
    "MAX_STEPS_PER_EPISODE": 100,
    "RENDER_MODE": "none",   # "human", "none"
    "IS_SLIPPERY": True,
    "NUM_CHECKPOINTS": 10,
    # Agent parameters
    "AGENT_NAME": "tabular_q_learning__vanilla_epsilon_greedy",
    "LEARNING_RATE": 0.1,
    "DISCOUNT_FACTOR": 0.9,
    "ACTION_SELECTOR": EpsilonGreedySelector(epsilon=0.1),
}

save_freq = config["NUM_EPISODES"] // config["NUM_CHECKPOINTS"]

run_name = f"{config['AGENT_NAME']}__lr_{config['LEARNING_RATE']}__df_{config['DISCOUNT_FACTOR']}__" \
           f"as_{config['ACTION_SELECTOR']}__episodes_{config['NUM_EPISODES']}__is_slippery_{config['IS_SLIPPERY']}"


# Training artefact directories
class RunDirectories(Enum):
    Q_TABLE_DATA = f"./.cache/{run_name}/data/q_table"
    METRIC_DATA = f"./.cache/{run_name}/data/metrics"
    Q_TABLE_PLOTS = f"./.cache/{run_name}/plots/q_table"
    METRIC_PLOTS = f"./.cache/{run_name}/plots/metrics"


for directory in RunDirectories:
    if not os.path.exists(directory.value):
        os.makedirs(directory.value)

# Save run configuration dictionary as pickle file
with open(f"./.cache/{run_name}/config.pkl", "wb") as f:
    pickle.dump(config, f)


########################################################################################################################
# Training loop
########################################################################################################################

# Instantiate the environment and agent
env = gym.make('FrozenLake-v1', render_mode=config['RENDER_MODE'], is_slippery=config['IS_SLIPPERY'])
agent = Agent(
    gamma=config['DISCOUNT_FACTOR'],
    alpha=config['LEARNING_RATE'],

    num_states=env.observation_space.n,
    num_actions=env.action_space.n
)

# Initialise performance metrics
episode_total_reward = []
episode_length = []
episode_discounted_return_per_step = []
for episode in range(config['NUM_EPISODES']):

    # Reset the environment
    state, info = env.reset()

    # Run the training_episode to completion, or until the maximum number of steps is reached
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

        if config['RENDER_MODE'] == "human":
            env.render()

    # Record performance metrics
    episode_total_reward.append(sum(episode_rewards))
    episode_length.append(len(episode_rewards))
    episode_discounted_return_per_step.append(
        sum([agent.gamma ** i * episode_rewards[i] for i in range(len(episode_rewards))]) / len(episode_rewards)
    )

    if episode % save_freq == 0:
        print(f"Episode {episode} of {config['NUM_EPISODES']} completed")
        # Save the Q-table to a file
        agent.save_q_table(f"{RunDirectories.Q_TABLE_DATA.value}/q_table_episode_{episode}.npy")

        # Plot the Q-table while suppressing the MatplotlibDeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_q_table(agent.q_table, episode_num=episode, save_dir=RunDirectories.Q_TABLE_PLOTS.value)

# Save final checkpoint
agent.save_q_table(f"{RunDirectories.Q_TABLE_DATA.value}/q_table_episode_{config['NUM_EPISODES']}.npy")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plot_q_table(agent.q_table, episode_num=config['NUM_EPISODES'], save_dir=RunDirectories.Q_TABLE_PLOTS.value)

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
