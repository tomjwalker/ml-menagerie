import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import warnings    # There's an annoying warning in matplotlib to suppress
from enum import Enum
import os
import pickle

from plotting import plot_q_table, plot_training_metrics_single_trial, plot_training_metrics_multiple_trials
from reinforcement_learning.low_level_implementations.tabular_q_learning.agents import Agent
from reinforcement_learning.low_level_implementations.tabular_q_learning.action_selection import (EpsilonGreedySelector,
                                                                                                  SoftmaxSelector)
from metrics import (EpisodeReward, EpisodeLength, DiscountedReturn, CumulativeReward, CumulativeDiscountedReturn)

########################################################################################################################
# Run parameters
########################################################################################################################


# Training configuration parameters
config = {
    # Environment parameters
    "NUM_TRIALS": 5,
    "NUM_EPISODES": 2000,
    "MAX_STEPS_PER_EPISODE": 100,
    "RENDER_MODE": "none",   # "human", "none"
    "IS_SLIPPERY": False,
    "NUM_CHECKPOINTS": 10,    # Per trial, for saving q-tables
    "LAKE_SIZE": 8,   # If None, uses default 4x4 lake, else generates random lake of size LAKE_SIZE x LAKE_SIZE
    # Agent parameters
    "AGENT_NAME": "tabular_q_learning",
    "LEARNING_RATE": 0.1,
    "DISCOUNT_FACTOR": 0.9,
    "ACTION_SELECTOR": EpsilonGreedySelector(epsilon=0.1, decay_scheme=None),
}
save_freq = config["NUM_EPISODES"] // config["NUM_CHECKPOINTS"]
run_name = f"{config['AGENT_NAME']}__lr_{config['LEARNING_RATE']}__df_{config['DISCOUNT_FACTOR']}__" \
           f"as_{config['ACTION_SELECTOR']}__episodes_{config['NUM_EPISODES']}__is_slippery_" \
           f"{config['IS_SLIPPERY']}__map_size_{config['LAKE_SIZE']}"


# Training artefact directories
class RunDirectories(Enum):
    Q_TABLE_DATA = f"./.cache/{run_name}/data/q_table"
    METRIC_DATA = f"./.cache/{run_name}/data/metrics"
    Q_TABLE_PLOTS = f"./.cache/{run_name}/plots/q_table"
    METRIC_PLOTS = f"./.cache/{run_name}/plots/metrics"


# Create directories if they don't exist
for directory in RunDirectories:
    if not os.path.exists(directory.value):
        os.makedirs(directory.value)

# Save run configuration dictionary as pickle file
with open(f"./.cache/{run_name}/config.pkl", "wb") as f:
    pickle.dump(config, f)


METRICS = [
    EpisodeReward(num_episodes=config['NUM_EPISODES'], num_trials=config['NUM_TRIALS']),
    EpisodeLength(num_episodes=config['NUM_EPISODES'], num_trials=config['NUM_TRIALS']),
    DiscountedReturn(num_episodes=config['NUM_EPISODES'], num_trials=config['NUM_TRIALS']),
    CumulativeReward(num_episodes=config['NUM_EPISODES'], num_trials=config['NUM_TRIALS']),
    CumulativeDiscountedReturn(num_episodes=config['NUM_EPISODES'], num_trials=config['NUM_TRIALS']),
]

########################################################################################################################
# Training loop
########################################################################################################################

experiment_metrics = {
    metric_name: np.zeros((config['NUM_TRIALS'], config['NUM_EPISODES'])) for metric_name in
    ["episode_total_reward", "episode_length", "episode_discounted_return_per_step"]
}

for trial in range(config['NUM_TRIALS']):

    print(
        "##############################################################################################################"
        f"Trial {trial + 1} of {config['NUM_TRIALS']} started"
        "##############################################################################################################"
    )

    # Instantiate the environment and agent
    if config["LAKE_SIZE"] is None:
        env = gym.make('FrozenLake-v1', render_mode=config['RENDER_MODE'], is_slippery=config['IS_SLIPPERY'])
    else:
        env = gym.make(
            'FrozenLake-v1',
            desc=generate_random_map(size=config["LAKE_SIZE"], p=0.8, seed=42),
            render_mode=config['RENDER_MODE'],
            is_slippery=config['IS_SLIPPERY']
        )
    agent = Agent(
        gamma=config['DISCOUNT_FACTOR'],
        alpha=config['LEARNING_RATE'],
        action_selector=config['ACTION_SELECTOR'],
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
            action = agent.choose_action(state, episode)

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
        [metric.update(episode_rewards, agent, episode=episode, trial=trial) for metric in METRICS]

        if episode % save_freq == 0:
            print(f"Episode {episode} of {config['NUM_EPISODES']} completed")
            # Save the Q-table to a file
            agent.save_q_table(f"{RunDirectories.Q_TABLE_DATA.value}/trial_{trial}/q_table_episode_{episode}.npy")

            # Plot the Q-table while suppressing the MatplotlibDeprecationWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_q_table(agent.q_table, episode_num=episode, save_dir=RunDirectories.Q_TABLE_PLOTS.value)

    # Save final checkpoint
    agent.save_q_table(f"{RunDirectories.Q_TABLE_DATA.value}/trial_{trial}/q_table_episode"
                       f"_{config['NUM_EPISODES']}.npy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_q_table(agent.q_table, episode_num=config['NUM_EPISODES'], save_dir=RunDirectories.Q_TABLE_PLOTS.value)


# Run `finalise` on metrics - necessary for some metrics e.g. cumulative, which require post-processing after trial
[metric.finalise() for metric in METRICS]

# Save and plot performance metrics
[metric.save(save_dir=RunDirectories.METRIC_DATA.value) for metric in METRICS]
for metric in METRICS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_training_metrics_multiple_trials(
            {run_name: metric.values},
            metric_name=metric.save_name,
            save_dir=RunDirectories.METRIC_PLOTS.value,
            show_individual_trials=True
        )
