from plotting import plot_training_metrics_single_trial, plot_training_metrics_multiple_trials
import numpy as np
import pickle
from enum import Enum
import os
from collections import defaultdict
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from reinforcement_learning.low_level_implementations.algorithms.agents import QLearningAgent
from reinforcement_learning.low_level_implementations.algorithms.action_selection import EpsilonGreedySelector

# =============================================================================
# Evaluation settings
# =============================================================================

EVAL_NAME = "agent_sweep"

# # Specify runs to inspect
# RUN_DIRECTORIES = {
#     "learning_rate_0.1": ".cache/tabular_q_learning__lr_0.1__df_0.9__as_EpsilonGreedySelector"
#                          "__epsilon_0_1_decay_scheme_None__episodes_2000__is_slippery_False__map_size_8",
#     "learning_rate_0.1_eg_linear_decay": ".cache/tabular_q_learning__lr_0.1__df_0.9"
#                                          "__as_EpsilonGreedySelector__epsilon_0_1_decay_scheme_linear__episodes_2000"
#                                          "__is_slippery_False__map_size_8",
#     "learning_rate_0.1_eg_exponential_decay": ".cache/tabular_q_learning__lr_0.1__df_0.9"
#                                          "__as_EpsilonGreedySelector__epsilon_0_1_decay_scheme_exponential__episodes"
#                                               "_2000__is_slippery_False__map_size_8",
# }

# Specify runs to inspect
RUN_DIRECTORIES = {
    "qlearning_eg_linear_decay": ".cache/QLearningAgent__lr_0.1__df_0.9__as_EpsilonGreedySelector__"
                                 "epsilon_0_1_decay_scheme_linear__episodes_2000__is_slippery_False__map_size_8",
    "double_qlearning_eg_linear_decay": ".cache/DoubleQLearningAgent__lr_0.1__df_0.9__as_EpsilonGreedySelector__"
                                 "epsilon_0_1_decay_scheme_linear__episodes_10000__is_slippery_False__map_size_8",
    "sarsa_eg_linear_decay": ".cache/SarsaAgent__lr_0.1__df_0.9__as_EpsilonGreedySelector__"
                                 "epsilon_0_1_decay_scheme_linear__episodes_2000__is_slippery_False__map_size_8",
    "expected_sarsa_eg_linear_decay": ".cache/ExpectedSarsaAgent__lr_0.1__df_0.9__as_EpsilonGreedySelector__"
                                 "epsilon_0_1_decay_scheme_linear__episodes_2000__is_slippery_False__map_size_8",

}


METRICS_DIRECTORIES = {
    run_name: f"{run_directory}/data/metrics" for run_name, run_directory in RUN_DIRECTORIES.items()
}

X_LIMIT = 10000    # None for no limit

MAKE_VIDEOS = False


class EvalDirectories(Enum):
    PLOTS = f"./.cache/evaluations/plots/{EVAL_NAME}"
    VIDEOS = f"./.cache/evaluations/videos"


for directory in EvalDirectories:
    if not os.path.exists(directory.value):
        os.makedirs(directory.value)


# =============================================================================
# Plot metrics
# =============================================================================

# Load metrics
metrics = defaultdict(dict)
for run_name, metric_dir in METRICS_DIRECTORIES.items():
    # Loop over all files present in the metrics directory
    for metric_file in os.listdir(metric_dir):
        # Unpickle the metric
        with open(f"{metric_dir}/{metric_file}", "rb") as f:
            metric = pickle.load(f)
        if X_LIMIT is not None:
            metric.values = metric.values[:, :X_LIMIT]
        # Add to the dictionary of metrics
        metrics[metric.save_name][run_name] = metric.values

# Plot training metrics. Loop over all metrics, and plot all runs for each metric
for metric_name, runs_dict in metrics.items():
    plot_training_metrics_multiple_trials(
        metrics_over_multiple_trials=runs_dict,
        metric_name=metric_name,
        save_dir=EvalDirectories.PLOTS.value,
    )


# =============================================================================
# Load agent and render activity at stages of training loop
# =============================================================================

if MAKE_VIDEOS:

    # Load pickled configs
    configs = {}
    for run_name, run_directory in RUN_DIRECTORIES.items():
        with open(f"{run_directory}/config.pkl", "rb") as f:
            configs[run_name] = pickle.load(f)

    # For each config, get multi-episode videos of behaviour at the start, middle, and end of training
    for run_name, config in configs.items():

        # Get best performing run from the trial with the maximum accumulated discounted reward
        cum_return_all_trials = metrics["cumulative_discounted_return"][run_name]
        best_trial = np.argmax(cum_return_all_trials[:, -1])

        # Get mid-training Q-table
        save_freq = config["num_episodes"] // config["NUM_CHECKPOINTS"]
        mid_training_episode = save_freq * (config["NUM_CHECKPOINTS"] // 2)    # Done this way as checkpoints were worked
        # out with "//" operator, which returns an integer
        mid_training_episode = int(mid_training_episode)

        for training_episode in [0, mid_training_episode, config["num_episodes"]]:

            # Load environment (most immediately useful for defining the state and action space for instantiating the agent)
            if config.get("LAKE_SIZE") is not None:
                env = gym.make(
                    'FrozenLake-v1',
                    desc=generate_random_map(size=config["LAKE_SIZE"], p=0.8, seed=42),
                    render_mode="rgb_array_list",
                    is_slippery=config['IS_SLIPPERY']
                )
            else:
                env = gym.make('FrozenLake-v1', render_mode="rgb_array_list", is_slippery=config['IS_SLIPPERY'])

            # TODO: Load agent and render activity at stages of training loop
            agent = QLearningAgent(
                num_states=env.observation_space.n,
                num_actions=env.action_space.n,
                action_selector=config['ACTION_SELECTOR'],
                gamma=config['DISCOUNT_FACTOR'],
                alpha=config['LEARNING_RATE'],
            )

            # Load Q-table
            q_table = np.load(f"{RUN_DIRECTORIES[run_name]}/data/q_table/trial_{best_trial}/q_table_episode"
                              f"_{training_episode}.npy")
            agent.q_table = q_table

            # Want to record a video of 10 episodes of the agent's activity, at this stage of training
            episode_samples = []
            for episode in range(50):
                # Render activity
                state, info = env.reset()
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    action = agent.choose_action(state, episode)
                    state, reward, terminated, truncated, info = env.step(action)
                frames = env.render()
                episode_samples.extend(frames)

            save_video(
                frames=episode_samples,
                video_folder=f"{EvalDirectories.VIDEOS.value}/{run_name}/episode_{training_episode}/video.mp4",
                fps=10,
            )
