from plotting import plot_training_metrics_multiple_trials
import numpy as np
import pickle
from enum import Enum
import os
from collections import defaultdict
import gymnasium as gym
from gymnasium.utils.save_video import save_video


# =============================================================================
# Evaluation settings
# =============================================================================

EVAL_NAME = "cliff_walker"

# Specify runs to inspect
RUN_DIRECTORIES = {
    "qlearning_eg_0_1": ".cache/6873c2c5-07b5-4f85-a71d-a37dedfcc7df",
    "sarsa_eg_0_1": ".cache/beec2c12-8ec0-4319-9db5-57b353d838bf",
}

METRICS_DIRECTORIES = {
    run_name: f"{run_directory}/data/metrics" for run_name, run_directory in RUN_DIRECTORIES.items()
}

X_LIMIT = 500    # None for no limit

MAKE_VIDEOS = True

mid_training_episode = 600    # If None, calculates mid-point intermediate episode


class EvalDirectories(Enum):
    PLOTS = f"./.cache/evaluations/{EVAL_NAME}/plots"
    VIDEOS = f"./.cache/evaluations/{EVAL_NAME}/videos"


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

        # Get a typical performing run from the trial with the median accumulated discounted reward
        cum_return_all_trials = metrics["cumulative_discounted_return"][run_name]
        final_scores = cum_return_all_trials[:, -1]
        median_trial = np.argmin(np.abs(final_scores - np.median(final_scores)))

        # Get mid-training episode, if not provided
        if mid_training_episode is None:
            save_freq = config.environment_config.num_episodes // config.environment_config.num_checkpoints
            mid_training_episode = save_freq * (config.environment_config.num_checkpoints // 2)
            # out with "//" operator, which returns an integer
            mid_training_episode = int(mid_training_episode)
        else:
            # Find closest episode to the provided mid-training episode
            save_freq = config.environment_config.num_episodes // config.environment_config.num_checkpoints
            mid_training_episode = save_freq * (mid_training_episode // save_freq)
            mid_training_episode = int(mid_training_episode)

        for training_episode in [0, mid_training_episode, config.environment_config.num_episodes]:

            # Load environment (most immediately useful for defining the state and action space for instantiating
            # the agent)
            config.environment_config.render_mode = "rgb_array_list"
            env = config.environment_config()

            agent = config.agent_config.agent_type(
                gamma=config.agent_config.discount_factor,
                alpha=config.agent_config.learning_rate,
                action_selector=config.agent_config.action_selector,
                num_states=env.observation_space.n,
                num_actions=env.action_space.n
            )

            # Load Q-table
            q_table = np.load(f"{RUN_DIRECTORIES[run_name]}/data/q_table/trial_{median_trial}/q_table_episode"
                              f"_{training_episode}.npy")
            agent.q_table = q_table

            # Want to record a video of 50 episodes of the agent's activity, at this stage of training
            episode_samples = []
            for episode in range(50):
                # Render activity
                state, info = env.reset()
                terminated = False
                truncated = False
                steps = 0
                while (not (terminated or truncated)) and steps < config.environment_config.max_steps_per_episode:
                    action = agent.choose_action(state, episode)
                    state, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                frames = env.render()
                episode_samples += frames

            save_video(
                frames=episode_samples,
                video_folder=f"{EvalDirectories.VIDEOS.value}/{run_name}/episode_{training_episode}/video.mp4",
                fps=3,
            )
