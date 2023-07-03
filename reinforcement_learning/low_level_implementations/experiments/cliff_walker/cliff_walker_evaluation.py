from utils import plot_training_metrics_per_step
import numpy as np
import pickle
from enum import Enum
import os
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from reinforcement_learning.low_level_implementations.tabular_q_learning.agents import Agent
from reinforcement_learning.low_level_implementations.tabular_q_learning.utils import EpsilonGreedySelector

# =============================================================================
# Settings
# =============================================================================

EVAL_NAME = "slippery_vs_non_slippery"


class EvalDirectories(Enum):
    PLOTS = f"./.cache/evaluations/plots/{EVAL_NAME}"
    VIDEOS = f"./.cache/evaluations/videos"


for directory in EvalDirectories:
    if not os.path.exists(directory.value):
        os.makedirs(directory.value)


# =============================================================================
# Specify runs to inspect
# =============================================================================

RUN_DIRECTORIES = {
    # LR sweep, slippery=False
    "learning_rate_0.1": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.1__df_0.9"
                         "__as_EpsilonGreedySelector(epsilon=0.1)__episodes_10000__is_slippery_False",
    # LR sweep, slippery=True
    "learning_rate_0.1_slippery": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.1__df_0.9"
                                  "__as_EpsilonGreedySelector(epsilon=0.1)__episodes_10000__is_slippery_True",

}

# =============================================================================
# Plot metrics
# =============================================================================

METRICS = [
    "episode_discounted_return_per_step",
    "episode_length",
    "episode_total_reward",
]

for metric in METRICS:

    # Load metrics
    metrics = {
        run_name: np.load(f"{run_directory}/data/metrics/{metric}.npy") for
        run_name, run_directory in RUN_DIRECTORIES.items()
    }

    # Plot
    plot_training_metrics_per_step(
        training_metrics=metrics,
        metric_name=metric,
        save_dir=EvalDirectories.PLOTS.value,
    )


# =============================================================================
# Load agent and render activity at stages of training loop
# =============================================================================

# Load pickled configs
configs = {}
for run_name, run_directory in RUN_DIRECTORIES.items():
    with open(f"{run_directory}/config.pkl", "rb") as f:
        configs[run_name] = pickle.load(f)


# For each config, get multi-episode videos of behaviour at the start, middle, and end of training
for run_name, config in configs.items():

    # Get mid-training Q-table
    save_freq = config["NUM_EPISODES"] // config["NUM_CHECKPOINTS"]
    mid_training_episode = save_freq * (config["NUM_CHECKPOINTS"] // 2)    # Done this way as checkpoints were worked
    # out with "//" operator, which returns an integer
    mid_training_episode = int(mid_training_episode)

    for training_episode in [0, mid_training_episode, config["NUM_EPISODES"]]:

        # Load environment (most immediately useful for defining the state and action space for instantiating the agent)
        env = gym.make('FrozenLake-v1', render_mode="rgb_array_list", is_slippery=config['IS_SLIPPERY'])
        # env = RecordVideo(env, video_folder=f"{EvalDirectories.VIDEOS.value}/episode_{training_episode}")

        # TODO: Load agent and render activity at stages of training loop
        agent = Agent(
            num_states=env.observation_space.n,
            num_actions=env.action_space.n,
            gamma=config['DISCOUNT_FACTOR'],
            action_selector=config['ACTION_SELECTOR'],
            alpha=config['LEARNING_RATE'],
        )

        # Load Q-table
        q_table = np.load(f"{RUN_DIRECTORIES[run_name]}/data/q_table/q_table_episode_{training_episode}.npy")
        agent.q_table = q_table

        # Want to record a video of 10 episodes of the agent's activity, at this stage of training
        episode_samples = []
        for episode in range(50):
            # Render activity
            state, info = env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.choose_action(state)
                state, reward, terminated, truncated, info = env.step(action)
            frames = env.render()
            episode_samples.extend(frames)

        save_video(
            frames=episode_samples,
            video_folder=f"{EvalDirectories.VIDEOS.value}/{run_name}/episode_{training_episode}/video.mp4",
            fps=10,
        )
