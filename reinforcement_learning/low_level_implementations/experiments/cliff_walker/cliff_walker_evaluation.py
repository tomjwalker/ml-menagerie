from utils import plot_training_metrics_per_step
import numpy as np
import json
import gymnasium as gym

from reinforcement_learning.low_level_implementations.tabular_q_learning.agent import Agent

# =============================================================================
# Specify runs to inspect
# =============================================================================

run_directories = {
    # LR sweep, slippery=False
    "learning_rate_0.1": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.1__df_0.9__er_0.1__"
                         "episodes_10000__is_slippery_False",
    # LR sweep, slippery=True
    "learning_rate_0.1_slippery": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.1__df_0.9__er_0.1__"
                                  "episodes_10000__is_slippery_True",

}

# =============================================================================
# Plot metrics
# =============================================================================

# Discounted return per episode
discounted_return_per_episode = {
    run_name: np.load(f"{run_directory}/data/metrics/episode_discounted_return_per_step.npy") for
    run_name, run_directory in run_directories.items()
}

plot_training_metrics_per_step(
    training_metrics=discounted_return_per_episode,
    metric_name="discounted_return_per_step",
    save_dir=None,    # Show rather than save
)

# =============================================================================
# Load agent and render activity at stages of training loop
# =============================================================================

# Load config JSONs
configs = {
    run_name: json.load(open(f"{run_directory}/config.json", "r")) for
    run_name, run_directory in run_directories.items()
}

for run_name, config in configs.items():
    print(config)

    # TODO: Load agent and render activity at stages of training loop


