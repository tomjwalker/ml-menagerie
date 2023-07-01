from utils import plot_q_table, plot_training_metrics_per_step
import numpy as np

# =============================================================================
# Specify runs to inspect
# =============================================================================

run_directories = {
    # LR sweep, slippery=False
    "learning_rate_0.1": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.1__df_0.9__er_0.1__episodes_2000",
    "learning_rate_0.3": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.3__df_0.9__er_0.1__episodes_2000",
    "learning_rate_0.033": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.033__df_0.9__er_0.1__episodes_2000",
    # LR sweep, slippery=True
    "learning_rate_0.1_slippery": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.1__df_0.9__er_0.1__"
                                  "episodes_2000__is_slippery_True",
    "learning_rate_0.3_slippery": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.3__df_0.9__er_0.1__"
                                  "episodes_2000__is_slippery_True",
    "learning_rate_0.033_slippery": ".cache/tabular_q_learning__vanilla_epsilon_greedy__lr_0.033__df_0.9__er_0.1__"
                                    "episodes_2000__is_slippery_True",
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
