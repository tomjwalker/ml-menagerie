import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import warnings    # There's an annoying warning in matplotlib to suppress
import os
import pandas as pd

from plotting import (plot_q_table, plot_training_metrics_multiple_trials, plot_v_table_with_arrows)

from reinforcement_learning.low_level_implementations.algorithms.agents import QLearningAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import SarsaAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import ExpectedSarsaAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import DoubleQLearningAgent

from metrics import (EpisodeReward, EpisodeLength, DiscountedReturn, CumulativeReward, CumulativeDiscountedReturn)
from training_configs import TRAINING_CONFIGS


########################################################################################################################
# Helper functions
########################################################################################################################
class RunDirectories:
    def __init__(self, run_name):

        self.run_name = run_name

        self.q_table_data_dir = f"./.cache/{run_name}/data/q_table"
        self.metric_data_dir = f"./.cache/{run_name}/data/metrics"
        self.q_table_plots_dir = f"./.cache/{run_name}/plots/q_table"
        self.v_table_plots_dir = f"./.cache/{run_name}/plots/v_table"
        self.metric_plots_dir = f"./.cache/{run_name}/plots/metrics"

        self.config_filepath = f"./.cache/{run_name}/config.pkl"
        self.runlog_filepath = f"./.cache/run_log.csv"

        self.create_directories()

    def is_directory(self, attribute_name):
        path = getattr(self, attribute_name)
        path = os.path.normpath(path)  # Normalize the path. This checks whether string is right format
        return os.path.isdir(path)

    def create_directories(self):

        # Create directories if they don't exist
        for attribute_name in self.__dict__:
            if attribute_name.startswith('_'):
                continue  # Skip private attributes

            attribute = getattr(self, attribute_name)
            if (not os.path.exists(attribute)) and ("_dir" in attribute_name):
                os.makedirs(attribute)


def add_metrics_to_runlog(summary_metrics_dict, directories_obj, clean_runlog=True):
    """
    Adds metric data from across all trials to the run log csv
    If clean_runlog == True, will remove all rows in the runlog which don't have any run data (after merging the
    current summary_metrics)
    """
    summary_metrics = pd.Series(summary_metrics_dict)
    summary_metrics.name = run_name
    summary_metrics = pd.DataFrame(summary_metrics).T
    run_log = pd.read_csv(directories_obj.runlog_filepath)
    run_log = pd.merge(run_log, summary_metrics, how="left", on="run_name")

    if clean_runlog:
        # TODO: more robust way to check which cols are metric data
        metric_cols = ["best_" in column for column in run_log.columns]
        metric_data = run_log.iloc[:, metric_cols]
        no_run_data = metric_data.isna().all(axis=1)
        run_log = run_log[~no_run_data]

    run_log.to_csv(directories_obj.runlog_filepath)

########################################################################################################################
# Run parameters
########################################################################################################################


for config in TRAINING_CONFIGS:

    run_name = config.run_name
    num_episodes = config.environment_config.num_episodes
    num_trials = config.environment_config.num_trials
    save_freq = num_episodes // config.environment_config.num_checkpoints

    # Directories object collates all directories and filepaths required for training loop
    directories = RunDirectories(run_name)

    METRICS = [
        EpisodeReward(num_episodes=num_episodes, num_trials=num_trials),
        EpisodeLength(num_episodes=num_episodes, num_trials=num_trials),
        DiscountedReturn(num_episodes=num_episodes, num_trials=num_trials),
        CumulativeReward(num_episodes=num_episodes, num_trials=num_trials),
        CumulativeDiscountedReturn(num_episodes=num_episodes, num_trials=num_trials),
    ]

    # Save run log and config pickle
    config.update_run_log(path=directories.runlog_filepath)
    config.pickle(path=directories.config_filepath)

    ####################################################################################################################
    # Training loop
    ####################################################################################################################

    for trial in range(num_trials):

        # Instantiate the environment and agent
        if config.environment_config.lake_size is None:
            env = gym.make('FrozenLake-v1', render_mode=config.environment_config.render_mode,
                           is_slippery=config.environment_config.is_slippery)
        else:
            env = gym.make(
                'FrozenLake-v1',
                desc=generate_random_map(size=config.environment_config.lake_size, p=0.8, seed=42),
                render_mode=config.environment_config.render_mode,
                is_slippery=config.environment_config.is_slippery
            )
        agent = config.agent_config.agent_type(
            gamma=config.agent_config.discount_factor,
            alpha=config.agent_config.learning_rate,
            action_selector=config.agent_config.action_selector,
            num_states=env.observation_space.n,
            num_actions=env.action_space.n
        )

        print(
            f"""
            ############################################################################################################
            Run {run_name}. 
            Agent: {agent}. 
            Environment: {env}
            Trial {trial + 1} of {num_trials} started
            ############################################################################################################
            """
        )

        # Initialise performance metrics
        episode_total_reward = []
        episode_length = []
        episode_discounted_return_per_step = []
        for episode in range(num_episodes):

            # Reset the environment
            state, info = env.reset()

            # Run the training_episode to completion, or until the maximum number of steps is reached
            terminated = False
            truncated = False
            episode_rewards = []
            while not (terminated or truncated):

                # QLearningAgent action decision
                action = agent.choose_action(state, episode)

                # Environment transition
                new_state, reward, terminated, truncated, info = env.step(action)

                # Learning. Different agents have different update rule signatures
                if isinstance(agent, (QLearningAgent, DoubleQLearningAgent)):
                    agent.update_q_table(state, action, new_state, reward)
                elif isinstance(agent, (SarsaAgent, ExpectedSarsaAgent)):
                    agent.update_q_table(state, action, new_state, reward, episode)
                else:
                    agent_type = type(agent).__name__
                    raise ValueError(f"Unsupported agent type: {agent_type}.")

                # Update state
                state = new_state

                # Record the reward
                episode_rewards.append(reward)

                if config.environment_config.render_mode == "human":
                    env.render()

            # Record performance metrics
            [metric.update(episode_rewards, agent, episode=episode, trial=trial) for metric in METRICS]

            if episode % save_freq == 0:
                print(f"Episode {episode} of {num_episodes} completed")
                # Save the Q-table to a file
                agent.save_q_table(f"{directories.q_table_data_dir}/trial_{trial}/q_table_episode_{episode}.npy")

                # Plot the Q-table while suppressing the MatplotlibDeprecationWarning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plot_q_table(
                        agent.q_table,
                        episode_num=episode,
                        save_dir=f"{directories.q_table_plots_dir}/trial_{trial}/"
                    )
                    plot_v_table_with_arrows(
                        agent.q_table,
                        episode_num=episode,
                        save_dir=f"{directories.v_table_plots_dir}/trial_{trial}/"
                    )

        # Save final checkpoint
        agent.save_q_table(f"{directories.q_table_data_dir}/trial_{trial}/q_table_episode"
                           f"_{num_episodes}.npy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_q_table(
                agent.q_table,
                episode_num=num_episodes,
                save_dir=f"{directories.q_table_plots_dir}/trial_{trial}/"
            )
            plot_v_table_with_arrows(
                agent.q_table,
                episode_num=num_episodes,
                save_dir=f"{directories.v_table_plots_dir}/trial_{trial}/"
            )

    # Run `finalise` on metrics - necessary for some metrics e.g. cumulative, which require post-processing after trial
    [metric.finalise() for metric in METRICS]

    # Save and plot performance metrics
    [metric.save(save_dir=directories.metric_data_dir) for metric in METRICS]

    summary_metrics = {}
    for metric in METRICS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_training_metrics_multiple_trials(
                {run_name: metric.values},
                metric_name=metric.save_name,
                save_dir=directories.metric_plots_dir,
                show_individual_trials=True
            )
        summary_metrics.update(metric.summarise())

    add_metrics_to_runlog(summary_metrics, directories, clean_runlog=True)
