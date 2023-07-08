import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import warnings    # There's an annoying warning in matplotlib to suppress
import os
import pickle

from plotting import (plot_q_table, plot_training_metrics_multiple_trials, plot_v_table_with_arrows)

from reinforcement_learning.low_level_implementations.algorithms.agents import QLearningAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import SarsaAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import ExpectedSarsaAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import DoubleQLearningAgent

# from reinforcement_learning.low_level_implementations.algorithms.action_selection import EpsilonGreedySelector
# from reinforcement_learning.low_level_implementations.algorithms.action_selection import SoftmaxSelector

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
        self.config_filepath_dir = f"./.cache/{run_name}/config.pkl"
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


########################################################################################################################
# Run parameters
########################################################################################################################

# # Training configuration parameters
# config = {
#     # Environment parameters
#     "NUM_TRIALS": 5,
#     "NUM_EPISODES": 10000,
#     "MAX_STEPS_PER_EPISODE": 100,
#     "RENDER_MODE": "none",   # "human", "none"
#     "IS_SLIPPERY": False,
#     "NUM_CHECKPOINTS": 10,    # Per trial, for saving q-tables
#     "LAKE_SIZE": 8,   # If None, uses default 4x4 lake, else generates random lake of size LAKE_SIZE x LAKE_SIZE
#     # QLearningAgent parameters
#     "AGENT_TYPE": DoubleQLearningAgent,
#     "LEARNING_RATE": 0.1,
#     "DISCOUNT_FACTOR": 0.9,
#     "ACTION_SELECTOR": EpsilonGreedySelector(epsilon=0.1, decay_scheme="linear"),
# }
# save_freq = config["NUM_EPISODES"] // config["NUM_CHECKPOINTS"]
# run_name = f"{config['AGENT_TYPE'].__name__}__lr_{config['LEARNING_RATE']}__df_{config['DISCOUNT_FACTOR']}__" \
#            f"as_{config['ACTION_SELECTOR']}__episodes_{config['NUM_EPISODES']}__is_slippery_" \
#            f"{config['IS_SLIPPERY']}__map_size_{config['LAKE_SIZE']}"


# Training artefact directories

for config in TRAINING_CONFIGS:

    run_name = config.run_name

    NUM_EPISODES = config.environment_config.num_episodes
    NUM_TRIALS = config.environment_config.num_trials

    save_freq = NUM_EPISODES // config.environment_config.num_checkpoints

    # Directories object collates all directories and filepaths required for training loop
    directories = RunDirectories(run_name)


    #
    # # Save run configuration dictionary as pickle file
    # with open(f"./.cache/{run_name}/config.pkl", "wb") as f:
    #     pickle.dump(config, f)

    METRICS = [
        EpisodeReward(num_episodes=NUM_EPISODES, num_trials=NUM_TRIALS),
        EpisodeLength(num_episodes=NUM_EPISODES, num_trials=NUM_TRIALS),
        DiscountedReturn(num_episodes=NUM_EPISODES, num_trials=NUM_TRIALS),
        CumulativeReward(num_episodes=NUM_EPISODES, num_trials=NUM_TRIALS),
        CumulativeDiscountedReturn(num_episodes=NUM_EPISODES, num_trials=NUM_TRIALS),
    ]

    ####################################################################################################################
    # Training loop
    ####################################################################################################################
    #
    # experiment_metrics = {
    #     metric_name: np.zeros((NUM_TRIALS, NUM_EPISODES)) for metric_name in
    #     ["episode_total_reward", "episode_length", "episode_discounted_return_per_step"]
    # }

    for trial in range(NUM_TRIALS):

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
            Trial {trial + 1} of {NUM_TRIALS} started
            ############################################################################################################
            """
        )

        # Initialise performance metrics
        episode_total_reward = []
        episode_length = []
        episode_discounted_return_per_step = []
        for episode in range(NUM_EPISODES):

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
                print(f"Episode {episode} of {NUM_EPISODES} completed")
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
                           f"_{NUM_EPISODES}.npy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_q_table(
                agent.q_table,
                episode_num=NUM_EPISODES,
                save_dir=f"{directories.q_table_plots_dir}/trial_{trial}/"
            )
            plot_v_table_with_arrows(
                agent.q_table,
                episode_num=NUM_EPISODES,
                save_dir=f"{directories.v_table_plots_dir}/trial_{trial}/"
            )


    # Run `finalise` on metrics - necessary for some metrics e.g. cumulative, which require post-processing after trial
    [metric.finalise() for metric in METRICS]

    # Save and plot performance metrics
    [metric.save(save_dir=directories.metric_data_dir) for metric in METRICS]
    for metric in METRICS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_training_metrics_multiple_trials(
                {run_name: metric.values},
                metric_name=metric.save_name,
                save_dir=directories.metric_plots_dir,
                show_individual_trials=True
            )
