import uuid
import pickle
import os
import pandas as pd

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class EnvironmentConfig:
    def __init__(self, env_name, num_trials, num_episodes, max_steps_per_episode, render_mode, num_checkpoints):
        self.num_trials = num_trials
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.render_mode = render_mode   # "human", "none"
        self.num_checkpoints = num_checkpoints    # Per trial, for saving q-tables
        self.env_name = env_name

    def __call__(self):
        env = gym.make(self.env_name, render_mode=self.render_mode)
        return env


class FrozenLakeConfig(EnvironmentConfig):
    def __init__(self, env_name, num_trials, num_episodes, max_steps_per_episode, render_mode, num_checkpoints,
                 lake_size,
                 is_slippery):
        super().__init__(env_name, num_trials, num_episodes, max_steps_per_episode, render_mode, num_checkpoints)
        # Next attribute: if None, uses default 4x4 lake, else generates random lake of size LAKE_SIZE x LAKE_SIZE
        self.lake_size = lake_size
        self.is_slippery = is_slippery

    def __call__(self):
        if self.lake_size is None:
            env = gym.make('FrozenLake-v1', render_mode=self.render_mode, is_slippery=self.is_slippery)
        else:
            env = gym.make(
                'FrozenLake-v1',
                desc=generate_random_map(size=self.lake_size, p=0.8, seed=42),
                render_mode=self.render_mode,
                is_slippery=self.is_slippery
            )
        return env


class AgentConfig:
    def __init__(self, agent_type, learning_rate, discount_factor, action_selector):
        self.agent_type = agent_type
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.action_selector = action_selector


class TrainingConfig:
    """
    Combines EnvironmentConfig and AgentConfig.
    Generates a UUID run name.
    Has methods for pickling and unpickling itself (some attributes are Python objects).
    Also has a method for creating/updating a run log, (where any object attributes are represented as strings).
    """

    def __init__(self, environment_config: EnvironmentConfig, agent_config: AgentConfig):
        self.environment_config = environment_config
        self.agent_config = agent_config
        self.run_name: str = self.generate_run_name()

    def generate_run_name(self):
        return str(uuid.uuid4())

    def pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def unpickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def update_run_log(self, path):
        """
        Either creates or updates a run log at the given path.
        Uses pandas for load/save to csv.
        Any attributes which are Python objects are represented as strings.
        Checks to see if the run uuid already exists. If it does, it appends a number to the end of the uuid.
        """

        # Check if the run uuid already exists. If it does, append a number to the end of the uuid.
        run_name = self.run_name
        run_name_exists = True
        run_name_counter = 1
        while run_name_exists:
            if os.path.exists(path):
                run_log = pd.read_csv(path)
                if run_name in run_log["run_name"].values:
                    run_name = self.run_name + "_" + str(run_name_counter)
                    run_name_counter += 1
                else:
                    run_name_exists = False
            else:
                run_name_exists = False

        # Create a dataframe from the TrainingConfig object.
        # Any attributes which are Python objects are represented as strings.
        config_dict = {"run_name": self.run_name}
        # Add attributes from the environment and agent configs to config_dict.
        config_dict.update(vars(self.environment_config))
        config_dict.update(vars(self.agent_config))
        config_dict = {key: [str(value)] for key, value in config_dict.items()}
        config_df = pd.DataFrame.from_dict(config_dict)

        # Append the dataframe to the run log.
        if os.path.exists(path):
            run_log = pd.read_csv(path)
            # Append the dataframe to the run log via concat.
            run_log = pd.concat([run_log, config_df], axis=0)

        else:
            run_log = config_df
        run_log.to_csv(path, index=False)
