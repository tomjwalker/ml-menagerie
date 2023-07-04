import numpy as np
import os

from low_level_implementations.tabular_q_learning.action_selection import (EpsilonGreedySelector, SoftmaxSelector)



class Agent:
    """
    A tabular Q-learning agent

    Attributes
    - num_states: int, size of state space - sets a dimension of the Q table
    - num_actions: int, size of action space - sets a dimension of the Q table
    - gamma: float, discount factor
    - alpha: float, learning rate
    - action_selector: object, selects actions based on the Q-table
    - q_table: np.ndarray of shape (num_states, num_actions), initialised to zeros

    Methods
    - choose_action: select an action using an exploration-exploitation strategy (here: epsilon greedy)
    - update_q_table: update the Q-table based on the observed sample (state, action, next_state, reward)
    - learn: update the Q-table based on the observed sample (state, action, next_state, reward)
    - save_q_table: save the Q-table to a file
    - load_q_table: load the Q-table from a file
    """

    def __init__(self, num_states, num_actions, action_selector, gamma=0.9, alpha=0.1):
        self.num_states = num_states    # size of state space - sets a dimension of the Q table
        self.num_actions = num_actions    # size of action space - sets a dimension of the Q table
        self.gamma = gamma    # discount factor
        self.alpha = alpha    # learning rate
        self.action_selector = action_selector

        # Initialise Q table to zeros
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

    def choose_action(self, state, episode):
        """
        Select an action using an exploration-exploitation strategy
        """

        possible_action_values = self.q_table[state, :]

        action = self.action_selector(possible_action_values, episode)

        return action

    def update_q_table(self, state, action, next_state, reward):
        """
        Update the Q-table based on the observed sample (state, action, next_state, reward)
        """

        max_q_next_state = self.q_table[next_state, :].max()
        q_old = self.q_table[state, action]

        # Update equation for Q-learning
        self.q_table[state, action] = q_old + self.alpha * (reward + (self.gamma * max_q_next_state) - q_old)

    def learn(self, state, action, next_state, reward, done):
        """
        Perform one learning step
        1. Update Q-table based on the observed sample
        2. Adjust exploration rate or learning parameters (optional)
        """

        if not done:
            self.update_q_table(state, action, next_state, reward)
            # Can add an epsilon-tuning step here in the future

    def save_q_table(self, filepath="./cache/q_table.npy"):
        """
        Save Q-table to a file
        """
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        # Save Q-table as npy file
        np.save(filepath, self.q_table)

    def load_q_table(self, filepath):
        """
        Load Q-table from a file
        """
        self.q_table = np.load(filepath)
