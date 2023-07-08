import numpy as np
import os


class QLearningAgent:
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


class DoubleQLearningAgent:
    def __init__(self, num_states, num_actions, action_selector, gamma=0.9, alpha=0.1):
        self.num_states = num_states    # size of state space - sets a dimension of the Q table
        self.num_actions = num_actions    # size of action space - sets a dimension of the Q table
        self.gamma = gamma    # discount factor
        self.alpha = alpha    # learning rate
        self.action_selector = action_selector

        # Initialise Q table to zeros
        self.q_table_1 = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.q_table_2 = np.zeros((self.num_states, self.num_actions), dtype=float)

    def choose_action(self, state, episode):
        """
        Select an action using an exploration-exploitation strategy
        """

        # Sutton & Barto pp. 136 - behaviour policy aggregation of estimates e.g. sum or mean
        aggregated_q_table = self.q_table_1 + self.q_table_2

        possible_action_values = aggregated_q_table[state, :]

        action = self.action_selector(possible_action_values, episode)

        return action

    def update_q_table(self, state, action, next_state, reward):
        """
        Update the Q-table based on the observed sample (state, action, next_state, reward)
        """

        coin_flip = np.random.uniform(0, 1) > 0.5

        if coin_flip:
            ## Q1 update version
            # Next couple of lines are robust to ties (multiple action maxes)
            actions_max_q1 = np.where(self.q_table_1[next_state, :] == np.max(self.q_table_1[next_state, :]))[0]
            action_max_q1 = np.random.choice(actions_max_q1)
            max_q_next_state = self.q_table_2[next_state, action_max_q1].max()
            q_old = self.q_table_1[state, action]
            self.q_table_1[state, action] = q_old + self.alpha * (reward + (self.gamma * max_q_next_state) - q_old)
        else:
            ## Q2 update version
            # Next couple of lines are robust to ties (multiple action maxes)
            actions_max_q2 = np.where(self.q_table_2[next_state, :] == np.max(self.q_table_2[next_state, :]))[0]
            action_max_q2 = np.random.choice(actions_max_q2)
            max_q_next_state = self.q_table_1[next_state, action_max_q2].max()
            q_old = self.q_table_2[state, action]
            self.q_table_2[state, action] = q_old + self.alpha * (reward + (self.gamma * max_q_next_state) - q_old)

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


class SarsaAgent:

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

    def update_q_table(self, state, action, next_state, reward, episode):
        """
        Update the Q-table based on the observed sample (state, action, next_state, reward)
        """

        next_action = self.choose_action(next_state, episode)

        q_old = self.q_table[state, action]
        q_old_next_sa = self.q_table[next_state, next_action]

        # Update equation for Q-learning
        self.q_table[state, action] = q_old + self.alpha * (reward + (self.gamma * q_old_next_sa) - q_old)

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


class ExpectedSarsaAgent:

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

    def update_q_table(self, state, action, next_state, reward, episode):
        """
        Update the Q-table based on the observed sample (state, action, next_state, reward)

        Expected Sarsa implementation: Sutton & Barto pp. 133
        """

        q_next_states = self.q_table[next_state, :]
        policy = self.action_selector.get_probabilities(q_next_states, episode)
        expected_q_next_state = np.sum(q_next_states * policy)

        q_old = self.q_table[state, action]

        # Update equation for Q-learning
        self.q_table[state, action] = q_old + self.alpha * (reward + (self.gamma * expected_q_next_state) - q_old)

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
