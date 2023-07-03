import numpy as np


class EpsilonGreedySelector:

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, q_values_for_current_state):
        """

        Args:
            q_values_for_current_state: np.ndarray of shape (num_actions,), Q-values for the current state

        Returns:

        """
        # `flatnonzero` in combination with the inner `max` is used to handle multiple actions with the same value
        best_actions = np.flatnonzero(q_values_for_current_state == q_values_for_current_state.max())

        if np.random.rand() < self.epsilon:
            # Explore
            action = np.random.choice([*range(len(q_values_for_current_state))])
        else:
            # Exploit. If there are multiple actions with the same value (ties), choose one at random
            action = np.random.choice(best_actions)

        return action

    def __str__(self):
        return f"EpsilonGreedySelector(epsilon={self.epsilon})"
