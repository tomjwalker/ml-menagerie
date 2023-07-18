import numpy as np


class ActionSelector:

    def __call__(self, q_values_for_current_state, episode):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def get_probabilities(self, q_values_for_current_state, episode):
        raise NotImplementedError


class EpsilonGreedySelector(ActionSelector):

    def __init__(self, epsilon=0.1, decay_scheme=None, **kwargs):
        """
        Args:
            epsilon: (starting) value of epsilon
            decay_scheme: None or str, if None then epsilon is fixed, otherwise epsilon is decayed according to the
                specified scheme.
            **kwargs: Used to pass parameters to the decay scheme. For example, if `decay_scheme` is "linear", then
                `num_episodes_window` and `decay_rate` can be passed as keyword arguments.
        """
        self.epsilon = epsilon
        self.decay_scheme = decay_scheme
        self.decay_params = kwargs

    def linear_decay(self, episode):
        num_episodes = self.decay_params.get("num_episodes_window", None)    # sets limit at which epsilon stops decaying
        decay_rate = self.decay_params.get("decay_rate", 0.01)    # e.g. 0.01 will reduce epsilon by 1% each episode
        final_epsilon = self.decay_params.get("final_epsilon", 0)
        if num_episodes is None and decay_rate is None:
            raise ValueError("Must provide either `num_episodes_window` or `decay_rate` to `linear_decay`")

        if num_episodes is not None:
            self.epsilon = max((1 - (episode / num_episodes)) * self.epsilon, final_epsilon)
        else:
            self.epsilon = max((1 - (decay_rate * episode)) * self.epsilon, final_epsilon)

    def exponential_decay(self, episode):
        decay_rate = self.decay_params.get("decay_rate", 0.95)
        final_epsilon = self.decay_params.get("final_epsilon", 0.001)
        if decay_rate is None:
            raise ValueError("Must provide `decay_rate` to `exponential_decay`")

        self.epsilon = max((decay_rate ** episode) * self.epsilon, final_epsilon)

    def update_epsilon(self, episode):
        if self.decay_scheme is None:
            return
        elif self.decay_scheme == "linear":
            self.linear_decay(episode)
        elif self.decay_scheme == "exponential":
            self.exponential_decay(episode)
        else:
            raise ValueError(f"Unknown decay scheme: {self.decay_scheme}")

    def __call__(self, q_values_for_current_state, episode):
        """
        Args:
            q_values_for_current_state: np.ndarray of shape (num_actions,), Q-values for the current state

        Returns: int, action to take
        """

        self.update_epsilon(episode)

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
        return f"EpsilonGreedySelector__epsilon_{self.epsilon}_decay_scheme_{self.decay_scheme}".replace(".", "_")

    def get_probabilities(self, q_values_for_current_state, episode):

        self.update_epsilon(episode)
        num_actions = len(q_values_for_current_state)

        # Get best action(s) (ties are broken randomly)
        best_actions = np.flatnonzero(q_values_for_current_state == q_values_for_current_state.max())
        probabilities = np.zeros(num_actions)
        probabilities[best_actions] = (1 - self.epsilon) / len(best_actions)
        probabilities += self.epsilon / num_actions

        return probabilities


class SoftmaxSelector(ActionSelector):

    # TODO: Check suspicious behaviour of this selector

    def __init__(self, temperature=0.2):
        self.temperature = temperature

    def __call__(self, q_values_for_current_state, episode):

        # Compute the softmax probabilities
        probabilities = np.exp(q_values_for_current_state / self.temperature)
        probabilities = probabilities / np.sum(probabilities)

        # Select action using softmax probabilities
        action = np.random.choice([*range(len(q_values_for_current_state))], p=probabilities)

        return action

    def __str__(self):
        return f"SoftmaxSelector(temperature={self.temperature})"

    def get_probabilities(self, q_values_for_current_state, episode):

        # Compute the softmax probabilities
        probabilities = np.exp(q_values_for_current_state / self.temperature)
        probabilities = probabilities / np.sum(probabilities)

        return probabilities
