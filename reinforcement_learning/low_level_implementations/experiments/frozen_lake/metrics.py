import numpy as np
import os
import pickle


class RLMetric:
    """
    Superclass for reinforcement learning metrics.
    """

    def __init__(self, num_episodes, num_trials=None):
        """
        Initialize the RLMetric object.

        Args:
            num_episodes (int): The number of episodes.
            num_trials (int, optional): The number of trials. Defaults to None (1 trial).
        """
        self.save_name = None
        self.render_name = None
        self.num_episodes = num_episodes
        if num_trials is None:
            num_trials = 1
        self.num_trials = num_trials
        self.values = np.zeros((num_trials, num_episodes))

    def update(self, episode_rewards, agent, episode, trial=None):
        """
        Run after each episode to update the metric value.

        Args:
            episode_rewards (list): Rewards obtained in the episode.
            agent: The RL agent.
            episode (int): The episode number.
            trial (int, optional): The trial number. Defaults to None (corresponding to 1 trial).
        """
        if trial is None:
            trial = 0
        # raise NotImplementedError("Subclasses must implement the `update` method.")

    def finalise(self):
        """
        Run after all episodes of a trial have been completed.
        This method can be overridden by subclasses if post-processing is required.
        """
        pass

    def save(self, save_dir):
        """
        Save the metric values to a file.

        Args:
            save_dir (str): The directory to save the file in.
        """
        save_path = os.path.join(save_dir, self.save_name + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def summarise(self):

        mean_across_trials = self.values.mean(axis=0)
        best_score_across_trials = mean_across_trials.max()
        # For argmax, pick last rather than first if ties, assuming variance will decrease (e.g. reducing epsilon)
        best_episode = np.where(mean_across_trials == best_score_across_trials)[0][-1]
        standard_deviation = self.values.std(axis=0)[best_episode]

        metrics = {
            f"best_avg_{self.save_name}": best_score_across_trials,
            f"best_std_{self.save_name}": standard_deviation,
        }

        return metrics


class EpisodeReward(RLMetric):
    """
    Metric to track the total reward obtained in each episode.
    """

    def __init__(self, num_episodes, num_trials=None):
        super().__init__(num_episodes, num_trials)
        self.save_name = "episode_reward"
        self.render_name = "Episode Reward"

    def update(self, episode_rewards, agent, episode, trial=None):
        super().update(episode_rewards, agent, episode, trial)
        total_reward = sum(episode_rewards)
        self.values[trial, episode] = total_reward


class EpisodeLength(RLMetric):
    """
    Metric to track the length of each episode.
    """

    def __init__(self, num_episodes, num_trials=None):
        super().__init__(num_episodes, num_trials)
        self.save_name = "episode_length"
        self.render_name = "Episode Length"

    def update(self, episode_rewards, agent, episode, trial=None):
        super().update(episode_rewards, agent, episode, trial)
        episode_length = len(episode_rewards)
        self.values[trial, episode] = episode_length


class DiscountedReturn(RLMetric):
    """
    Metric to track the discounted return obtained in each episode.
    """

    def __init__(self, num_episodes, num_trials=None, per_step=False):
        super().__init__(num_episodes, num_trials)
        if per_step:
            self.save_name = "episode_discounted_return_per_step"
            self.render_name = "Discounted Return Per Step"
        else:
            self.save_name = "discounted_return"
            self.render_name = "Discounted Return"
        self.per_step = per_step

    def update(self, episode_rewards, agent, episode, trial=None):
        super().update(episode_rewards, agent, episode, trial)
        discounted_reward = sum([agent.gamma ** i * episode_rewards[i] for i in range(len(episode_rewards))])
        if self.per_step:
            discounted_reward /= len(episode_rewards)
        self.values[trial, episode] = discounted_reward


class CumulativeReward(EpisodeReward):
    """
    Metric to track the cumulative reward obtained across episodes in a trial.
    """

    def __init__(self, num_episodes, num_trials=None):
        super().__init__(num_episodes, num_trials)
        self.save_name = "cumulative_reward"
        self.render_name = "Cumulative Reward"

    def finalise(self):
        """
        Run after all episodes of a trial have been completed.
        """
        self.values = np.cumsum(self.values, axis=1)


class CumulativeDiscountedReturn(DiscountedReturn):
    """
    Metric to track the cumulative discounted return obtained across episodes in a trial.
    """

    def __init__(self, num_episodes, num_trials=None, per_step=False):
        super().__init__(num_episodes, num_trials, per_step)
        if per_step:
            self.save_name = "cumulative_discounted_return_per_step"
            self.render_name = "Cumulative Discounted Return Per Step"
        else:
            self.save_name = "cumulative_discounted_return"
            self.render_name = "Cumulative Discounted Return"

    def finalise(self):
        """
        Run after all episodes of a trial have been completed.
        """
        self.values = np.cumsum(self.values, axis=1)
