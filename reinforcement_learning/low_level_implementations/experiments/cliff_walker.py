import gymnasium as gym
import matplotlib.pyplot as plt

from reinforcement_learning.low_level_implementations.tabular_q_learning.agent import Agent

########################################################################################################################
# Training loop parameters
########################################################################################################################

NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 100
SAVE_FREQ = 1000
RENDER_MODE = "none"    # "human" to render the environment/episode. "none" to turn off rendering
# Slippery version of environment has stochastic transitions: based on action, agent may move in a direction other
# than the one intended (with probability 1/3 for all directions except 180 degrees opposite action selection)
IS_SLIPPERY = False

########################################################################################################################
# Agent parameters
########################################################################################################################
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2


# Instantiate the environment and agent
env = gym.make('FrozenLake-v1', render_mode=RENDER_MODE, is_slippery=IS_SLIPPERY)
agent = Agent(
    gamma=DISCOUNT_FACTOR,
    alpha=LEARNING_RATE,
    epsilon=EXPLORATION_RATE,
    num_states=env.observation_space.n,
    num_actions=env.action_space.n
)

# Initialise performance metrics
episode_total_reward = []
episode_length = []
episode_discounted_return_per_step = []
for episode in range(NUM_EPISODES):

    # Reset the environment
    state, info = env.reset()

    # Run the episode to completion, or until the maximum number of steps is reached
    terminated = False
    truncated = False
    episode_rewards = []
    while not (terminated or truncated):

        # Agent action decision
        action = agent.choose_action(state)

        # Environment transition
        new_state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            print("here we go...!")

        # Agent learning
        agent.update_q_table(state, action, new_state, reward)

        if reward == 1:
            print(agent.q_table.sum())

        # Update state
        state = new_state

        # Record the reward
        episode_rewards.append(reward)

        if RENDER_MODE == "human":
            env.render()

    # Record performance metrics
    episode_total_reward.append(sum(episode_rewards))
    episode_length.append(len(episode_rewards))
    episode_discounted_return_per_step.append(
        sum([agent.gamma ** i * episode_rewards[i] for i in range(len(episode_rewards))]) / len(episode_rewards)
    )

    if episode % SAVE_FREQ == 0:
        print(f"Episode {episode} of {NUM_EPISODES} completed")
        # Save the Q-table to a file
        agent.save_q_table(f"./cache/q_table_episode_{episode}.npy")

        # Plot the Q-table
        plt.imshow(agent.q_table)
        plt.colorbar()
        plt.title(f"Q-table: episode {episode}")
        plt.show()

# Plot performance metrics
plt.plot(episode_total_reward)
plt.title("Episode total reward")
plt.show()

plt.plot(episode_length)
plt.title("Episode length")
plt.show()

plt.plot(episode_discounted_return_per_step)
plt.title("Episode discounted return per step")
plt.show()

