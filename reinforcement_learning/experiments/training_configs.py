from configs import CliffWalkingConfig, AgentConfig, TrainingConfig

from algorithms.agents import QLearningAgent
from algorithms.agents import SarsaAgent
from algorithms.agents import ExpectedSarsaAgent
from algorithms.agents import DoubleQLearningAgent

from algorithms.action_selection import EpsilonGreedySelector

# =============================================================================
# Define training configs
# =============================================================================

# Define environment configs
ENVIRONMENT_CONFIGS = [
    # FrozenLakeConfig(
    #     num_trials=10,
    #     num_episodes=5000,
    #     max_steps_per_episode=100,
    #     render_mode="none",
    #     num_checkpoints=10,
    #     lake_size=8,
    #     is_slippery=False
    # ),
    CliffWalkingConfig(
        num_trials=10,
        num_episodes=5000,
        max_steps_per_episode=100,
        render_mode="none",
        num_checkpoints=10,
    )
]

# Define agent configs
AGENT_CONFIGS = [
    AgentConfig(
        agent_type=QLearningAgent,
        learning_rate=0.1,
        discount_factor=0.9,
        action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme=None)
    ),
    AgentConfig(
        agent_type=SarsaAgent,
        learning_rate=0.1,
        discount_factor=0.9,
        action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme=None)
    ),
    # AgentConfig(
    #     agent_type=DoubleQLearningAgent,
    #     learning_rate=0.1,
    #     discount_factor=0.9,
    #     action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme=None)
    # ),
    # AgentConfig(
    #     agent_type=ExpectedSarsaAgent,
    #     learning_rate=0.1,
    #     discount_factor=0.9,
    #     action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme=None)
    # ),
]

# Define training configs
TRAINING_CONFIGS = []
for environment_config in ENVIRONMENT_CONFIGS:
    for agent_config in AGENT_CONFIGS:
        training_config = TrainingConfig(
            environment_config=environment_config,
            agent_config=agent_config
        )
        TRAINING_CONFIGS.append(training_config)

