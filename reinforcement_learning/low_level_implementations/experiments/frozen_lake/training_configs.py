from configs import FrozenLakeConfig, AgentConfig, TrainingConfig

from reinforcement_learning.low_level_implementations.algorithms.agents import QLearningAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import SarsaAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import ExpectedSarsaAgent
from reinforcement_learning.low_level_implementations.algorithms.agents import DoubleQLearningAgent

from reinforcement_learning.low_level_implementations.algorithms.action_selection import EpsilonGreedySelector
from reinforcement_learning.low_level_implementations.algorithms.action_selection import SoftmaxSelector

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
    FrozenLakeConfig(
        num_trials=2,
        num_episodes=100,
        max_steps_per_episode=100,
        render_mode="none",
        num_checkpoints=5,
        lake_size=4,
        is_slippery=False
    ),
]

# Define agent configs
AGENT_CONFIGS = [
    AgentConfig(
        agent_type=QLearningAgent,
        learning_rate=0.1,
        discount_factor=0.9,
        action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme="linear")
    ),
    AgentConfig(
        agent_type=SarsaAgent,
        learning_rate=0.1,
        discount_factor=0.9,
        action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme="linear")
    ),
    AgentConfig(
        agent_type=DoubleQLearningAgent,
        learning_rate=0.1,
        discount_factor=0.9,
        action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme="linear")
    ),
    AgentConfig(
        agent_type=ExpectedSarsaAgent,
        learning_rate=0.1,
        discount_factor=0.9,
        action_selector=EpsilonGreedySelector(epsilon=0.1, decay_scheme="linear")
    ),
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

