import numpy as np
from .base import BaseAgent
import random


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
    ):
        super().__init__(env)
        self.q_table = {}
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.exploration_rate = epsilon
        self.agent_type = "q_learning"

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(self.action_space)
        return max(
            self.q_table.get(state, {}),
            key=self.q_table.get(state, {}).get,
            default=random.choice(self.action_space),
        )

    def update(self, state, action, reward, next_state, done):
        old_value = self.q_table.get(state, {}).get(action, 0)
        next_max = max(self.q_table.get(next_state, {}).values(), default=0)
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )
        self.q_table.setdefault(state, {})[action] = new_value
