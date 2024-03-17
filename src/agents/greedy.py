import numpy as np
from .base import BaseAgent
import random


class EpsilonGreedyAgentConstantStepSize(BaseAgent):
    def __init__(self, actions, epsilon=0.1, step_size=0.1):
        super().__init__(actions)
        self.epsilon = epsilon
        self.step_size = step_size
        self.q_values = np.zeros(len(actions))
        self.arm_count = np.zeros(len(actions))
        self.last_action = None

    def act(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_values)

    def update(self, state, action, reward, next_state):
        self.arm_count[action] += 1
        self.q_values[action] += self.step_size * (reward - self.q_values[action])
