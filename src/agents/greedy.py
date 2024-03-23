import numpy as np
from .base import BaseAgent
import random


class EpsilonGreedyAgentConstantStepSize(BaseAgent):
    def __init__(self, env, epsilon=0.1, step_size=0.1):
        super().__init__(env)
        self.epsilon = epsilon
        self.step_size = step_size
        self.q_values = np.zeros(len(self.action_space))
        self.arm_count = np.zeros(len(self.action_space))
        self.last_action = None
        self.agent_type = "epsilon_greedy_constant_step_size"

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.q_values)

    def update(self, state, action, reward, next_state, done):
        self.arm_count[action] += 1
        self.q_values[action] += self.step_size * (reward - self.q_values[action])
