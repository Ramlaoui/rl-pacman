import numpy as np
from .base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        self.actions = env.get_action_space()

    def choose_action(self, state):
        return np.random.choice(self.actions)
