import numpy as np
from .base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, actions):
        super().__init__(actions)

    def act(self, observation):
        return np.random.choice(self.actions)
