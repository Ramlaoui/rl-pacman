import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .base import BaseAgent


class MonteCarloAgent(BaseAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        super().__init__(env)
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        self.returns = {}
        self.policy = {}
        self.scores = []
        self.agent_type = "monte_carlo"

    def choose_action(self, state):
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = [
                self.Q.get((state, a), 0) for a in range(len(self.action_space))
            ]
            max_q_value = max(q_values)
            # In case there are multiple actions with the same max Q-value, we select randomly among them
            actions_with_max_q = [a for a, q in enumerate(q_values) if q == max_q_value]
            return np.random.choice(actions_with_max_q)

    def update_Q(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            state = tuple(state)
            G = self.gamma * G + reward
            if (state, action) not in self.returns:
                self.returns[(state, action)] = []
            self.returns[(state, action)].append(G)
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])

    def update_policy(self):
        for state, _ in self.Q:
            state = tuple(state)
            q_values = [
                self.Q.get((state, a), 0) for a in range(len(self.action_space))
            ]
            best_action = np.argmax(q_values)
            self.policy[state] = best_action
