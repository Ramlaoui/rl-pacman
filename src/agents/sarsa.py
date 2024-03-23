import os
import sys
import time
from tqdm import tqdm
import numpy as np
from .base import BaseAgent


class SarsaLambdaAgent(BaseAgent):
    def __init__(self, env, lambda_=0.9, gamma=0.99, alpha=0.1, epsilon=0.1):
        super().__init__(env)
        self.env = env
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}
        self.E = {}
        self.agent_type = "sarsa"

    def choose_action(self, state):
        state = tuple(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            self.Q.setdefault(state, {})
            return max(
                self.Q[state],
                key=self.Q[state].get,
                default=np.random.choice(self.action_space),
            )

    def update_sarsa(self, state, action, reward, next_state, next_action, done):
        # Ensure state-action pairs are in the Q and E tables
        state = tuple(state)
        next_state = tuple(next_state)
        self.Q.setdefault(state, {}).setdefault(action, 0)
        self.Q.setdefault(next_state, {}).setdefault(next_action, 0)
        self.E.setdefault(state, {}).setdefault(action, 0)

        # Compute the TD error
        delta = (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )

        # Increment the eligibility trace for the current state-action
        self.E[state][action] += 1

        # Now, update Q and E only for visited state-action pairs
        for s, actions in self.E.items():
            for a in actions:
                s = tuple(s)
                self.Q[s][a] += (
                    self.alpha * delta * self.E[s][a]
                )  # Update Q-value using eligibility trace
                self.E[s][a] *= self.gamma * self.lambda_  # Decay eligibility trace
