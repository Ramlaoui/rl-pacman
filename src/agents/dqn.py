import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from .base import BaseAgent


class DQNAgent(BaseAgent):
    def __init__(
        self,
        env,
        alpha=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=1,
        memory_size=10000,
        batch_size=20,
    ):
        super().__init__(env)
        self.state_size = self.env.get_state_size()
        self.action_size = len(self.action_space)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.agent_type = "dqn"

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
        )
        model.apply(self._init_weights)
        return model

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_values = self.model(state).detach().numpy()
        return np.argmax(q_values[0])

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.replay()

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(
                    self.model(next_state).detach()
                )

            current_q = self.model(state)[action]
            loss = self.criterion(current_q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.01:  # Ensuring epsilon does not become too small
            self.epsilon *= self.epsilon_decay
