import numpy as np
from tqdm import tqdm
from .game import Game
from .levels import base_level


class Trainer:
    def __init__(self, env, agent, level=None, gui=True):
        self.agent = agent
        if level is not None:
            self.base_level = level
            self.env = Game(**level, gui=gui, ai=True)
        else:
            self.env = env
            self.env.gui = gui
        self.gui = gui

    def train(self, episodes=1000, render=False, log_interval=100, verbose=True):
        self.rewards = []
        if verbose:
            print("Training...")
        for episode in tqdm(range(episodes), total=episodes):
            step = 0
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done and step < 10000:
                step += 1
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            self.rewards.append(total_reward)
            self.env.gui = self.gui
            if verbose and episode % log_interval == 0:
                if render:
                    self.env.gui = True
                print(
                    f"Episode {episode}/{episodes}, Average Reward: {np.mean(self.rewards[-log_interval:])}"
                )
        return self.agent

    def test(self, episodes=100, render=False, verbose=True):
        if verbose:
            print("Testing...")
        self.rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if render:
                    self.env.render()
                total_reward += reward
            self.rewards.append(self.env.total_reward)
            if verbose:
                print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}")
        return self.rewards
