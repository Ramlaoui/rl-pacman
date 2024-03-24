import os
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .game import PacManEnv
from .levels import base_level


class Trainer:
    def __init__(self, env, agent, level=None, gui=True, run_name=""):
        self.agent = agent
        self.run_name = run_name
        if self.run_name != "":
            self.log_path = Path("logs")
            self.run_path = self.log_path / f"{agent.agent_type}-{self.run_name}"
            shutil.rmtree(self.run_path, ignore_errors=True)
            os.makedirs(self.run_path, exist_ok=True)
        if level is not None:
            self.base_level = level
            self.env = PacManEnv(**level, gui=gui, ai=True)
        else:
            self.env = env
            self.env.gui = gui
        self.gui = gui

    def plot_score(self, title=""):
        self.agent.plot_score(title)

    def train(self, episodes=1000, render=False, log_interval=100, verbose=True):
        if verbose:
            print("Training...")
            pbar = tqdm(range(episodes))
        for episode in range(episodes):
            step = 0
            state = self.env.reset()
            if self.agent.agent_type == "monte_carlo":
                episode_data = []
            done = False
            total_reward = 0
            while not done and step < 10000:
                step += 1
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if self.agent.agent_type in [
                    "q_learning",
                    "epsilon_greedy_constant_step_size",
                    "dqn",
                ]:
                    self.agent.update(state, action, reward, next_state, done)
                elif self.agent.agent_type == "sarsa":
                    next_action = self.agent.choose_action(next_state)
                    self.agent.update_sarsa(
                        state, action, reward, next_state, next_action, done
                    )
                elif self.agent.agent_type == "monte_carlo":
                    episode_data.append([state, action, reward])
                state = next_state
                total_reward += reward
            self.agent.scores.append(total_reward)
            self.env.gui = self.gui
            if verbose and episode % log_interval == 0:
                if render:
                    self.env.gui = True
                    if self.run_name != "":
                        self.env.log_path = self.run_path / f"episode-{episode}.png"
            if self.agent.agent_type == "monte_carlo":
                self.agent.update_Q(episode_data)
                self.agent.update_policy()
            if verbose:
                pbar.set_description(
                    f"Episode {episode} - {log_interval}-Average Reward: {np.mean(self.agent.scores[-log_interval:])}"
                )
                pbar.update(1)
        pbar.close()
        self.agent.smoothed_scores = np.convolve(
            self.agent.scores, np.ones(50) / 50, mode="valid"
        )
        return self.agent

    def play(self, render=True, slow=False):
        return self.agent.play(render=render, slow=slow)

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
