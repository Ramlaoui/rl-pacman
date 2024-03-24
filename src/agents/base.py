import time
import numpy as np
import matplotlib.pyplot as plt


class BaseAgent:
    def __init__(self, env):
        self.env = env
        self.scores = []
        self.smoothed_scores = []
        self.action_space = env.get_action_space()

    def choose_action(self, state):
        raise NotImplementedError

    def update_Q(self, episode):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, next_action, done):
        raise NotImplementedError

    def update_policy(self):
        raise NotImplementedError

    def plot_score(self, title=""):
        # Rolling average of the scores
        self.smoothed_scores = np.convolve(
            self.scores, np.ones(50) / 50, mode="valid"
        )
        plt.plot(self.scores, label="Score")
        plt.plot(
            self.smoothed_scores,
            label="Average score",
        )
        plt.title(f"Scores over episodes\n{title}")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    def play(self, render=True, slow=False):
        state = self.env.reset()
        done = False
        i = 0
        reward = 0
        while not done:
            i += 1
            action = self.choose_action(state)
            state, reward_, done, _, info = self.env.step(action)
            reward += reward_
            if render:
                self.env.render()  # add option for slow
            if i > 5000:
                print(
                    f"Stopped playing to continue running the code. Score: {info['score']}"
                )
                break
        if render:
            print("Game over!")

        return reward

    def test(self, episodes=100, render=False):
        scores = []
        for _ in range(episodes):
            score = self.play(render=render, slow=False)
            scores.append(score)
        return np.convolve(scores, np.ones(100) / 100, mode="valid")
