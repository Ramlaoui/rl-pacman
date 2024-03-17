class BaseAgent:
    def __init__(self, actions):
        self.actions = actions

    def act(self, observation):
        raise NotImplementedError
