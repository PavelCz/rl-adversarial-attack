import gym


class RandomAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def predict(self, *_):
        return self.env.action_space.sample(), None
