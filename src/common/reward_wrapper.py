import gym


class RewardZeroToNegativeBiAgentWrapper(gym.RewardWrapper):
    def reward(self, reward):
        if reward[0] == 1 and reward[1] == 0:
            return [1, -1]
        elif reward[0] == 0 and reward[1] == 1:
            return [-1, 1]
        else:
            return reward
