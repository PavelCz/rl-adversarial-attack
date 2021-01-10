import gym


class SimpleRuleBasedAgent:
    def __init__(self, env: gym.Env):
        pass

    def predict(self, obs, deterministic=True):
        vertical_pos_agent = obs[0]
        vertical_pos_ball = obs[2]

        if vertical_pos_ball > vertical_pos_agent:  # Ball is below agent
            action = 2  # Down
        elif vertical_pos_ball < vertical_pos_agent:  # Ball is above agent
            action = 1  # Up
        else:
            action = 0  # No operation / stay at position
        return action, None
