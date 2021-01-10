import gym
import numpy as np


def flip_observation_horizontally(obs):
    # Ball directions should be ['NW', 'W', 'SW', 'SE', 'E', 'NE']
    # COnsequently, when we reverse ball direction array, we flip horizontally
    agent_pos, ball_pos, ball_dir = obs[:2], obs[2:4], obs[4:]

    # Perform the flipping
    agent_pos[1] = 1 - agent_pos[1]
    ball_pos[1] = 1 - ball_pos[1]
    ball_dir.reverse()

    # Merge into single obs list
    reversed_obs = agent_pos + ball_pos + ball_dir
    return reversed_obs


class OpponentWrapper(gym.Wrapper):
    """
    This wrapper allows multi-agent environment to be used with agents that expect normal gym-environments. It allows to
    set the opponent agent to a separate agent. This opponent will not learn, instead it will simply be evaluated to
    determine the actions of the opponent.
    Currently this wrapper only supports 2-agent environments.

    :param env: A multi-agent gym environment that will be wrapped in order to support setting an opponent. This can be
        useful for e.g. self-play
    :param opponent: This agent will be used as the opponent in the multi-agent environment
    """

    def __init__(self, env: gym.Env, opponent=None, num_skip_steps=0):
        super(OpponentWrapper, self).__init__(env)
        self.opponent = opponent
        self.opponent_obs = None  # The previous observation that is meant for the opponent
        # Overwrite the variable to make sense for single-agent env
        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]
        self.opponent_right_side = True
        self.num_skip_steps = num_skip_steps

    def reset(self):
        """
        Reset the environment
        """
        # If this destructuring assignment
        multi_obs = self.env.reset()
        if len(multi_obs) != 2:
            raise AttributeError(
                f"Wrong number of observations, should be 2, is {len(multi_obs)}. Currently, only environments for "
                f"exactly 2 agents are supported")
        if self.opponent_right_side:
            [main_obs, opponent_obs] = self.env.reset()
        else:
            [opponent_obs, main_obs] = self.env.reset()
        # Save opponent observation for later
        self.opponent_obs = opponent_obs
        # Return the observation that is meant for the main agent
        return np.array(main_obs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Flip the observation for the opponent, such that from the opponents viewpoint it is also on the left side
        obs_for_opponent = flip_observation_horizontally(self.opponent_obs)
        # Get the action taken by the opponent, based on the observation that is meant for the opponent
        opponent_action, _states = self.opponent.predict(np.array(obs_for_opponent), deterministic=True)

        # action = list(action)
        # opponent_action = list(opponent_action)

        # Concatenate opponent + main agent actions
        if self.opponent_right_side:
            actions = [action, opponent_action]
            # Here we destructure the 4 lists which have 2 elements each into their components
            # If this destructuring fails this most likely means these 4 lists don't have 2 pieces, i.e. the multi-agent
            # environment is not a 2-agent environment
            for _ in range(self.num_skip_steps + 1):
                [main_obs, opponent_obs], [main_reward, _], [main_done, _], info = self.env.step(actions)
        else:  # Same as other case above just with left and right side switched
            actions = [opponent_action, action]
            for _ in range(self.num_skip_steps + 1):
                [opponent_obs, main_obs], [_, main_reward], [_, main_done], info = self.env.step(actions)

        # Update the observation that is needed by the opponent in the next step
        self.opponent_obs = opponent_obs
        # Return only the vars that are meant for the main agent
        return np.array(main_obs), main_reward, main_done, info

    def set_opponent(self, opponent):
        self.opponent = opponent

    def set_opponent_right_side(self, opponent_right_side):
        self.opponent_right_side = opponent_right_side
