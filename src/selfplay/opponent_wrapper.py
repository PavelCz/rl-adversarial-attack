import gym


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

    def __init__(self, env: gym.Env, opponent=None):
        super(OpponentWrapper, self).__init__(env)
        self.opponent = opponent
        self.opponent_obs = None  # The previous observation that is meant for the opponent
        # Overwrite the variable to make sense for single-agent env
        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]

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
        [main_obs, opponent_obs] = self.env.reset()
        # Save opponent observation for later
        self.opponent_obs = opponent_obs
        # Return the observation that is meant for the main agent
        return main_obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Get the action taken by the opponent, based on the observation that is meant for the opponent
        opponent_action, _states = self.opponent.predict(self.opponent_obs)
        # Concatenate opponent + main agent actions
        actions = [action, opponent_action]
        # Here we destructure the 4 lists which have 2 elements each into their components
        # If this destructuring fails this most likely means these 4 lists don't have 2 pieces, i.e. the multi-agent
        # environment is not a 2-agent environment
        [main_obs, opponent_obs], [main_reward, _], [main_done, _], info = self.env.step(actions)
        # Update the observation that is needed by the opponent in the next step
        self.opponent_obs = opponent_obs
        # Return only the vars that are meant for the main agent
        return main_obs, main_reward, main_done, info

    def set_opponent(self, opponent):
        self.opponent = opponent