import gym
import numpy as np

from src.common.observation_utils import flip_observation_horizontally


class MAGymCompatibilityWrapper(gym.Wrapper):
    """
    This wrapper allows multi-agent environment to be used with agents that expect normal gym-environments. It allows to
    set the opponent agent to a separate agent. This opponent will not learn, instead it will simply be evaluated to
    determine the actions of the opponent.
    Currently this wrapper only supports 2-agent environments.

    :param env: A multi-agent gym environment that will be wrapped in order to support setting an opponent. This can be
        useful for e.g. self-play
    :param opponent: This agent will be used as the opponent in the multi-agent environment
    """

    def __init__(self, env: gym.Env, image_observations, opponent=None, num_skip_steps=0, ):
        super(MAGymCompatibilityWrapper, self).__init__(env)
        self.opponent = opponent
        self.opponent_obs = None  # The previous observation that is meant for the opponent
        # Overwrite the variable to make sense for single-agent env
        self.observation_space = env.observation_space[0]  # Box(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))
        self.action_space = self.action_space[0]
        self.opponent_right_side = True
        self.num_skip_steps = num_skip_steps
        self.image_observations = image_observations

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
        if self.image_observations == 'both':
            img_obs_op = True
        elif self.image_observations == 'main':
            img_obs_op = False
        elif self.image_observations == 'none':  # This means feature observations
            img_obs_op = False
        else:
            raise AttributeError("self.image_observation must be either 'main' or 'both'")

        # Flip the observation for the opponent, such that from the opponents viewpoint it is also on the left side
        obs_for_opponent = flip_observation_horizontally(self.opponent_obs, img_obs_op)
        # Get the action taken by the opponent, based on the observation that is meant for the opponent
        opponent_action, _states = self.opponent.predict(np.array(obs_for_opponent), deterministic=False)

        # action = list(action)
        if isinstance(opponent_action, list) or (isinstance(opponent_action, np.ndarray) and len(opponent_action.shape) > 0):
            opponent_action = opponent_action[0]
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action = action[0]

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
        # if(main_done is True):
        #    print("=\n=\n=\n")
        # print(f"\r{(main_obs[:4], main_reward, main_done, info, action)}")
        # Return only the vars that are meant for the main agent
        return np.array(main_obs), main_reward, main_done, info

    def set_opponent(self, opponent):
        self.opponent = opponent

    def set_opponent_right_side(self, opponent_right_side):
        self.opponent_right_side = opponent_right_side
