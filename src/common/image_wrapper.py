import numpy as np
import gym
from gym import spaces

class ObservationVectorToImage(gym.ObservationWrapper):
    """
    Return observation in num_channels x weight x height (Tensor format)
    """
    def __init__(self, env):
        super(ObservationVectorToImage, self).__init__(env)
        self.env = env
        self.observation_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(3, 200, 150), dtype=np.uint8),
                                  gym.spaces.Box(low=0.0, high=1.0, shape=(3, 200, 150), dtype=np.uint8)]

    def observation(self, observation):
        observation_img = self.env.render(mode ='rgb_array')[2:202, 2:152, :]/255.
        return (np.swapaxes(observation_img, 2, 0), np.swapaxes(observation_img, 2, 0))