import numpy as np
import gym
from gym import spaces
from PIL import Image
from skimage.transform import resize

class ObservationVectorToImage(gym.ObservationWrapper):
    """
    Return observation in channels x weight x height with values ranging from 0 to 1(Tensor format)
    agent : Choose "both", "p1" or "p2" to decide which agent observes the whole image space
    """
    def __init__(self, env, agent):
        super(ObservationVectorToImage, self).__init__(env)
        self.env = env
        self.agent = agent
        if self.agent == 'both':
            self.observation_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(3, 40, 30), dtype=np.uint8),
                                  gym.spaces.Box(low=0.0, high=1.0, shape=(3, 40, 30), dtype=np.uint8)]
        elif self.agent == 'p1':
            self.observation_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(3, 40, 30), dtype=np.uint8),
                                  self.observation_space[1]]
        elif self.agent == 'p2': 
            self.observation_space = [self.observation_space[0],
                                    gym.spaces.Box(low=0.0, high=1.0, shape=(3, 40, 30), dtype=np.uint8)]
        else:                        
            raise AssertionError ("Argument takes value \"both\", \"p1\" or \"p2\" but received {}".format(self.agent))

    def observation(self, observation):
        observation_img = self.env.render(mode ='rgb_array')[2:202, 2:152, :]
        # skimage.transform.resize will return image as float with values ranging from 0 to 1
        observation_img = np.float32(resize(observation_img,(40,30), anti_aliasing=True))
        # img = Image.fromarray((observation_img * 255).astype(np.uint8))
        # img.save("observation_img.png")

        if self.agent == 'both':
            return (np.swapaxes(observation_img, 2, 0), np.swapaxes(observation_img, 2, 0))
        elif self.agent == 'p1':
            return (np.swapaxes(observation_img, 2, 0), observation[1])
        elif self.agent == 'p2': 
            return (observation[0], np.swapaxes(observation_img, 2, 0))
        else:                        
            raise AssertionError ("Argument takes value \"both\", \"p1\" or \"p2\" but received {}".format(self.agent))