import numpy as np
import gym
from gym import spaces

class ObserveOpponent(gym.ObservationWrapper):
    """
    Return observation in [agent pos(2), ball pos(2), balldir(6-onehot), opponent pos(2)]
    agent : Choose "both", "p1" or "p2" to decide which agent observes the whole image space
    """
    def __init__(self, env, agent):
        super(ObserveOpponent, self).__init__(env)
        self.agent = agent
        if self.agent == 'both':
            self.observation_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(12,)),
                                  gym.spaces.Box(low=0.0, high=1.0, shape=(12,))]
        elif self.agent == 'p1':
            self.observation_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(12,)), self.observation_space[1]]
        elif self.agent == 'p2': 
            self.observation_space = [self.observation_space[0], gym.spaces.Box(low=0.0, high=1.0,  shape=(12,))]
        else:                        
            raise AssertionError ("Argument takes value \"both\", \"p1\" or \"p2\" but received {}".format(self.agent))

    def observation(self, observation):
        if self.agent == 'both':
            return (np.float32(np.concatenate((observation[0], observation[1][:2]))), 
                    np.float32(np.concatenate((observation[1], observation[0][:2]))))
        elif self.agent == 'p1':
            return (np.float32(np.concatenate((observation[0], observation[1][:2]))), observation[1])
        elif self.agent == 'p2': 
            return (observation[0], np.float32(np.concatenate((observation[1], observation[0][:2]))))
        else:                        
            raise AssertionError ("Argument takes value \"both\", \"p1\" or \"p2\" but received {}".format(self.agent))