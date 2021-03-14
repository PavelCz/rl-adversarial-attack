import numpy as np
import gym
import random
from gym import spaces

from src.attacks.fgsm import fgsm_attack_sb3


class AdversarialTrainingWrapper(gym.ObservationWrapper):
    """
    A wrapper that will adversarially perturb a subset of observations for robustness training
    """
    def __init__(self, env: gym.Env, adversarial_probability: float, img_obs: bool):
        super(AdversarialTrainingWrapper, self).__init__(env)
        self.adversarial_probability = adversarial_probability
        self.img_obs = img_obs
        # The vcitim model must be set externally before training starts
        self.victim_model = None

    def observation(self, observation):
        if random.random() < self.adversarial_probability:  # Perform adversarial perturbation with given probability
            observation = fgsm_attack_sb3(observation, self.victim_model, 0.02, img_obs=self.img_obs)
        return observation
