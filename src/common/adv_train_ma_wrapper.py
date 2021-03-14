import numpy as np
import gym
import random
import torch
from gym import spaces

from src.attacks.fgsm import fgsm_attack

class AdvTrainWrapperNFSP:
    @staticmethod
    def adv_rl_ob(observation, model, args):
        if random.random() < 0.5:
            observation = fgsm_attack(torch.tensor(observation).to(args.device), model, 0.02, 'rl', args)
        return observation
    @staticmethod
    def adv_sl_ob(observation, model, args):
        if random.random() < 0.5:
            observation = fgsm_attack(torch.tensor(observation).to(args.device), model, 0.02, 'sl', args)
        return observation