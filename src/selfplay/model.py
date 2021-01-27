import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math

from functools import partial

def DQN(env, args, agent):
    if args.obs_img == 'both' or args.obs_img == agent:
        model = DQNConv(env)
    else:
        model = DQNFC(env)
    return model

def Policy(env, args, agent):
    if args.obs_img == 'both' or args.obs_img == agent:
        model = PolicyConv(env)
    else:
        model = PolicyFC(env)
    return model

class DQNFC(nn.Module):
    def __init__(self, env):
        super(DQNFC, self).__init__()
        
        index = np.argmin([len(env.observation_space[0].shape), len(env.observation_space[1].shape)])
        self.input_shape = env.observation_space[index].shape
        self.num_actions = env.action_space[0].n
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:
            # e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.forward(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

class PolicyFC(DQNFC):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """
    def __init__(self, env):
        super(PolicyFC, self).__init__(env)
        self.fc = nn.Sequential(
            nn.Linear(self.input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return action


class DQNConv(nn.Module):
    """
    Basic DQN

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env):
        super(DQNConv, self).__init__()
        
        index = np.argmax([len(env.observation_space[0].shape), len(env.observation_space[1].shape)])
        self.input_shape = env.observation_space[index].shape
        self.num_actions = env.action_space[0].n

        self.flatten = Flatten()
        
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 8, kernel_size=3, stride=2),
            cReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1),
            cReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1),
            cReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.forward(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

class PolicyConv(DQNConv):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """
    def __init__(self, env):
        super(PolicyConv, self).__init__(env)
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return action


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

