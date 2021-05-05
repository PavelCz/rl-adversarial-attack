from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class MCDropout(DQNPolicy):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            dropout_rate=0.5
    ):
        super(MCDropout, self).__init__(observation_space, action_space, lr_schedule)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # self.latent_dim_pi = last_layer_dim_pi

        # Policy network
        self.q_net.q_net = self.make_q_net2(action_space, dropout_rate, observation_space)
        self.q_net_target.q_net = self.make_q_net2(action_space, dropout_rate, observation_space)

    def make_q_net2(self, action_space, dropout_rate, observation_space):
        q_net = nn.Sequential()
        feature_dim = observation_space.shape[0]
        num_nodes = 64
        num_layers = 2
        num_actions = action_space.n
        for i in range(num_layers):
            q_net.add_module(f"dropout", nn.Dropout(p=dropout_rate))
            in_dim = feature_dim if i == 0 else num_nodes
            out_dim = num_actions if i == (num_layers - 1) else num_nodes
            q_net.add_module(f"linear{i}", nn.Linear(in_dim, out_dim))
            q_net.add_module("relu", nn.ReLU())
        return q_net

    def forward(self, obs: th.Tensor, deterministic: bool = True):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.q_net(obs)
