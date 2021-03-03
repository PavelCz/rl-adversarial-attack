import gym
import torch
import numpy as np
import time

from src.common.environment_dynamics import approximate_step_pong_duel
from src.common.observation_utils import flip_observation_horizontally, is_ball_moving_towards_player


class WhiteBoxAdversarialAgent:
    def __init__(self, env: gym.Env, victim, victim_type):
        self.victim = victim
        self.env = env
        self.victim_type = victim_type

    def predict(self, obs, deterministic=True):
        if is_ball_moving_towards_player(obs, 'p1'):
            # Try to perform good actions by copying victim
            current_best_action = self._predict_copy_victim(obs)
        else:
            # Ball is not moving twoards attacker -> confuse victim by moving adversarially
            current_best_action = self._predict_adversarially(obs)
        return current_best_action, None

    def _predict_adversarially(self, obs):
        # Choose best action by determining which will cause a worse state for the victim
        # Get observation as it should be passed to the victim q_net
        victim_obs = self._switch_obs(obs.copy())

        # Get the logits when performing forward pass with given observations
        logits = self._victim_q_net_forward(victim_obs)

        victim_action = torch.argmax(logits).item()
        lowest_max = 1
        current_best_action = 0
        for action in range(self.env.action_space.n):
            # Approximate the next state when victim follows its policy and attacker performs selected action
            approx_state = approximate_step_pong_duel(self.env, obs.copy(), action_p1=action, action_p2=victim_action)

            # Convert the state to use by victim q_net
            victim_approx_state = self._switch_obs(approx_state)

            # Get the action the victim will perform in the current state
            # Get the logits victim q_net in predicted state
            logits = self._victim_q_net_forward(victim_approx_state)

            # Turn the logits into probability distribution
            pred = torch.softmax(logits, dim=0)
            max = torch.max(pred).item()
            # We are looking for the action that causes the highest entropy
            # Because pred is a probability distribution that sums up to 1, the action that causes the lowest max value should have the
            # highest entropy
            if max < lowest_max:
                current_best_action = action
                lowest_max = max
        # Perform the action with the lowest entropy in the real world
        return current_best_action

    def _predict_copy_victim(self, obs):
        # Check the q_net, what would the victim do in this state
        logits = self._victim_q_net_forward(obs)
        action = torch.argmax(logits).item()
        return action

    def _victim_q_net_forward(self, victim_obs: np.ndarray) -> torch.Tensor:
        """
        Access the reference to the victim q-net and get the ouput of a single forward pass with the given observations.
        :param victim_obs: The observation as an numpy array
        :return: The logits after forward pass as a torch tensor
        """
        with torch.no_grad():
            victim_q_net = self._get_victim_q_net()
            # Get the action the victim will perform in the current state
            victim_obs = torch.tensor([victim_obs])
            # Move obs to correct device
            victim_obs = victim_obs.to(victim_q_net.device)
            logits = victim_q_net(victim_obs)
            logits = logits.squeeze()
        return logits

    def _get_victim_q_net(self):
        if self.victim_type == 'sb3':
            # Get the victim q_net from a stable-baselines3 model
            victim_q_net = self.victim.q_net
        elif self.victim_type == 'nfsp':
            raise NotImplementedError('White-box adversarial policy not implemented for NFSP')
        else:
            raise NotImplementedError(f'White-box adversarial policy not implemented for the type {self.victim_type}')
        return victim_q_net

    def _switch_obs(self, obs):
        """
        Flip the given obs in such a way, that if the received obs is for player 1, the output obs will be for player 2 and vice versa
        """
        switched_obs = np.concatenate([
            obs[-2:],
            obs[2:10],
            obs[:2]
        ])

        reversed_obs = flip_observation_horizontally(switched_obs, image_observations=False)
        return reversed_obs
