import gym
import torch
import numpy as np

from src.common.environment_dynamics import approximate_step_pong_duel
from src.common.observation_utils import flip_observation_horizontally


class WhiteBoxAdversarialAgent:
    def __init__(self, env: gym.Env, victim, victim_type):
        self.victim = victim
        self.env = env
        self.victim_type = victim_type

    def predict(self, obs, deterministic=True):
        # Choose best action by determining which will cause a worse state for the victim

        # Get observation as it should be passed to the victim q_net
        victim_obs = self._switch_obs(obs.copy())

        # Get the victim q_net
        if self.victim_type == 'sb3':
            # Get the victim q_net from a stable-baselines3 model
            victim_q_net = self.victim.q_net
        elif self.victim_type == 'nfsp':
            raise NotImplementedError('White-box adversarial policy not implemented for NFSP')
        else:
            raise NotImplementedError(f'White-box adversarial policy not implemented for the type {self.victim_type}')

        # Get the action the victim will perform in the current state
        victim_obs = torch.tensor([victim_obs])
        # Move obs to correct device
        victim_obs = victim_obs.to(victim_q_net.device)
        logits = victim_q_net(victim_obs)
        victim_action = torch.argmax(logits).item()

        lowest_max = 1
        current_best_action = 0
        for action in range(self.env.action_space.n):
            # Apporximate the next state when victim follows its policy and attacker performs selected action
            approx_state = approximate_step_pong_duel(self.env, obs.copy(), action_p1=action, action_p2=victim_action)

            # Convert the state to use by victim q_net
            victim_approx_state = self._switch_obs(approx_state)
            # Get the action the victim will perform in the current state
            victim_approx_state = torch.tensor([victim_approx_state])
            # Move obs to correct device
            victim_approx_state = victim_approx_state.to(victim_q_net.device)

            # Get the logits victim q_net in predicted state
            logits = victim_q_net(victim_approx_state)
            logits = logits.squeeze()
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
        return current_best_action, None

    def predict_slow(self, obs, deterministic=True):
        # Choose best action by determining which will cause a worse state for the victim
        # obs not necessary here, as the state is implicitly in the current state of self.env

        # Make copy of environment for internal lookahead
        sim_env = copy.deepcopy(self.env)

        victim_q_net = self.victim.q_net

        lowest_max = 1
        current_best_action = 0
        for action in range(self.env.action_space.n):
            next_obs, _, _, _ = sim_env.step(action)

            # This is the new observation for us, this gets us the next observation for our victim
            # sim_env.w

            next_obs = torch.tensor([next_obs])
            # Move obs to correct device
            next_obs = next_obs.to(victim_q_net.device)

            # Get the logits victim q_net in predicted state
            logits = victim_q_net(next_obs)
            logits = logits.squeeze()
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
        return current_best_action, None

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