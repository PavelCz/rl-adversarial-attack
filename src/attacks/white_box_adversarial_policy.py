import copy

import gym
import torch


class WhiteBoxAdversarialAgent:
    def __init__(self, env: gym.Env, victim):
        self.victim = victim
        self.env = env

    def predict(self, obs, deterministic=True):
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
            #sim_env.w

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
