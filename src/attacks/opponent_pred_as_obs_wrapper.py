import gym
import torch


class OpponentPredictionObs(gym.ObservationWrapper):
    """
    """
    def __init__(self, env):
        super(OpponentPredictionObs, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(15,))

    def observation(self, observation):
        # Get the current observation meant for the opponent which is the victim
        op_obs = self.env.opponent_obs
        victim_q_net = self.env.opponent.q_net

        # Preprocess obs for q_net
        victim_obs = torch.tensor([op_obs])
        # Move obs to correct device
        victim_obs = victim_obs.to(victim_q_net.device)

        # Get the logits from the victim for this observation
        logits = victim_q_net(victim_obs)
        logits = logits.squeeze()
        pred = torch.softmax(logits, dim=0)

        # Concatenate these predictions to the original observation
        return torch.cat([torch.tensor(observation), pred.detach()])

