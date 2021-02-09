import gym
import torch
import numpy as np

class StableBaselinesWrapper(gym.Wrapper):
    """
    Wrap ma-gym environment to environment that suitable for stable-baseline. 
    eval_agent : Choose "p1" or "p2" to decide which agent is evaluated. The evaluated agent which will
    not be trained can be considered as a built-in agent in a single agent environment. 
    """
    def __init__(self, env, eval_agent, agent_model):
        self.env = env
        self.eval_agent = eval_agent
        self.agent_model = agent_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.eval_agent == 'p1':
            self.observation_space = self.env.observation_space[1]
            self.action_space = self.env.action_space[1]
        else:
            self.observation_space = self.env.observation_space[0]
            self.action_space = self.env.action_space[0]

    def reset(self):
        p1_state, p2_state = self.env.reset()
        if self.eval_agent == 'p1':
            self.eval_state = p1_state
            return p2_state
        else:
            self.eval_state = p2_state
            return p1_state

    def step(self, action):
        eval_action = self.agent_model.act(torch.tensor(self.eval_state).to(self.device)) 
        if self.eval_agent == 'p1':
            actions = [eval_action, action]
            [_, state], [_, reward], [_, done], info = self.env.step(actions)
        else:
            actions = [action, eval_action]
            [state,_], [reward,_], [done,_], info = self.env.step(actions)
        return state, reward, done, info