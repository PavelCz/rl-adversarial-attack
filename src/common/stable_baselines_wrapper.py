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
        super(StableBaselinesWrapper, self).__init__(env)
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
        if self.eval_agent == 'p1':
            self.eval_state, self.train_state = self.env.reset()
        else:
            self.train_state, self.eval_state = self.env.reset()
        return np.array(self.train_state)

    def step(self, action):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action = action[0]
        eval_action = self.agent_model.act(torch.tensor(self.eval_state).to(self.device)) 
        if self.eval_agent == 'p1':
            actions = [eval_action, action]
            [eval_state, state], [_, reward], [_, done], info = self.env.step(actions)
        else:
            actions = [action, eval_action]
            [state, eval_state], [reward,_], [done,_], info = self.env.step(actions)

        # Use for training adversarial policy 
        # reward = self.bounce_reward(self.train_state, state, reward)

        self.eval_state = eval_state
        self.train_state = state 
        return np.array(state), reward, done, info

    def bounce_reward(self, state, new_state, reward):
        """
        Bounce reward for training adversarial policy. Adversary gets a minus reward
        when the victim hits the ball. This reward is formulated in a way that hoping
        the average frames per epsiode will decrease
        """
        ball_dir = np.nonzero(state[4:10])[0]
        new_ball_dir = np.nonzero(new_state[4:10])[0]
        if self.eval_agent == 'p1' and ball_dir in [0,1,2] and new_ball_dir in [3,4,5] and reward != 1:
            return -1
        elif self.eval_agent =='p2' and ball_dir in [3,4,5] and new_ball_dir in [0,1,2] and reward != 1:
            return -1
        else:
            return 0
    