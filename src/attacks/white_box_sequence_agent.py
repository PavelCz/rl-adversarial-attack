import gym
import numpy as np
import copy


class WhiteBoxSequenceAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.planned_actions = []

    def predict(self, obs, deterministic=True):
        # obs not necessary here, as the state is implicitly in the current state of self.env
        if len(self.planned_actions) > 0:
            action = self.planned_actions.pop(0)
            return action, None
        else:
            max_steps = 1000
            rewards = []
            for action in range(self.env.action_space.n):
                new_planned_actions = []
                # Make copy of environment for internal lookahead
                sim_env = copy.deepcopy(self.env)

                # Perform step in simulated env
                _, reward, _, _ = sim_env.step(action)

                if reward == 1:
                    return action, None
                elif reward == -1:
                    rewards.append(reward)
                else:
                    new_planned_actions.append(action)
                    for i in range(max_steps):
                        next_action = self.env.action_space.sample()
                        new_planned_actions.append(next_action)
                        # Perform step in simulated env
                        _, reward, _, _ = sim_env.step(next_action)
                        # Encountered state with reward -> stop simulation
                        if reward != 0:
                            break
                    if reward == 1:
                        action = new_planned_actions.pop(0)
                        self.planned_actions = new_planned_actions
                        return action, 0
                    else:
                        rewards.append(reward)

            best_action = np.argmax(rewards)
            return best_action, None
