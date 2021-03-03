import gym
import numpy as np
import copy


class WhiteBoxMonteCarloAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def predict(self, obs, deterministic=True):
        # obs not necessary here, as the state is implicitly in the current state of self.env

        max_steps = 1000
        rewards = []
        for action in range(self.env.action_space.n):
            # Make copy of environment for internal lookahead
            sim_env = copy.deepcopy(self.env)

            # Perform step in simulated env
            _, reward, _, _ = sim_env.step(action)

            if reward != 0:
                rewards.append(reward)
            else:
                for i in range(max_steps):
                    # Perform step in simulated env
                    _, reward, _, _ = sim_env.step(self.env.action_space.sample())
                    # Encountered state with reward -> stop simulation
                    if reward != 0:
                        break
                # After reaching max_steps no reward != 0 encountered
                rewards.append(reward)

        best_action = np.argmax(rewards)
        return best_action, None
