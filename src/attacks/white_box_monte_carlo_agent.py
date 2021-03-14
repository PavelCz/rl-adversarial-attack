import gym
import numpy as np
import copy


class WhiteBoxMonteCarloAgent:
    """
    A white-box adversarial policy attempts based on Monte Carlo evaluation. Use white-box information to predict and perfectly simulate
    future states. Choose state which is most advantageous according to simulations.
    """
    def __init__(self, env: gym.Env, num_sims, sim_max_steps):
        self.env = env
        self.num_sims = num_sims
        self.sim_max_steps = sim_max_steps

    def predict(self, obs, deterministic=True):
        # obs not necessary here, as the state is implicitly in the current state of self.env

        rewards = []
        for action in range(self.env.action_space.n):
            sum_reward = 0
            for sim in range(self.num_sims):
                # Make copy of environment for internal lookahead
                sim_env = copy.deepcopy(self.env)

                # Perform step in simulated env
                _, reward, _, _ = sim_env.step(action)

                if reward != 0:
                    sum_reward += reward
                else:
                    for i in range(self.sim_max_steps):
                        # Perform step in simulated env
                        _, reward, _, _ = sim_env.step(self.env.action_space.sample())
                        # Encountered state with reward -> stop simulation
                        if reward != 0:
                            break
                    # After reaching max_steps no reward != 0 encountered
                    sum_reward += reward

            avg_reward = sum_reward / self.num_sims
            rewards.append(avg_reward)
        # Choose the action with the highest average reward
        best_action = np.argmax(rewards)
        return best_action, None
