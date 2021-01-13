import os
import copy
import time

import gym
import ma_gym  # Necessary so the PongDuel env exists
from stable_baselines3 import PPO, DQN, A2C

from stable_baselines3.a2c import MlpPolicy

from src.agents.random_agent import RandomAgent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.opponent_wrapper import OpponentWrapper


def learn(max_agents, num_learn_steps, num_eval_eps, num_skip_steps=0):
    # Initialize environment
    env = gym.make('Pong-v0')

    main_model = A2C('CnnPolicy', env, verbose=0, tensorboard_log="output/tb-log")  # , exploration_fraction=0.3)

    print("Learning...")
    main_model.learn(total_timesteps=num_learn_steps, tb_log_name="log")

    obs = env.reset()
    rewards = []
    done = False
    while not done:
        action, _states = main_model.predict(obs, deterministic=False)
        print(action)
        ob, reward, done, info = env.step(action)
        env.render()
        rewards.append(reward)
    print(f"{sum(rewards) / len(rewards)}")


if __name__ == '__main__':
    learn(0, 2000, 0)
