import gym
from stable_baselines3 import DQN, SAC, A2C
import glob

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.naive_selfplay_evaluation import evaluate
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper
from ma_gym.wrappers import Monitor


def main(num_eps=1, render=True):
    # Initialize environment
    env = gym.make('PongDuel-v0')
    # env = RewardZeroToNegativeBiAgentWrapper(env)
    env = MAGymCompatibilityWrapper(env)
    op = RandomAgent(env)
    env.set_opponent(op)
    model_dir = 'output/models/'
    model_name = "dqn-500k"
    for i in range(1, 40):
        print(f'Evaluating model {model_name}{i}')
        model = DQN.load(model_dir + model_name + str(i) + '.out')
        model.set_env(env)
        avg_reward, _ = evaluate(model, env, slowness=0.05, num_eps=num_eps, render=render)
        print(avg_reward)


if __name__ == '__main__':
    main(num_eps=1000, render=False)
