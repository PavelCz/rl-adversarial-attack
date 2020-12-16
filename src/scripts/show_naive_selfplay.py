import gym
from stable_baselines3 import DQN

from src.selfplay.learning import evaluate
from src.selfplay.opponent_wrapper import OpponentWrapper


def main():
    # Initialize environment
    env = gym.make('PongDuel-v0')
    env = OpponentWrapper(env)
    model_dir = '../output/models/'
    agent_name = "dqn-29.out"
    model = DQN.load(model_dir + agent_name)
    env.set_opponent(model)
    evaluate(model, env, num_eps=1, render=True)


if __name__ == '__main__':
    main()
