import gym
from stable_baselines3 import DQN, SAC, A2C

from src.selfplay.learning import evaluate
from src.selfplay.opponent_wrapper import OpponentWrapper


def main():
    # Initialize environment
    env = gym.make('PongDuel-v0')
    env = OpponentWrapper(env)
    model_dir = 'output/models/'
    agent_name = "sac-6.out"
    model = A2C.load(model_dir + agent_name)
    env.set_opponent(model)
    evaluate(model, env, slowness=0.5, num_eps=1, render=True, print_obs=True)


if __name__ == '__main__':
    main()
