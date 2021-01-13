import gym
from stable_baselines3 import DQN, SAC, A2C

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.learning import evaluate
from src.selfplay.opponent_wrapper import OpponentWrapper
from ma_gym.wrappers import Monitor


def main(save_video=False, num_eps=1):
    # Initialize environment
    env = gym.make('PongDuel-v0')
    if save_video:
        env = Monitor(env, './output/recordings', video_callable=lambda episode_id: True, force=True)
    # env = RewardZeroToNegativeBiAgentWrapper(env)
    env = OpponentWrapper(env)
    model_dir = '../output/models/'
    agent_name = "dqn-vs-rule-based-250k-1.out"
    model = DQN.load(model_dir + agent_name)
    op = RandomAgent(env)
    # op = SimpleRuleBasedAgent(env)
    env.set_opponent(op)
    avg_reward = evaluate(model, env, slowness=0.05, num_eps=num_eps, render=True, print_obs=False, verbose=False)
    print(avg_reward)


if __name__ == '__main__':
    main(100)
