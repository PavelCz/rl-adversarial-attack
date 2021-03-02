import gym
from stable_baselines3 import DQN, SAC, A2C

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.common.image_wrapper import ObservationVectorToImage
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.naive_selfplay_training import evaluate
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper
from ma_gym.wrappers import Monitor
from ma_gym.envs.pong_duel import pong_duel


def main(save_video=False, num_eps=1, render=True, attack="fgsm", save_perturbed_img=False):
    pong_duel.AGENT_COLORS[1] = 'red'
    # Initialize environment
    env = gym.make('PongDuel-v0')
    if save_video:
        env = Monitor(env, './output/recordings', video_callable=lambda episode_id: True, force=True)
    # env = RewardZeroToNegativeBiAgentWrapper(env)
    env = ObservationVectorToImage(env, 'both')
    env = MAGymCompatibilityWrapper(env, image_observations='both')
    model_dir = 'output/gcp_models/'

    # Models
    agent_name = "dqn-img-obs-training3.out"
    op_name = 'dqn-img-obs-training3.out'

    model = DQN.load(model_dir + agent_name)
    # op = RandomAgent(env)
    # op = SimpleRuleBasedAgent(env)
    op = DQN.load(model_dir + op_name)
    env.set_opponent(op)
    avg_reward, _ = evaluate(model, env, attack=attack, slowness=0.05, num_eps=num_eps, render=render,
                             save_perturbed_img=save_perturbed_img, print_obs=False, verbose=False, img_obs=True)
    print(avg_reward)


if __name__ == '__main__':
    main()
