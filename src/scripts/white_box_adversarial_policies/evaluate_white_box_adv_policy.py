import gym
from stable_baselines3 import DQN, SAC, A2C

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.attacks.white_box_adversarial_policy import WhiteBoxAdversarialAgent
from src.common.image_wrapper import ObservationVectorToImage
from src.common.opponent_wrapper import ObserveOpponent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.naive_selfplay_evaluation import evaluate
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper
from ma_gym.wrappers import Monitor
from ma_gym.envs.pong_duel import pong_duel


def main(save_video=False, num_eps=1, render=True, attack=None, save_perturbed_img=False):
    # Initialize environment
    env = gym.make('PongDuel-v0')
    if save_video:
        env = Monitor(env, './output/recordings', video_callable=lambda episode_id: True, force=True)
    # env = RewardZeroToNegativeBiAgentWrapper(env)
    env = ObserveOpponent(env, 'both')
    env = MAGymCompatibilityWrapper(env, image_observations='none')
    model_dir = '../../output/gcp-models/'

    # Models
    agent_name = "gcp-feature-based-op-obs7.out"

    victim = DQN.load(model_dir + agent_name)

    adv = WhiteBoxAdversarialAgent(env, victim, victim_type='sb3')

    env.set_opponent(victim)
    avg_reward, _ = evaluate(adv, env, attack=attack, slowness=0.05, num_eps=num_eps, render=render,
                             save_perturbed_img=save_perturbed_img, )
    print(avg_reward)


if __name__ == '__main__':
    main()
