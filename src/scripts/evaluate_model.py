import gym
from stable_baselines3 import DQN, SAC, A2C

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.attacks.opponent_pred_as_obs_wrapper import OpponentPredictionObs
from src.common.image_wrapper import ObservationVectorToImage
from src.common.info_wrapper import InfoWrapper
from src.common.opponent_wrapper import ObserveOpponent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.naive_selfplay_training import evaluate
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper
from ma_gym.wrappers import Monitor
from ma_gym.envs.pong_duel import pong_duel


def main(save_video=False, num_eps=50, render=False, attack=None, save_perturbed_img=False):
    pong_duel.AGENT_COLORS[1] = 'red'
    # Initialize environment
    env = gym.make('PongDuel-v0')
    env = InfoWrapper(env)
    if save_video:
        env = Monitor(env, './output/recordings', video_callable=lambda episode_id: True, force=True)
    # env = RewardZeroToNegativeBiAgentWrapper(env)
    # env = ObservationVectorToImage(env, 'both')
    env = ObserveOpponent(env, 'both')
    env = MAGymCompatibilityWrapper(env, image_observations='none')
    # env = OpponentPredictionObs(env)
    model_dir = '../../output/models/models/'

    # Models
    agent_name = "gcp-fine-tuned2.out"
    # agent_name = 'gcp-feature-based-op-obs6.out'

    #op_name = 'gcp-feature-based-op-obs6.out'
    # op_name = 'models/gcp-feature-based-op-obs6.out'

    model = DQN.load(model_dir + agent_name)
    # op = RandomAgent(env)
    op = SimpleRuleBasedAgent(env)
    # op = DQN.load(model_dir + op_name)
    env.set_opponent(op)
    avg_reward, total_steps, infos = evaluate(model, env, attack=attack, slowness=0.05, num_eps=num_eps, render=render,
                             save_perturbed_img=save_perturbed_img, print_obs=False, verbose=False, img_obs=False, return_infos=True)
    print(avg_reward)
    print(total_steps)
    print(infos)


if __name__ == '__main__':
    main()
