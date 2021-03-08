import gym
from stable_baselines3 import DQN

from src.attacks.white_box_monte_carlo_agent import WhiteBoxMonteCarloAgent
from src.attacks.white_box_sequence_agent import WhiteBoxSequenceAgent
from src.common.opponent_wrapper import ObserveOpponent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.naive_selfplay_evaluation import evaluate
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper
from ma_gym.wrappers import Monitor
from ma_gym.envs.pong_duel import pong_duel


def main(save_video=False, num_eps=1, render=False, attack=None, save_perturbed_img=False):
    pong_duel.AGENT_COLORS[1] = 'red'
    # Initialize environment
    env = gym.make('PongDuel-v0')
    env = RewardZeroToNegativeBiAgentWrapper(env)
    if save_video:
        env = Monitor(env, './output/recordings', video_callable=lambda episode_id: True, force=True)
    # env = RewardZeroToNegativeBiAgentWrapper(env)
    # env = ObservationVectorToImage(env, 'both')
    env = ObserveOpponent(env, 'both')
    env = MAGymCompatibilityWrapper(env, image_observations='none')
    model_dir = '../../output/gcp-models/'

    # Models
    op_name = 'gcp-feature-based-op-obs7.out'

    model = WhiteBoxMonteCarloAgent(env, num_sims=10, sim_max_steps=2000)
    op = DQN.load(model_dir + op_name)
    env.set_opponent(op)
    avg_reward, total_steps = evaluate(model, env, attack=attack, slowness=0.05, num_eps=num_eps, render=render,
                                       save_perturbed_img=save_perturbed_img, print_obs=False, verbose=False, img_obs=False)
    print(avg_reward)
    print(total_steps)


if __name__ == '__main__':
    main()
