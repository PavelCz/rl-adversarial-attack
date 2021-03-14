import gym
from stable_baselines3 import DQN, SAC, A2C

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.attacks.opponent_pred_as_obs_wrapper import OpponentPredictionObs
from src.common.image_wrapper import ObservationVectorToImage
from src.common.info_wrapper import InfoWrapper
from src.common.opponent_wrapper import ObserveOpponent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.naive_selfplay_evaluation import evaluate
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper
from ma_gym.wrappers import Monitor
from ma_gym.envs.pong_duel import pong_duel


def main():
    # Only necessary for when working with image observations
    pong_duel.AGENT_COLORS[1] = 'red'

    model_dir = '../../output/models/'
    num_eps = 200
    attack = None

    agent_name = 'adv-trainedA7.out'
    op_name = None  # 'gcp-feature-based-op-obs7.out'
    # agent_name = 'gcp-feature-based-op-obs6.out'
    # op_name = 'gcp-feature-based-op-obs6.out'
    # op_name = 'models/gcp-feature-based-op-obs6.out'

    # ==== EVALUATION ====

    # Initialize environment
    env = _make_env(save_video=False)
    env.seed(1)

    model, op = _make_agents(env, model_dir, agent_name, op_name=op_name)

    env.set_opponent(op)
    avg_reward1, _, infos1 = evaluate(model,
                                      env,
                                      attack=attack,
                                      num_eps=num_eps // 2,
                                      return_infos=True)

    # Repeat previous evaluation with sides switched

    # Initialize environment
    env = _make_env(save_video=False)
    env.seed(1)

    # Initialize agents with switched sides
    model, op = _make_agents(env, model_dir, op_name, op_name=agent_name)

    env.set_opponent(op)
    avg_reward2, _, infos2 = evaluate(model,
                                      env,
                                      attack=attack,
                                      num_eps=num_eps // 2,
                                      return_infos=True)

    # Calculate the reward for player 1
    # Because player 1 and 2 are switched in the second evaluation, we the inverse is the reward for p1
    avg_reward = (avg_reward1 + (1 - avg_reward2)) / 2

    p1_hits = infos1['p1_ball_hits'] + infos2['p2_ball_hits']
    p1_ball_receives = infos1['ball_moved_towards_p1'] + infos2['ball_moved_towards_p2']
    p1_misses = p1_ball_receives - p1_hits
    p1_miss_percentage = p1_misses / p1_ball_receives

    p2_hits = infos1['p2_ball_hits'] + infos2['p1_ball_hits']
    p2_ball_receives = infos1['ball_moved_towards_p2'] + infos2['ball_moved_towards_p1']
    p2_misses = p2_ball_receives - p2_hits
    p2_miss_percentage = p2_misses / p2_ball_receives

    print(avg_reward)
    print(f'P1 miss percentage: {p1_miss_percentage}')
    print(f'P2 miss percentage: {p2_miss_percentage}')


def _make_agents(env, model_dir, agent_name, op_name=None):
    # Models
    if agent_name is None:
        model = SimpleRuleBasedAgent(env)
    else:
        model = DQN.load(model_dir + agent_name)

    if op_name is None:
        op = SimpleRuleBasedAgent(env)
    else:
        op = DQN.load(model_dir + op_name)
    return model, op


def _make_env(save_video, max_rounds=None):
    if max_rounds is None:
        env = gym.make('PongDuel-v0')
    else:
        env = gym.make('PongDuel-v0', max_rounds=max_rounds)

    env = InfoWrapper(env)
    if save_video:
        env = Monitor(env, './output/recordings', video_callable=lambda episode_id: True, force=True)
    # env = RewardZeroToNegativeBiAgentWrapper(env)
    # env = ObservationVectorToImage(env, 'both')
    env = ObserveOpponent(env, 'both')
    env = MAGymCompatibilityWrapper(env, image_observations='none')
    # env = OpponentPredictionObs(env)
    return env


if __name__ == '__main__':
    main()
