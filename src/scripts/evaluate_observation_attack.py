import gym
from src.scripts.evaluate_model import _make_agents, _make_env
from src.selfplay.naive_selfplay_evaluation import evaluate
from ma_gym.envs.pong_duel import pong_duel


def main():
    # Only necessary for when working with image observations
    pong_duel.AGENT_COLORS[1] = 'red'

    model_dir = '../../output/models/'
    num_eps = 200
    attack = 'fgsm'

    agent_name = 'gcp-feature-based-op-obs7.out'
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
                                      num_eps=num_eps,
                                      return_infos=True)

    # Repeat previous evaluation with sides switched

    p1_hits = infos1['p1_ball_hits']
    p1_ball_receives = infos1['ball_moved_towards_p1']
    p1_misses = p1_ball_receives - p1_hits
    p1_miss_percentage = p1_misses / p1_ball_receives

    print(avg_reward1)
    print(f'P1 miss percentage: {p1_miss_percentage}')


if __name__ == '__main__':
    main()
