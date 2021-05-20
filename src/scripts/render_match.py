import gym
from stable_baselines3 import DQN, SAC, A2C

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.attacks.opponent_pred_as_obs_wrapper import OpponentPredictionObs
from src.common.image_wrapper import ObservationVectorToImage
from src.common.info_wrapper import InfoWrapper
from src.common.opponent_wrapper import ObserveOpponent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.scripts.evaluate_model import _make_env, _make_agents
from src.selfplay.naive_selfplay_evaluation import evaluate
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper
from ma_gym.wrappers import Monitor
from ma_gym.envs.pong_duel import pong_duel


def main():
    model_dir = '/home/pavelc/_code/output/ucb/remote/models/'
    #model_dir = "/home/pavel/code/tum-adlr/output/ucb/models/"
    attack = None
    save_video = False

    agent_name = 'dropout1M-remote25.out'
    op_name = '../adversaries/models/do-1M-adversary.out1.out'  # 'models/gcp-feature-based-op-obs7.out'  # "gcp-fine-tuned2.out"

    # Initialize environment
    env = _make_env(save_video=save_video, max_rounds=10)

    model, op = _make_agents(env, model_dir, agent_name, op_name=op_name)

    env.set_opponent(op)
    result = evaluate(model,
                      env,
                      render=True,
                      attack=attack,
                      num_eps=1,
                      return_infos=True)
    print(result)


if __name__ == '__main__':
    main()
