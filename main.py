import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import ma_gym
from ma_gym.wrappers import Monitor
import time, os
from tensorboardX import SummaryWriter

from src.common.utils import create_log_dir, print_args, set_global_seeds
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.common.image_wrapper import ObservationVectorToImage
from src.common.opponent_wrapper import ObserveOpponent
from src.common.info_wrapper import InfoWrapper

from src.selfplay.arguments import get_args
from src.scripts.train import train
from src.scripts.test import test
from src.attacks.adversarial_policy import train_adversarial_policy

def main():
    args = get_args()
    print_args(args)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)

    env = gym.make(args.env)

    # Seed for training reproducibility
    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.obs_opp:
        env = ObserveOpponent(env, args.obs_opp)
    if args.obs_img:
        env = ObservationVectorToImage(env, args.obs_img)

    if args.evaluate:
        if not args.obs_img:
            env = InfoWrapper(env)
        if args.monitor:
            env = Monitor(env, './', video_callable=lambda episode_id: True, force=True)
        test(env, args)
        env.close()
        return

    # Modify original reward to a binary reward: +1 for scoring and -1 for missing the ball in training 
    env = RewardZeroToNegativeBiAgentWrapper(env)
    if args.policy_attack:
        train_adversarial_policy(env, args)
        env.close()
        return
    train(env, args, writer)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
