import gym
import numpy as np

import stable_baselines3 as sb3
from src.common.stable_baselines_wrapper import StableBaselinesWrapper

from src.common.utils import load_with_p1_name, load_with_p2_name
from src.selfplay.model import DQN, Policy
from time import sleep

def train_adversarial_policy(env, args):
    # Load victim model
    victim_model = DQN(env, args, args.policy_attack).to(args.device)
    victim_policy = Policy(env, args, args.policy_attack).to(args.device)
    victim_model.eval(); victim_policy.eval()
    if args.policy_attack == 'p1':
        load_with_p1_name(victim_model, victim_policy, args, "PongDuel-v0-opp-robust-dqn-model.pth")
    elif args.policy_attack == 'p2':
        load_with_p2_name(victim_model, victim_policy, args, "PongDuel-v0-opp-robust-dqn-model.pth")
    else:
        raise AssertionError ("Argument takes value \"p1\" or \"p2\" but received {}".format(args.policy_attack))
    
    # Wrap the ma-gym environment compatible to stable baseline
    env = StableBaselinesWrapper(env, args.policy_attack, victim_policy)
    
    model = sb3.DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps= 0.8* args.max_frames + 1)

    if args.policy_attack == 'p1':
        model.save("adversary_p2")
    else:
        model.save("adversary_p1")

    # Evaluate the trained adversary 
    model.set_env(env)
    adv_reward_list = []
    length_list = []
    for _ in range(10):
        state = env.reset()
        adv_episode_reward = 0
        episode_length = 0
        while True:
            adv_action, _states = model.predict(state)
            state, reward, done, info = env.step(adv_action)
            adv_episode_reward += reward
            episode_length += 1
    
            if done:
                print(adv_episode_reward)
                adv_reward_list.append(adv_episode_reward)
                length_list.append(episode_length)
                break

    print("Test Result - Length {:.2f} Adv/Reward {:.2f}".format(
        np.mean(length_list), np.mean(adv_reward_list)))
