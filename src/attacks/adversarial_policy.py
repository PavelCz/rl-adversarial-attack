import gym
import numpy as np

import stable_baselines3 as sb3
from src.common.stable_baselines_wrapper import StableBaselinesWrapper

from src.common.utils import load_model_single
from src.selfplay.model import DQN, Policy
from time import sleep

def train_adversarial_policy(env, args):
    # Load victim model
    if args.policy_attack == 'p1' or args.policy_attack == 'p2':
        victim_model = DQN(env, args, args.policy_attack).to(args.device)
        victim_policy = Policy(env, args, args.policy_attack).to(args.device)
        victim_model.eval(); victim_policy.eval()
        load_model_single(victim_model, victim_policy, args, args.policy_attack)
    else:
        raise AssertionError ("Argument takes value \"p1\" or \"p2\" but received {}".format(args.policy_attack))
    
    env = StableBaselinesWrapper(env, args.policy_attack, victim_policy)
    # Load adversary model
    model = sb3.DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.max_frames + 1)

    if args.policy_attack == 'p1':
        model.save("adversary_p2")
    else:
        model.save("adversary_p1_bounce")
    # model = sb3.DQN.load("adversary_p1")
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
            # env.render()
            adv_episode_reward += reward
            episode_length += 1
    
            if done:
                print(adv_episode_reward)
                adv_reward_list.append(adv_episode_reward)
                length_list.append(episode_length)
                break

    print("Test Result - Length {:.2f} Adv/Reward {:.2f}".format(
        np.mean(length_list), np.mean(adv_reward_list)))
