import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from src.common.stable_baselines_wrapper import StableBaselinesWrapper

from src.common.utils import load_model_single
from src.selfplay.model import DQN, Policy

def train_adversarial_policy(env, args):
    # Load victim model
    if args.adv_policy == 'p1' or args.adv_policy == 'p2':
        victim_model = DQN(env, args, args.adv_policy).to(args.device)
        victim_policy = Policy(env, args, args.adv_policy).to(args.device)
        victim_model.eval(); victim_policy.eval()
        load_model_single(victim_model, victim_policy, args, args.adv_policy)
    else:
        raise AssertionError ("Argument takes value \"p1\" or \"p2\" but received {}".format(args.adv_policy))
    
    env = StableBaselinesWrapper(env, args.adv_policy, victim_policy)
    # Load adversary model
    model = PPO(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=args.max_frames + 1)
    model.learn(total_timesteps=args.max_frames+1)

    model.save("adversary")

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
