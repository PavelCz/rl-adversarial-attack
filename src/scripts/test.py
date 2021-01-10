import torch
import torch.optim as optim

import numpy as np
import os

from time import sleep
from src.common.utils import load_model
from src.selfplay.model import DQN, Policy
from src.attacks.fgsm import fgsm_attack

def test(env, args): 
    p1_current_model = DQN(env, args).to(args.device)
    p2_current_model = DQN(env, args).to(args.device)
    p1_policy = Policy(env, args).to(args.device)
    p2_policy = Policy(env, args).to(args.device)
    p1_current_model.eval(), p2_current_model.eval()
    p1_policy.eval(), p2_policy.eval()

    load_model(models={"p1": p1_current_model, "p2": p2_current_model},
               policies={"p1": p1_policy, "p2": p2_policy}, args=args)

    p1_reward_list = []
    p2_reward_list = []
    length_list = []

    for _ in range(10):
        state = env.reset()
        p1_state = state[0]
        p2_state = state[1]
        p1_episode_reward = 0
        p2_episode_reward = 0
        episode_length = 0
        while True:
            if args.render:
                env.render()
                sleep(0.01)

            # Random Action Agent
            # p1_action = env.action_space.sample()[0]
            # p2_action = env.action_space.sample()[1]

            # Agents follow average strategy
            if args.fgsm == 1:
                p1_state = fgsm_attack(torch.tensor(p1_state).to(args.device), p1_policy, 0.05, args)
            elif args.fgsm == 2:
                p2_state = fgsm_attack(torch.tensor(p2_state).to(args.device), p2_policy, 0.05, args)

            p1_action = p1_policy.act(torch.tensor(p1_state).to(args.device))
            p2_action = p2_policy.act(torch.tensor(p2_state).to(args.device))

            actions = [p1_action, p2_action]
            next_state, reward, done, info = env.step(actions)
            p1_state = next_state[0] ; p2_state = next_state[1]
            p1_episode_reward += reward[0]
            p2_episode_reward += reward[1]
            episode_length += 1
            if all(done):
                print(p1_episode_reward, p2_episode_reward)
                p1_reward_list.append(p1_episode_reward)
                p2_reward_list.append(p2_episode_reward)
                length_list.append(episode_length)
                break

    print("Test Result - Length {:.2f} p1/Reward {:.2f} p2/Reward {:.2f}".format(
        np.mean(length_list), np.mean(p1_reward_list), np.mean(p2_reward_list)))
    