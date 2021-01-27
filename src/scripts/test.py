import torch
import torch.optim as optim

import numpy as np
import os
import random

from time import sleep
from src.common.utils import load_model, load_model_tmp
from src.selfplay.model import DQN, Policy
from src.attacks.fgsm import fgsm_attack, plot_perturbed_view
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent

def test(env, args): 
    p1_current_model = DQN(env, args, 'p1').to(args.device)
    p2_current_model = DQN(env, args, 'p2').to(args.device)
    p1_policy = Policy(env, args, 'p1').to(args.device)
    p2_policy = Policy(env, args, 'p2').to(args.device)
    p1_current_model.eval(), p2_current_model.eval()
    p1_policy.eval(), p2_policy.eval()
    load_model(models={"p1": p1_current_model, "p2": p2_current_model},
               policies={"p1": p1_policy, "p2": p2_policy}, args=args)
    # load_model_tmp(models={"p1": p1_current_model},
                    # policies={"p1": p1_policy}, args=args)

    p1_reward_list = []
    p2_reward_list = []
    length_list = []
    p1 = SimpleRuleBasedAgent(env)
    p2 = SimpleRuleBasedAgent(env)
    env = env
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

            # Agents follow average strategy
            if args.fgsm == 1:
                p1_state = fgsm_attack(torch.tensor(p1_state).to(args.device), p1_policy, 0.02, args)
                # Plot perturbed state when paddle 1 might miss the ball 
                if p1_state[3] < 0.02 and args.plot_fgsm :
                    plot_perturbed_view(env.render(mode ='rgb_array'), p1_state)
            elif args.fgsm == 2:
                p2_state = fgsm_attack(torch.tensor(p2_state).to(args.device), p2_policy, 0.02, args)
                # Plot perturbed state when paddle 2 might miss the ball
                if p2_state[3] > 0.98 and args.plot_fgsm :
                    plot_perturbed_view(env.render(mode ='rgb_array'), p2_state)

            # Random Action Agent
            # p1_action = env.action_space.sample()[0]
            # p2_action = env.action_space.sample()[1]

            # Rule-based Angent
            # p1_action, _ = p1.predict(p1_state)
            # p2_action, _ = p2.predict(p2_state)

            # NFSP Agent
            p1_action = p1_policy.act(torch.tensor(p1_state).to(args.device))
            p2_action = p2_policy.act(torch.tensor(p2_state).to(args.device))

            # if random.random() > args.eta:
            #     p1_action = p1_policy.act(torch.tensor(p1_state).to(args.device))
            #     p2_action = p2_policy.act(torch.tensor(p1_state).to(args.device))
            # else:
            #     p1_action = p1_current_model.act(torch.tensor(p1_state).to(args.device), 0)
            #     p2_action = p2_current_model.act(torch.tensor(p2_state).to(args.device), 0)

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
    