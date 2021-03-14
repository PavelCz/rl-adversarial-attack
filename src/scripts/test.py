import torch
import torch.optim as optim

import numpy as np
import os
import random

from time import sleep
import stable_baselines3 as sb3

from src.common.utils import load_model, load_with_p1_name, load_with_p2_name
from src.selfplay.model import DQN, Policy
from src.attacks.fgsm import fgsm_attack, plot_perturbed
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.common.obervation_utils import flip_observation_horizontally

def test(env, args): 
    p1_current_model = DQN(env, args, 'p1').to(args.device)
    p2_current_model = DQN(env, args, 'p2').to(args.device)
    p1_policy = Policy(env, args, 'p1').to(args.device)
    p2_policy = Policy(env, args, 'p2').to(args.device)
    p1_current_model.eval(), p2_current_model.eval()
    p1_policy.eval(), p2_policy.eval()
    load_model(models={"p1": p1_current_model, "p2": p2_current_model},
               policies={"p1": p1_policy, "p2": p2_policy}, args=args)
    # Load adversary model
    if args.policy_attack:
        if args.policy_attack == 'p1':
            print("Load agent 2 adversary")
            adversary = sb3.DQN.load("adversary_p2")    
        elif args.policy_attack == 'p2':
            print("Load agent 1 adversary")
            adversary = sb3.DQN.load("adversary_p1")
        else:
            raise AssertionError ("Argument takes value \"p1\" or \"p2\" but received {}".format(args.policy_attack))
    p1_reward_list = []
    p2_reward_list = []
    length_list = []
    p1_ball_hits_list = []
    p2_ball_hits_list = []
    ball_moved_towards_p1_list = []
    ball_moved_towards_p2_list = []
    p1 = SimpleRuleBasedAgent(env)
    p2 = SimpleRuleBasedAgent(env)
    for _ in range(1):
        p1_state, p2_state = env.reset()
        p1_episode_reward = 0
        p2_episode_reward = 0
        episode_length = 0
        reward = [0,0]
        while True:
            if args.render:
                env.render()
                sleep(0.01)

            if args.fgsm:
                if args.fgsm == 'p1':
                    p1_state = fgsm_attack(torch.tensor(p1_state).to(args.device), p1_policy, 0.02, 'sl', args)
                    if args.plot_fgsm :
                        plot_perturbed(env.render(mode ='rgb_array'), p1_state, args)

                elif args.fgsm == 'p2':
                    p2_state = fgsm_attack(torch.tensor(p2_state).to(args.device), p2_policy, 0.02, 'sl', args)
                    if args.plot_fgsm :
                        plot_perturbed(env.render(mode ='rgb_array'), p2_state, args)
                elif args.fgsm == 'both':
                    p1_state = fgsm_attack(torch.tensor(p1_state).to(args.device), p1_policy, 0.02, 'sl', args)
                    p2_state = fgsm_attack(torch.tensor(p2_state).to(args.device), p2_policy, 0.02, 'sl', args)
                else:
                    raise AssertionError ("Argument takes value \"p1\" or \"p2\" but received {}".format(args.fgsm))
            
            # Remember to comment the corresponding NFSP agent
            if args.policy_attack:
                ball_dir = np.nonzero(p1_state[4:10])[0]
                if args.policy_attack == 'p1':
                    p2_action, _states = adversary.predict(p2_state)
                    if isinstance(p2_action, list) or isinstance(p2_action, np.ndarray):
                        p2_action = p2_action[0]
                else:
                    p1_action, _states = adversary.predict(p1_state)
                    if isinstance(p1_action, list) or isinstance(p1_action, np.ndarray):
                        p1_action = p1_action[0]
                    if ball_dir in [0,1,2]:
                        p1_action = p1_policy.act(torch.tensor(p1_state).to(args.device))
                    if ball_dir in [3,4,5]:
                        p1_action, _states = adversary.predict(p1_state)
                        if isinstance(p1_action, list) or isinstance(p1_action, np.ndarray):
                            p1_action = p1_action[0]

            # # Random Action Agent
            # p1_action = env.action_space.sample()[0]
            # p2_action = env.action_space.sample()[1]

            # Rule-based Angent
            # p1_action, _ = p1.predict(p1_state)
            # p2_action, _ = p2.predict(p2_state)

            # NFSP Agent
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
                p1_ball_hits_list.append(info["p1_ball_hits"])
                p2_ball_hits_list.append(info["p2_ball_hits"])
                ball_moved_towards_p1_list.append(info["ball_moved_towards_p1"])
                ball_moved_towards_p2_list.append(info["ball_moved_towards_p2"])
                length_list.append(episode_length)
                break

    print("Test Result - Length {:.2f} p1/Reward {:.2f} p2/Reward {:.2f}".format(
        np.mean(length_list), np.mean(p1_reward_list), np.mean(p2_reward_list)))
    p1_hit_prob = np.sum(p1_ball_hits_list)/np.sum(ball_moved_towards_p1_list)
    p2_hit_prob = np.sum(p2_ball_hits_list)/np.sum(ball_moved_towards_p2_list)
    print("P1 hits: {}, P2 hits: {}".format(p1_hit_prob, p2_hit_prob))