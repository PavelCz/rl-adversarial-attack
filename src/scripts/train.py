import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import random
import numpy as np
from collections import deque
from time import sleep

from src.common.utils import epsilon_scheduler, update_target, print_log, load_model, save_model
from src.selfplay.model import DQN, Policy
from src.selfplay.storage import ReplayBuffer, ReservoirBuffer

def train(env, args, writer):
    # Initialize replay memories Replay Buffer Rewervoir Buffer
    # Replay Buffer for RL - Best Response
    p1_replay_buffer = ReplayBuffer(args.buffer_size)
    p2_replay_buffer = ReplayBuffer(args.buffer_size)

    # Reservoir Buffer for SL - Average Strategy
    p1_reservoir_buffer = ReservoirBuffer(args.buffer_size)
    p2_reservoir_buffer = ReservoirBuffer(args.buffer_size)

    # Initialize Action Value Network and target network
    # RL Model for Player 1
    p1_current_model = DQN(env, args).to(args.device)
    p1_target_model = DQN(env, args).to(args.device)
    update_target(p1_current_model, p1_target_model)

    # RL Model for Player 2
    p2_current_model = DQN(env, args).to(args.device)
    p2_target_model = DQN(env, args).to(args.device)
    update_target(p2_current_model, p2_target_model)

    # Initialize Average Policy Network
    # SL Model for Player 1, 2
    p1_policy = Policy(env, args).to(args.device)
    p2_policy = Policy(env, args).to(args.device)

    if args.load_model and os.path.isfile(args.load_model):
        load_model(models={"p1": p1_current_model, "p2": p2_current_model},
                   policies={"p1": p1_policy, "p2": p2_policy}, args=args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)

    # RL Optimizer for Player 1, 2
    p1_rl_optimizer = optim.Adam(p1_current_model.parameters(), lr=args.lr)
    p2_rl_optimizer = optim.Adam(p2_current_model.parameters(), lr=args.lr)

    # SL Optimizer for Player 1, 2
    p1_sl_optimizer = optim.Adam(p1_policy.parameters(), lr=args.lr)
    p2_sl_optimizer = optim.Adam(p2_policy.parameters(), lr=args.lr)

    # Logging
    round = 0
    total_hit = 20
    p1_reward, p2_reward = 0, 0
    length_list = []
    p1_reward_list, p1_rl_loss_list, p1_sl_loss_list = [], [], []
    p2_reward_list, p2_rl_loss_list, p2_sl_loss_list = [], [], []
    p1_episode_reward, p2_episode_reward = 0, 0
    tag_interval_length = 0
    prev_time = time.time()
    prev_frame = 1

    # Main Loop
    state = env.reset()
    p1_state = state[0]
    p2_state = state[1]
    for frame_idx in range(1, args.max_frames + 1): 
        is_best_response = False
        # Policy is decided by a combination of Best Response and Average Strategy
        if random.random() > args.eta:
            # With probability 1 - eta choose average strategy pi
            p1_action = p1_policy.act(torch.FloatTensor(p1_state).to(args.device))
            p2_action = p2_policy.act(torch.FloatTensor(p1_state).to(args.device))
        else:
            # With probability eta choose best response strategy beta
            is_best_response = True
            epsilon = epsilon_by_frame(frame_idx)
            p1_action = p1_current_model.act(torch.FloatTensor(p1_state).to(args.device), epsilon)
            p2_action = p2_current_model.act(torch.FloatTensor(p2_state).to(args.device), epsilon)

        # Get actions from chosen policy
        actions = [p1_action, p2_action]
        next_state, reward, done, info = env.step(actions)

        p1_next_state = next_state[0]
        p2_next_state = next_state[1]
        
        # p1_reward = reward[0] - 1 if args.negative else reward[0]
        # p2_reward = reward[1] - 1 if args.negative else reward[1]

        # Direction of the ball
        ball_dir = np.argmax(p1_state[-6:])
        ball_dir_next = np.argmax(p1_next_state[-6:])

        next_round = info["rounds"]

        # Try to modify the reward by get +1 for hitting the ball but minus score when miss the ball
        if next_round == round:
            if ball_dir <= 2 and ball_dir_next > 2:
                p1_reward += 1
                total_hit -= 1
            elif ball_dir > 2 and ball_dir_next <= 2:
                p2_reward +=1
                total_hit -= 1
        elif next_round != round:
            round = next_round
            length_list.append(tag_interval_length)
            tag_interval_length = 0
            if reward[0] == 1:
                p2_reward -= max(total_hit,0)
            elif reward[1] == 1:
                p1_reward -= max(total_hit,0)

        # Logging
        p1_episode_reward += p1_reward
        p2_episode_reward += p2_reward
        # if next_round != round:
        #     round = next_round
        #     length_list.append(tag_interval_length)
        #     tag_interval_length = 0

        tag_interval_length += 1

        # Store (state, action, reward, next_state) to Replay Buffer for Reinforcement Learning
        p1_replay_buffer.push(p1_state, p1_action, p1_reward, p1_next_state, np.float32(done[0]))
        p2_replay_buffer.push(p2_state, p2_action, p2_reward, p2_next_state, np.float32(done[1]))
        
        # Store (state, action) to Reservoir Buffer for Supervised Learning if agent
        # follows best response policy
        if is_best_response:
            p1_reservoir_buffer.push(p1_state, p1_action)
            p2_reservoir_buffer.push(p2_state, p2_action)

        (p1_state, p2_state) = (p1_next_state, p2_next_state)


        # Episode done. Reset environment and clear logging records
        if all(done) or tag_interval_length >= args.max_tag_interval:
            state = env.reset()
            p1_state = state[0]; p2_state = state[1]
            p1_reward_list.append(p1_episode_reward)
            p2_reward_list.append(p2_episode_reward)
            writer.add_scalar("p1/episode_reward", p1_episode_reward, frame_idx)
            writer.add_scalar("p2/episode_reward", p2_episode_reward, frame_idx)
            writer.add_scalar("data/tag_interval_length", tag_interval_length, frame_idx)
            p1_episode_reward, p2_episode_reward, tag_interval_length, round = 0, 0, 0, 0
            p1_reward, p2_reward = 0, 0
            total_hit = 20

        if (len(p1_replay_buffer) > args.rl_start and
            len(p1_reservoir_buffer) > args.sl_start and
            frame_idx % args.train_freq == 0):
            # Update Best Response with Reinforcement Learning
            loss = compute_rl_loss(p1_current_model, p1_target_model, p1_replay_buffer, p1_rl_optimizer, args)
            p1_rl_loss_list.append(loss.item())
            writer.add_scalar("p1/rl_loss", loss.item(), frame_idx)

            loss = compute_rl_loss(p2_current_model, p2_target_model, p2_replay_buffer, p2_rl_optimizer, args)
            p2_rl_loss_list.append(loss.item())
            writer.add_scalar("p2/rl_loss", loss.item(), frame_idx)

            # Update Average Strategy with Supervised Learning
            loss = compute_sl_loss(p1_policy, p1_reservoir_buffer, p1_sl_optimizer, args)
            p1_sl_loss_list.append(loss.item())
            writer.add_scalar("p1/sl_loss", loss.item(), frame_idx)

            loss = compute_sl_loss(p2_policy, p2_reservoir_buffer, p2_sl_optimizer, args)
            p2_sl_loss_list.append(loss.item())
            writer.add_scalar("p2/sl_loss", loss.item(), frame_idx)
        

        if frame_idx % args.update_target == 0:
            update_target(p1_current_model, p1_target_model)
            update_target(p2_current_model, p2_target_model)


        # Logging and Saving models
        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, (p1_reward_list, p2_reward_list), length_list,
                      (p1_rl_loss_list, p2_rl_loss_list), (p1_sl_loss_list, p2_sl_loss_list))
            p1_reward_list.clear(), p2_reward_list.clear(), length_list.clear()
            p1_rl_loss_list.clear(), p2_rl_loss_list.clear()
            p1_sl_loss_list.clear(), p2_sl_loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            save_model(models={"p1": p1_current_model, "p2": p2_current_model},
                       policies={"p1": p1_policy, "p2": p2_policy}, args=args)
        
        # Render if rendering argument is on
        if args.render:
            env.render()


def compute_sl_loss(policy, reservoir_buffer, optimizer, args):
    state, action = reservoir_buffer.sample(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)

    probs = policy(state)
    probs_with_actions = probs.gather(1, action.unsqueeze(1))
    log_probs = probs_with_actions.log()

    loss = -1 * log_probs.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_rl_loss(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    # Q-Learning with target network
    q_values = current_model(state)
    target_next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.max(1)[0]

    expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret