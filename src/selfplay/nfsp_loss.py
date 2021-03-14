import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import random
import numpy as np

def compute_sl_loss(policy, reservoir_buffer, optimizer, args):
    state, action = reservoir_buffer.sample(args.batch_size)

    state = torch.tensor(np.float32(state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)

    loss = best_response_loss(state, action, policy)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def best_response_loss(state, action, model):
    probs = model(state)
    probs_with_actions = probs.gather(1, action.unsqueeze(1))
    log_probs = probs_with_actions.log()
    return -1 * log_probs.mean() 

def compute_rl_loss(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size).to(args.device)

    state = torch.tensor(np.float32(state)).to(args.device)
    next_state = torch.tensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.tensor(reward).to(args.device)
    done = torch.tensor(done).to(args.device)

    # DQN current Q values
    q_values = current_model(state)
    q_values_for_actions = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # DQN target Q values
    target_next_q_values = target_model(next_state)
    max_next_q_values = target_next_q_values.max(1)[0]

    target_q_values_for_actions = reward + (args.gamma ** args.frame_skipping) * max_next_q_values * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_values_for_actions, target_q_values_for_actions.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_rl_loss_DDQN(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size).to(args.device)

    state = torch.tensor(np.float32(state)).to(args.device)
    next_state = torch.tensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.tensor(reward).to(args.device)
    done = torch.tensor(done).to(args.device)

    # Double DQN(DDQN) current Q values
    q_values = current_model(state)
    q_values_for_actions = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # Double DQN(DDQN) current next Q values and target next Q values
    current_next_q_values = current_model(next_state)
    target_next_q_values = target_model(next_state)
    next_q_values = target_next_q_values.gather(1, current_next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)

    target_q_values_for_actions = reward + (args.gamma ** args.frame_skipping) * next_q_values * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_values_for_actions, target_q_values_for_actions.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss