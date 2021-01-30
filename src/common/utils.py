import math
import os
import datetime
import time
import pathlib
import random

import torch
import numpy as np

def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)
    return function

def create_log_dir(args):
    log_dir = ""
    log_dir = log_dir + "{}-".format(args.env)
    if args.negative:
        log_dir = log_dir + "negative-"
    if args.frame_skipping != 1:
        log_dir = log_dir + "{}-step-".format(args.frame_skipping)
    if args.obs_img:
        log_dir = log_dir + "img-"
    if args.ddqn:
        log_dir = log_dir + "double-"
    log_dir = log_dir + "dqn-{}".format(args.save_model)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now
    log_dir = os.path.join("runs", log_dir)
    return log_dir

def print_log(frame, prev_frame, prev_time, rewards, length_list, rl_losses, sl_losses):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    p1_avg_reward, p2_avg_reward = (np.mean(rewards[0]), np.mean(rewards[1]))
    if len(rl_losses[0]) != 0:
        p1_avg_rl_loss, p2_avg_rl_loss = (np.mean(rl_losses[0]), np.mean(rl_losses[1]))
        p1_avg_sl_loss, p2_avg_sl_loss = (np.mean(sl_losses[0]), np.mean(sl_losses[1]))
    else:
        p1_avg_rl_loss, p2_avg_rl_loss = 0., 0.
        p1_avg_sl_loss, p2_avg_sl_loss = 0., 0.

    avg_length = np.mean(length_list)

    print("Frame: {:<8} FPS: {:.2f} Avg. Frames per Round: {:.2f}".format(frame, fps, avg_length))
    print("Player 1 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.4f}/{:.4f}".format(
        p1_avg_reward, p1_avg_rl_loss, p1_avg_sl_loss))
    print("Player 2 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.4f}/{:.4f}".format(
        p2_avg_reward, p2_avg_rl_loss, p2_avg_sl_loss))

def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

def name_file(args, load_agent=None):
    fname = "{}-".format(args.env)
    if args.negative:
        fname += "negative-"
    if args.frame_skipping != 1:
        fname += "{}-step-".format(args.frame_skipping)
    if args.obs_img == "both" or args.obs_img == load_agent:
        fname += "img-"
    if args.ddqn:
        fname += "double-"
    fname += "dqn-{}.pth".format(args.save_model)
    return fname

def save_model(models, policies, args):
    fname = name_file(args)
    fname = os.path.join("models", fname)

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save({
        'p1_model': models['p1'].state_dict(),
        'p2_model': models['p2'].state_dict(),
        'p1_policy': policies['p1'].state_dict(),
        'p2_policy': policies['p2'].state_dict(),
    }, fname)

def load_model(models, policies, args):
    p1_fname = name_file(args, 'p1')
    p1_fname = os.path.join("models", p1_fname)
    p2_fname = name_file(args, 'p2')
    p2_fname = os.path.join("models", p2_fname)
    print("Load agent 1 from {}".format(p1_fname))
    print("Load agent 2 from {}".format(p2_fname))
    if args.device == torch.device("cpu"):
        # Models save on GPU load on CPU
        map_location = lambda storage, loc: storage
    else:
        # Models save on GPU load on GPU
        map_location = None
    
    if not os.path.exists(p1_fname):
        raise ValueError("No model saved with name {}".format(p1_fname))
    if not os.path.exists(p2_fname):
        raise ValueError("No model saved with name {}".format(p2_fname))

    p1_checkpoint = torch.load(p1_fname, map_location)
    p2_checkpoint = torch.load(p2_fname, map_location)
    models['p1'].load_state_dict(p1_checkpoint['p1_model'])
    models['p2'].load_state_dict(p2_checkpoint['p2_model'])
    policies['p1'].load_state_dict(p1_checkpoint['p1_policy'])
    policies['p2'].load_state_dict(p2_checkpoint['p2_policy'])

def save_checkpoint(models, policies, frame_idx, optimizers, args):
    fname = name_file(args)
    fname = os.path.join("checkpoints", fname)

    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    torch.save({
        'p1_model': models['p1'].state_dict(),
        'p2_model': models['p2'].state_dict(),
        'p1_policy': policies['p1'].state_dict(),
        'p2_policy': policies['p2'].state_dict(),
        'frame_idx': frame_idx,
        'p1_model_optimizer': optimizers['p1_model'].state_dict(),
        'p2_model_optimizer': optimizers['p2_model'].state_dict(),
        'p1_policy_optimizer': optimizers['p1_policy'].state_dict(),
        'p2_policy_optimizer': optimizers['p2_policy'].state_dict(),
    }, fname)

def load_checkpoint(models, policies, optimizers, args):
    fname = name_file(args)
    fname = os.path.join("checkpoints", fname)
    # Hack to load models saved with GPU
    if args.device == torch.device("cpu"):
        # Models save on GPU load on CPU
        map_location = lambda storage, loc: storage
    else:
        # Models save on CPU load on CPU
        map_location = None
    
    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    checkpoint = torch.load(fname, map_location)
    models['p1'].load_state_dict(checkpoint['p1_model'])
    models['p2'].load_state_dict(checkpoint['p2_model'])
    policies['p1'].load_state_dict(checkpoint['p1_policy'])
    policies['p2'].load_state_dict(checkpoint['p2_policy'])
    optimizers['p1_model'].load_state_dict(checkpoint['p1_model_optimizer'])
    optimizers['p2_model'].load_state_dict(checkpoint['p2_model_optimizer'])
    optimizers['p1_policy'].load_state_dict(checkpoint['p1_policy_optimizer'])
    optimizers['p2_policy'].load_state_dict(checkpoint['p2_policy_optimizer'])
    return checkpoint['frame_idx']+1

def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)

def load_model_tmp(models, policies, args):
    fname = name_file(args)
    # fname = "tmp" + fname
    fname = os.path.join("models", fname)
    print(fname)
    if args.device == torch.device("cpu"):
        # Models save on GPU load on CPU
        map_location = lambda storage, loc: storage
    else:
        # Models save on GPU load on GPU
        map_location = None
    
    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    checkpoint = torch.load(fname, map_location)
    models['p1'].load_state_dict(checkpoint['p1_model'])
    policies['p1'].load_state_dict(checkpoint['p1_policy'])