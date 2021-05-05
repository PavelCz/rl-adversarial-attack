import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.attacks.fgsm import fgsm_attack_sb3, perturbed_vector_observation


def evaluate(model, env, num_eps: int, slowness=0.05, render=False, save_perturbed_img=False, attack=None,
             img_obs=False, return_infos=False, mc_dropout=False, eval_name=""):
    """
    Evaluate a trained model
    :param model: Path to model
    :param env: Environment to evaluate in
    :param num_eps: Number of evaluation episodes. Output will be averaged over number of episodes
    :param slowness: When rendering sleep for this many seconds between each frame
    :param render: Whether to render the game
    :param save_perturbed_img: Whether to save an examples of perturbations from FGSM
    :param attack: Can be 'fgsm' or None
    :param img_obs: Whether to use image observations instead of feature observations
    :param return_infos: Whether to return infos in addition to normal output
    :return: Returns reward averaged over all rounds (each episode contains multiple rounds), the total number of steps / frames,
    additional info if return_infos==True. The info dictionary contains infos to calculate the miss probability.
    """
    if not mc_dropout:
        model.policy.eval()
    else:
        # Enable training so dropout is enabled.
        # Watch out, in case other features also depend on this setting, another way to enable dropout without setting training mode would
        # have to be found
        model.policy.train()
    timestamp = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())
    eval_writer = SummaryWriter(f'../output/ucb/tb-eval/{timestamp}{eval_name}')
    env.set_opponent_right_side(True)
    total_reward = 0
    total_rounds = 0
    total_steps = 0
    num_dropout_samples = 10
    all_variances = []
    if return_infos:
        infos = {}
    for episode in tqdm(range(num_eps), desc='Evaluating...'):
        ep_reward = 0
        # Evaluate the agent
        done = False
        obs = env.reset()
        info = None
        while not done:
            if attack == "fgsm":
                # Perturb observation
                obs = fgsm_attack_sb3(obs, model, 0.02, img_obs=img_obs)
            if render:
                time.sleep(slowness)
                if save_perturbed_img:
                    perturbed_vector_observation(env.render(mode='rgb_array'), obs)
                env.render()
            if mc_dropout:
                q_net = model.q_net
                q_net.zero_grad()  # Zero out the gradients
                obs = torch.tensor([obs])

                # Move obs to correct device
                obs = obs.to(q_net.device)

                # Sample with dropout
                q_vals = torch.zeros(1, model.action_space.n).to(q_net.device)
                q_val_list = []
                for i in range(num_dropout_samples):
                    pred = q_net(obs)
                    q_vals += pred
                    q_val_list.append(q_vals)
                means = q_vals / num_dropout_samples
                variance = torch.zeros(1, model.action_space.n).to(q_net.device)
                for q_val in q_val_list:
                    variance += (q_val - means) ** 2
                # Calculate _sample_ var as opposed to population var, therefore we only divide by n-1
                variance /= len(q_val_list) - 1
                for i in range(means.shape[1]):
                    eval_writer.add_scalar(tag=f"uncertainty/mean_{i}",
                                           scalar_value=means[0, i],
                                           global_step=total_steps)
                    eval_writer.add_scalar(tag=f"uncertainty/var_{i}",
                                           scalar_value=variance[0, i],
                                           global_step=total_steps)
                action = torch.argmax(means)
                eval_writer.add_scalar(tag=f"uncertainty/best_mean",
                                       scalar_value=means[0, action],
                                       global_step=total_steps)
                eval_writer.add_scalar(tag=f"uncertainty/best_var",
                                       scalar_value=variance[0, action],
                                       global_step=total_steps)
                all_variances.append(variance[0, action])
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_steps += 1

            # print(reward)
            ep_reward += reward
        total_reward += ep_reward
        total_rounds += info['rounds']
        if return_infos:
            for key in info:
                if key not in infos:
                    infos[key] = info[key]
                else:
                    infos[key] += info[key]
    env.close()

    if mc_dropout:
        eval_writer.add_histogram(tag=f"uncertainty/vars",
                                  values=np.array(all_variances),
                                  global_step=0)

    avg_round_reward = total_reward / total_rounds

    if return_infos:
        return avg_round_reward, total_steps, infos
    else:
        return avg_round_reward, total_steps


def evaluate_against_predecessors(previous_models, env_rule_based, env_normal, num_eval_eps):
    """ Evaluate against all predecessors in the list previous_models """
    print(f"Evaluating against predecessors...")
    last_model = previous_models[-1]
    last_model_index = len(previous_models) - 1
    for i, model in enumerate(previous_models):
        if i == 0:
            env = env_rule_based
        else:
            env = env_normal
        env.set_opponent(model)
        avg_round_reward, num_steps = evaluate(last_model, env, num_eps=num_eval_eps)
        print(f"Model {last_model_index} against {i}: {avg_round_reward}")
        print(f"Average number of steps: {num_steps / num_eval_eps}")
