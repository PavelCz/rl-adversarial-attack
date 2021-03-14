import time

from tqdm import tqdm

from src.attacks.fgsm import fgsm_attack_sb3, perturbed_vector_observation


def evaluate(model, env, num_eps: int, slowness=0.05, render=False, save_perturbed_img=False, attack=None,
             img_obs=False, return_infos=False):
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
    env.set_opponent_right_side(True)
    total_reward = 0
    total_rounds = 0
    total_steps = 0
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
            action, _states = model.predict(obs, deterministic=True)
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