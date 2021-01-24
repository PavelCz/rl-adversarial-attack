import os
import copy
import time

import gym
import ma_gym  # Necessary so the PongDuel env exists
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from src.agents.random_agent import RandomAgent
from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper

best_models = []


def learn_with_selfplay(max_agents,
                        num_learn_steps,
                        num_learn_steps_pre_training,
                        num_eval_eps,
                        num_skip_steps=0,
                        model_name='dqn',
                        only_rule_based_op=False,
                        patience=5):
    # Initialize environment
    train_env = gym.make('PongDuel-v0')
    train_env = RewardZeroToNegativeBiAgentWrapper(train_env)
    train_env = MAGymCompatibilityWrapper(train_env, num_skip_steps=num_skip_steps)
    train_env = Monitor(train_env)

    eval_env = gym.make('PongDuel-v0')
    eval_env = MAGymCompatibilityWrapper(eval_env, num_skip_steps=num_skip_steps)
    eval_op = SimpleRuleBasedAgent(eval_env)
    eval_env.set_opponent(eval_op)

    # Initialize first agent
    pre_train_agent = SimpleRuleBasedAgent(train_env)
    previous_models = [pre_train_agent]

    # Load potentially saved previous models
    for opponent_id in range(1, max_agents):
        path = _make_model_path(model_name, opponent_id)
        if os.path.isfile(path):
            model = DQN.load(path)
            previous_models.append(model)
        else:
            break

    # Initialize first round
    last_agent_id = len(previous_models) - 1
    prev_num_steps = 0
    patience_counter = 0
    if last_agent_id == 0:
        # main_model = A2C('MlpPolicy', policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)), env=train_env, verbose=0,
        #                 tensorboard_log="output/tb-log")
        # main_model = A2C('MlpPolicy', train_env, verbose=0, tensorboard_log="output/tb-log")  # , exploration_fraction=0.3)
        main_model = DQN('MlpPolicy', train_env, verbose=0, tensorboard_log="output/tb-log")  # , exploration_fraction=0.3)
    else:
        main_model = copy.deepcopy(previous_models[last_agent_id])
        main_model.set_env(train_env)
        main_model.tensorboard_log = "output/tb-log"

    # Start training with self-play over several rounds
    opponent_id = last_agent_id
    while opponent_id < max_agents - 1:
        print(f"Running training round {opponent_id + 1}")

        # Choose opponent based on setting
        if only_rule_based_op:
            # Use rule-based as opponent
            train_env.set_opponent(SimpleRuleBasedAgent(train_env))
        else:
            # Take opponent from the previous version of the model
            train_env.set_opponent(previous_models[opponent_id])

        # Train the model
        train_env.set_opponent_right_side(True)

        chosen_n_steps = num_learn_steps_pre_training if opponent_id == 0 else num_learn_steps  # Iteration 0 is pre-training
        main_model.learn(total_timesteps=chosen_n_steps, tb_log_name=model_name)  # , callback=learn_callback)

        # Do evaluation for this training round
        eval_env.set_opponent(eval_op)
        avg_round_reward, num_steps = evaluate(main_model, eval_env, num_eps=num_eval_eps)
        print(model_name)
        print(f"Average round reward after training: {avg_round_reward}")
        print(f"Average number of steps per episode: {num_steps / num_eval_eps}")

        # Check if there was improvement
        if num_steps > prev_num_steps:  # Model improved compared to last
            print('Model improved')
            prev_num_steps = num_steps
            # Reset patience counter
            patience_counter = 0

            # Save the further trained model to disk
            main_model.save(_make_model_path(model_name, opponent_id + 1))
            # Make a copy of the just saved model by loading it
            copy_of_model = DQN.load(_make_model_path(model_name, opponent_id + 1))
            # Save the copy to the list
            previous_models.append(copy_of_model)

            # From here we continue training the same main_model against itself
            opponent_id += 1
        else:
            print('Model did not improve')
            patience_counter += 1
            # Do not save the model
            if patience_counter > patience:
                print('Stopping early due to patience')
                break
            # Because our model did not improve compared to the previous one, we reset our main_model to the previous one
            main_model = DQN.load(_make_model_path(model_name, opponent_id))
            main_model.set_env(train_env)

            # Opponent does not change

    # Evaluate the last model against each of its previous iterations
    _evaluate_against_predecessors(previous_models, eval_env, num_eval_eps)


def _make_model_path(model_name: str, i: int):
    model_dir = 'output/models/'
    return model_dir + model_name + str(i) + '.out'


def evaluate(model, env, num_eps, slowness=0.1, render=False, print_obs=False, verbose=False):
    env.set_opponent_right_side(True)
    total_reward = 0
    total_rounds = 0
    total_steps = 0
    for episode in range(num_eps):
        ep_reward = 0
        # Evaluate the agent
        done = False
        obs = env.reset()
        info = None
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            if verbose:
                print(action)
            obs, reward, done, info = env.step(action)
            total_steps += 1

            # print(reward)
            ep_reward += reward
            if render:
                time.sleep(slowness)
                env.render()
            if print_obs:
                print('\r', *obs, end="")
        total_reward += ep_reward
        total_rounds += info['rounds']
    env.close()

    avg_round_reward = total_reward / total_rounds
    return avg_round_reward, total_steps


def _evaluate_against_predecessors(previous_models, env, num_eval_eps):
    last_model = previous_models[-1]
    last_model_index = len(previous_models) - 1
    for i, model in enumerate(previous_models):
        env.set_opponent(model)
        avg_round_reward, num_steps = evaluate(last_model, env, num_eps=num_eval_eps)
        print(f"Model {last_model_index} against {i}: {avg_round_reward}")
        print(f"Average number of steps: {num_steps / num_eval_eps}")
