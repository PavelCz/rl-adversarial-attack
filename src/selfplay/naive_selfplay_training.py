import os
import copy
from pathlib import Path

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.dqn import MlpPolicy

from src.agents.simple_rule_based_agent import SimpleRuleBasedAgent
from src.attacks.opponent_pred_as_obs_wrapper import OpponentPredictionObs
from src.common.AdversarialTrainingWrapper import AdversarialTrainingWrapper
from src.common.image_wrapper import ObservationVectorToImage
from src.common.opponent_wrapper import ObserveOpponent
from src.common.reward_wrapper import RewardZeroToNegativeBiAgentWrapper
from src.policies.mc_dropout import MCDropout
from src.selfplay.ma_gym_compatibility_wrapper import MAGymCompatibilityWrapper

from ma_gym.envs.pong_duel import pong_duel

from src.selfplay.naive_selfplay_evaluation import evaluate, evaluate_against_predecessors

best_models = []


def learn_with_selfplay(max_agents,
                        num_learn_steps,
                        num_learn_steps_pre_training,
                        num_eval_eps,
                        num_skip_steps=0,
                        model_name='dqn',
                        only_rule_based_op=False,
                        patience=5,
                        image_observations=True,
                        output_folder="output",
                        fine_tune_on=None,
                        opponent_pred_obs=False,
                        adversarial_training=None,
                        save_freq=None,
                        mc_dropout=False):
    """
    Train an agent with regular self-play. If there are checkpoints of previous training continue training with the checkpoints.

    :param max_agents: Stop after max_agents intermediate agents have been trained. An intermediate agent is saved when training
    successfully created an improved agent.
    :param num_learn_steps: Number of frames / steps for every learning iteration
    :param num_learn_steps_pre_training: Number of frames / steps for pre-training on the rule-based agent
    :param num_eval_eps: Number of episodes for intermediate evaluation. Intermediate evaluation determines whether trained agent improved
    compared to previous version
    :param num_skip_steps: Skip num_skip_steps frames performing the action from the previous step
    :param model_name: Name for saving the model. If there are already checkpoints with this name training is continued. Checkpoints will be
    saved as madel_namei, where i is the training iteration.
    :param only_rule_based_op: If set to true training is only performed on the rule-based agent.
    :param patience: Patience parameter for evaluation
    :param image_observations: Use image instead of feature observations
    :param output_folder: Root folder for outputs
    :param fine_tune_on: If not None instead of self-play training perform training of an adversarial policy against the victim specified as
    string to this parameter
    :param opponent_pred_obs:
        If this is set to True, the predictions of the opponents in the current state will beconcatenated to the observations for the main
        agent. This was an attempt to create a stronger adversarial policy, which could use this information, however in our experiments
        this didn't improve the adversarial policy
    :param adversarial_training: If set to True perform adversarial training using FGSM during training.
    :param save_freq: If not None save intermediate checkpoints during training with the given frequency
    :return:
    """
    eval_env, eval_env_rule_based, eval_op, train_env, train_env_rule_based = _init_envs(image_observations,
                                                                                         num_skip_steps,
                                                                                         opponent_pred_obs,
                                                                                         adversarial_training)

    # If fine tuning, load model to fine-tune from path
    if fine_tune_on is not None:
        path = Path(output_folder) / 'models' / fine_tune_on
        fine_tune_model = DQN.load(path)
        fine_tune_model.tensorboard_log = None
        if opponent_pred_obs:
            # We can't eval on agents that don't have a q_net so we change eval_op to the original model that is being
            # fine-tuned against, instead of the rule-based agent
            eval_op = fine_tune_model
            eval_env_rule_based.set_opponent(eval_op)
            eval_env_rule_based = OpponentPredictionObs(eval_env_rule_based)
            eval_env.set_opponent(eval_op)
            eval_env = OpponentPredictionObs(eval_env)
    else:
        fine_tune_model = None

    # Initialize first agent
    pre_train_agent = SimpleRuleBasedAgent(train_env_rule_based)
    previous_models = [pre_train_agent]

    if mc_dropout:
        # Set policy
        policy = MCDropout
    else:
        policy = MlpPolicy

    # Load potentially saved previous models
    for opponent_id in range(1, max_agents):
        path = _make_model_path(output_folder, model_name, opponent_id)
        if os.path.isfile(path):
            model = DQN.load(path, policy=policy)
            previous_models.append(model)
        else:
            break

    # Initialize first round
    last_agent_id = len(previous_models) - 1
    prev_num_steps = 0
    patience_counter = 0
    tb_path = Path(output_folder) / "tb-log"
    if last_agent_id == 0:
        # main_model = A2C('MlpPolicy', policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)), env=train_env, verbose=0,
        #                 tensorboard_log="output/tb-log")
        # main_model = A2C('MlpPolicy', train_env, verbose=0, tensorboard_log="output/tb-log")  # , exploration_fraction=0.3)
        main_model = DQN(policy, train_env_rule_based, verbose=0, tensorboard_log=tb_path)  # , exploration_fraction=0.3)
    else:
        main_model = copy.deepcopy(previous_models[last_agent_id])
        main_model.set_env(train_env)
        main_model.tensorboard_log = tb_path

    # Start training with self-play over several rounds
    opponent_id = last_agent_id
    while opponent_id < max_agents - 1:
        print(f"Running training round {opponent_id + 1}")
        if fine_tune_on is None:
            # Choose opponent based on setting
            if only_rule_based_op:
                current_train_env = train_env_rule_based
                # Use rule-based as opponent
                current_train_env.set_opponent(SimpleRuleBasedAgent(current_train_env))
            else:
                if opponent_id == 0:
                    current_train_env = train_env_rule_based
                else:
                    current_train_env = train_env
                # Take opponent from the previous version of the model
                current_train_env.set_opponent(previous_models[opponent_id])
        else:  # Use passed fine-tune agent as opponent
            current_train_env = train_env
            current_train_env.set_opponent(fine_tune_model)

        # Train the model
        current_train_env.set_opponent_right_side(True)

        chosen_n_steps = num_learn_steps_pre_training if opponent_id == 0 else num_learn_steps  # Iteration 0 is pre-training

        # In order to generate adversarial examples the adversarial training wrapper needs a references to the model that is
        # currently being trained
        if adversarial_training is not None:
            current_train_env.env.victim_model = main_model

        # Optionally add a callback to save intermediate checkpoints
        if save_freq is not None:
            checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                     save_path='./output/intermediate/',
                                                     name_prefix=model_name + str(opponent_id + 1) + '_interm')
        else:
            checkpoint_callback = None

        # === LEARNING ===
        main_model.learn(total_timesteps=chosen_n_steps, tb_log_name=model_name, callback=checkpoint_callback)

        # Do evaluation for this training round
        eval_env_rule_based.set_opponent(eval_op)
        avg_round_reward, num_steps = evaluate(main_model, eval_env_rule_based, num_eps=num_eval_eps)
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
            main_model.save(_make_model_path(output_folder, model_name, opponent_id + 1))
            # Make a copy of the just saved model by loading it
            copy_of_model = DQN.load(_make_model_path(output_folder, model_name, opponent_id + 1), policy=policy)
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
            main_model = DQN.load(_make_model_path(output_folder, model_name, opponent_id))
            main_model.set_env(train_env)

            # Opponent does not change

    if not opponent_pred_obs:
        # Evaluate the last model against each of its previous iterations
        # evaluate_against_predecessors(previous_models, env_rule_based=eval_env_rule_based, env_normal=eval_env, num_eval_eps=num_eval_eps)
        pass  # Not useful right now


def _init_envs(image_observations, num_skip_steps, opponent_pred_obs, adversarial_training):
    """ Initialize the environments with the necessaary wrappers for training. Wrappers are determined by settings in the arguments. """
    # In order to ensure symmetry for the agent when playing on either side, change second agent to red, so both have the same color
    if image_observations:
        pong_duel.AGENT_COLORS[1] = 'red'
        # Initialize environment
        train_env = gym.make('PongDuel-v0')
        train_env = RewardZeroToNegativeBiAgentWrapper(train_env)

        train_env_rule_based = ObservationVectorToImage(train_env, 'p1')
        train_env_rule_based = MAGymCompatibilityWrapper(train_env_rule_based, num_skip_steps=num_skip_steps, image_observations='main')

        if adversarial_training is not None:
            train_env_rule_based = AdversarialTrainingWrapper(train_env_rule_based,
                                                              adversarial_probability=adversarial_training,
                                                              img_obs=image_observations)
        train_env_rule_based = Monitor(train_env_rule_based)

        train_env = ObservationVectorToImage(train_env, 'both')
        train_env = MAGymCompatibilityWrapper(train_env, num_skip_steps=num_skip_steps, image_observations='both')

        if adversarial_training is not None:
            train_env = AdversarialTrainingWrapper(train_env,
                                                   adversarial_probability=adversarial_training,
                                                   img_obs=image_observations)
        train_env = Monitor(train_env)

        eval_env_rule_based = gym.make('PongDuel-v0')
        eval_env_rule_based = ObservationVectorToImage(eval_env_rule_based, 'p1')
        eval_env_rule_based = MAGymCompatibilityWrapper(eval_env_rule_based, num_skip_steps=num_skip_steps, image_observations='main')
        eval_op = SimpleRuleBasedAgent(eval_env_rule_based)
        eval_env_rule_based.set_opponent(eval_op)

        eval_env = gym.make('PongDuel-v0')
        eval_env = ObservationVectorToImage(eval_env, 'both')
        eval_env = MAGymCompatibilityWrapper(eval_env, num_skip_steps=num_skip_steps, image_observations='both')
    else:  # Init for feature observations
        train_env = gym.make('PongDuel-v0')
        train_env = ObserveOpponent(train_env, 'both')
        train_env = RewardZeroToNegativeBiAgentWrapper(train_env)
        train_env = MAGymCompatibilityWrapper(train_env, num_skip_steps=num_skip_steps, image_observations='none')
        if opponent_pred_obs:
            train_env = OpponentPredictionObs(train_env)
        if adversarial_training is not None:
            train_env = AdversarialTrainingWrapper(train_env,
                                                   adversarial_probability=adversarial_training,
                                                   img_obs=image_observations)
        train_env = Monitor(train_env)

        eval_env = gym.make('PongDuel-v0')
        eval_env = ObserveOpponent(eval_env, 'both')
        eval_env = MAGymCompatibilityWrapper(eval_env, num_skip_steps=num_skip_steps, image_observations='none')

        # For feature observations we don't need to separate between environment for rule-based and non-rule-based agents
        train_env_rule_based = train_env
        eval_env_rule_based = eval_env
        eval_op = SimpleRuleBasedAgent(eval_env_rule_based)
        eval_env_rule_based.set_opponent(eval_op)

    return eval_env, eval_env_rule_based, eval_op, train_env, train_env_rule_based


def _make_model_path(output_path, model_name: str, i: int):
    """ Given arguments determine the ouput path for model. """
    model_dir = Path(output_path) / 'models'
    return model_dir / (model_name + str(i) + '.out')
