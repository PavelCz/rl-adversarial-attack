from pathlib import Path

from src.selfplay.naive_selfplay_training import learn_with_selfplay


def main():
    # Settings for training and evaluation
    max_agents = 3
    num_eval_eps = 50
    num_learn_steps = 1_000_000
    num_learn_steps_pre_training = 1_000_000  # pre-training is done only against rule-based opponent
    only_rule_based_opponent = False  # True forces play against rule_based, i.e. no self-play
    patience = 20
    image_observations = False
    output_folder = "../../output"

    model_name = 'adv-retrain-adversary'

    fine_tune_on = 'gcp-feature-based-op-obs8-adv-retrain.out'

    print(f'Running training for model {model_name}')
    learn_with_selfplay(max_agents=max_agents,
                        num_learn_steps=num_learn_steps,
                        num_learn_steps_pre_training=num_learn_steps_pre_training,
                        num_eval_eps=num_eval_eps,
                        model_name=model_name,
                        only_rule_based_op=only_rule_based_opponent,
                        patience=patience,
                        image_observations=image_observations,
                        output_folder=output_folder,
                        fine_tune_on=fine_tune_on,
                        save_freq=100000)


if __name__ == '__main__':
    main()
