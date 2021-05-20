from src.selfplay.naive_selfplay_training import learn_with_selfplay


def main():
    # Settings for training and evaluation
    max_agents = 10
    num_eval_eps = 100
    num_learn_steps = 1_000_000
    num_learn_steps_pre_training = num_learn_steps  # pre-training is done only against rule-based opponent
    patience = 20
    image_observations = False
    output_folder = "../output/ucb"
    adversarial_training = False

    model_name = 'dropout1M-remote2'

    print(f'Running training for model {model_name}')
    learn_with_selfplay(max_agents=max_agents,
                        num_learn_steps=num_learn_steps,
                        num_learn_steps_pre_training=num_learn_steps_pre_training,
                        num_eval_eps=num_eval_eps,
                        model_name=model_name,
                        patience=patience,
                        image_observations=image_observations,
                        output_folder=output_folder,
                        adversarial_training=adversarial_training)


if __name__ == '__main__':
    main()
