from src.selfplay.naive_selfplay_training import learn_with_selfplay


def main():
    max_agents = 2
    num_eval_eps = 1000
    num_skip_steps_list = [0, 2]
    model_type = 'dqn'
    num_learn_steps_list = [2_000, 20_000, 300_000, 100_000, 750_000, 1_500_000]

    for learn_steps in num_learn_steps_list:
        for num_skip_steps in num_skip_steps_list:
            model_name = model_type + '-' + ('' if num_skip_steps == 0 else 'skip-') + str(num_skip_steps / 1000) + 'k-'

            print(f'Running training for model {model_name}')
            learn_with_selfplay(max_agents=max_agents,
                                num_learn_steps=learn_steps,
                                num_eval_eps=num_eval_eps,
                                num_skip_steps=num_skip_steps,
                                model_name=model_name)


if __name__ == '__main__':
    main()
