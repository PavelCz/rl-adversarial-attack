from src.selfplay.naive_selfplay_training import learn_with_selfplay


def main():
    max_agents = 2
    num_eval_eps = 100
    num_skip_steps_list = [0]
    model_type = 'dqn'
    num_learn_steps_list = [750_000]

    for num_learn_steps in num_learn_steps_list:
        for num_skip_steps in num_skip_steps_list:
            model_name = model_type + '-' + ('' if num_skip_steps == 0 else 'skip-') + str(num_learn_steps / 1000) + 'k-'

            print(f'Running training for model {model_name}')
            learn_with_selfplay(max_agents=max_agents,
                                num_learn_steps=num_learn_steps,
                                num_eval_eps=num_eval_eps,
                                num_skip_steps=num_skip_steps,
                                model_name=model_name)


if __name__ == '__main__':
    main()
