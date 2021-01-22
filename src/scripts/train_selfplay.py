from src.selfplay.naive_selfplay_training import learn_with_selfplay


def main():
    max_agents = 4
    num_eval_eps = 100
    num_skip_steps_list = [0]
    model_type = 'dqn'
    num_learn_steps_list = [500_000]
    only_rule_based_opponent = True  # True forces play against rule_based, i.e. no self-play

    for num_learn_steps in num_learn_steps_list:
        for num_skip_steps in num_skip_steps_list:
            num_in_k = num_learn_steps / 1000
            num_name = str(int(num_in_k) if num_in_k.is_integer() else num_in_k)
            model_name = model_type + '-' + ('' if num_skip_steps == 0 else 'skip-') + num_name + 'k'

            print(f'Running training for model {model_name}')
            learn_with_selfplay(max_agents=max_agents,
                                num_learn_steps=num_learn_steps,
                                num_eval_eps=num_eval_eps,
                                num_skip_steps=num_skip_steps,
                                model_name=model_name,
                                only_rule_based_op=only_rule_based_opponent)


if __name__ == '__main__':
    main()
