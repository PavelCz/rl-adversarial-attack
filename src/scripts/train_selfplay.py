from src.selfplay.learning import learn_with_selfplay


def main():
    learn_with_selfplay(max_agents=2, num_learn_steps=500_000, num_eval_eps=10, num_skip_steps=0, model_name='dqn-vs-rule-based-500k-')


if __name__ == '__main__':
    main()
