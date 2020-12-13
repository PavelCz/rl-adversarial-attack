from src.selfplay.learning import learn_with_selfplay


def main():
    learn_with_selfplay(max_agents=10, num_learn_steps=1_000_000, num_eval_eps=100)


if __name__ == '__main__':
    main()
