from src.selfplay.learning import learn_with_selfplay


def main():
    learn_with_selfplay(max_agents=20, num_learn_steps=1_00_000, num_eval_eps=10)


if __name__ == '__main__':
    main()
