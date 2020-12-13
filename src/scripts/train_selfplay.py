import gym
import ma_gym
import copy

from stable_baselines3 import PPO

from src.selfplay.opponent_wrapper import OpponentWrapper


def main():
    # Initialize environment
    env = gym.make('PongDuel-v0')
    env = OpponentWrapper(env)

    main_model = PPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=10000)

    secondary_model = copy.deepcopy(main_model)

    env.set_opponent(secondary_model)

    # main_model.set_env(env)

    avg_round_reward = evaluate(main_model, env)
    print(f"Average round reward: {avg_round_reward}")


def evaluate(model, env):
    num_eps = 100
    total_reward = 0
    total_rounds = 0
    for episode in range(num_eps):
        ep_reward = 0
        # Evaluate the agent
        done = False
        obs = env.reset()
        info = None
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            # env.render()
        total_reward += ep_reward
        total_rounds += info['rounds']
    env.close()

    avg_round_reward = total_reward / total_rounds
    return avg_round_reward


if __name__ == '__main__':
    main()
