import gym
import ma_gym
import copy

from stable_baselines3 import PPO

from src.selfplay.opponent_wrapper import OpponentWrapper

# Initialize environment
env = gym.make('PongDuel-v0')
env = OpponentWrapper(env)

main_model = PPO('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=10000)

secondary_model = copy.deepcopy(main_model)

env.set_opponent(secondary_model)

main_model.set_env(env)

ep_reward = 0
# Evaluate the agent
done = False
obs = env.reset()
while not done:
    action, _states = main_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    ep_reward += reward
    env.render()
env.close()
print(ep_reward)
