import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandHammer-v1', render_mode='human')
env.reset()

model = PPO.load("ppo_adroithand_hammer_", env=env, device='cpu')

obs, _ = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, term, trunc, info = env.step(action)
    print(obs[42:45])
    time.sleep(0.05)
    # if i == 23:
    #     print(env.env.env.env.get_env_state())

    if term or trunc:
        env.reset()
        print('\n')

time.sleep(1)

env.close()