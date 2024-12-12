import time
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


gym.register_envs(gymnasium_robotics)

env = gym.make('HalfCheetah-v5')#, exclude_current_positions_from_observation=False)

model = PPO.load("ppo_hc", env=env, device='cpu')

observations, actions = [], []

for i in range(1000):
  print(i)
  current_observations, current_actions = [], []
  obs, _ = env.reset()
  # env.unwrapped.set_state(qp, qv)
  # obs = env.unwrapped._get_obs()
  current_observations.append(obs)
  term, trunc = False, False
  while not (term or trunc):
      action, _states = model.predict(obs, deterministic=True)
      current_actions.append(action)
      obs, rewards, term, trunc, info = env.step(action)
      current_observations.append(obs)
      # time.sleep(0.005)

  actions.append(current_actions)
  observations.append(current_observations[:-1])
  time.sleep(0.1)

env.close()

actions = np.array(actions, dtype=object)
np.save("hc_actions_1k.npy", actions)

observations = np.array(observations, dtype=object)
np.save("hc_observations_1k.npy", observations)