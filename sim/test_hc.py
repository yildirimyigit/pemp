import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


gym.register_envs(gymnasium_robotics)

env = gym.make('HalfCheetah-v5', render_mode='human')
env.reset()

model = PPO.load("ppo_hc_", env=env, device='cpu')

qp = np.array([0.00, 1.16241491e+00, 7.88451918e-02, -4.59125719e-01, -1.81820100e-02, -1.49593736e-01])  #qp[0] is x position, excluded by default. check hopper class defn
qv = np.array([2.87365054e+00, 1.45815623e+00, 1.72433129e-02, -8.73125015e-03, -1.88852219e+00, -9.33476774e+00])

for i in range(1):
  obs, _ = env.reset()
  # env.unwrapped.set_state(qp, qv)
  term, trunc = False, False
  stop_step = 303
  step = 0
  while not (term or trunc):
      action, _states = model.predict(obs, deterministic=True)
      # if step == stop_step-1:
      #    print(obs)
      obs, rewards, term, trunc, info = env.step(action)
      time.sleep(0.1)
      # if step == stop_step:
      #    print(action)
      #    print(info)
      #    time.sleep(100)
      # print(step, action)
      # step += 1
      # break
  time.sleep(1)

env.close()

