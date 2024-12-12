import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np


gym.register_envs(gymnasium_robotics)

env = gym.make('Hopper-v5', render_mode='human')
env.reset()

qp = np.array([0.00, 1.53424147e+00, -1.61449997e-02, -1.00087732e-01, 3.35373329e-03, 7.88997277e-01])  #qp[0] is x position, excluded by default. check hopper class defn
qv = np.array([2.55663472e+00, -2.16262835e+00, -7.46945956e-02, 6.72141206e-01, 4.88956399e-04, -2.71774416e-05])

actions = np.load("data/period.npy")

for i in range(1):
  obs, _ = env.reset()
  env.unwrapped.set_state(qp, qv)
  term, trunc = False, False
  step = 0
  while True:
      action = actions[step%len(actions)]
      obs, rewards, term, trunc, info = env.step(action)
      step += 1
      time.sleep(0.05)
  time.sleep(1)

env.close()

