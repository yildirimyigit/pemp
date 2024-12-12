import time
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import sys

folder_path = '../models/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

folder_path = '../data/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

from cnmp import CNMP
from positional_encoders import *

gym.register_envs(gymnasium_robotics)

env = gym.make('Hopper-v5', render_mode='human')

# viewer = env.unwrapped.viewer
# viewer.cam.distance = 5.0

qp = np.array([0.00, 1.40937645e+00, -6.04250112e-04, -8.05713632e-02, 3.36481823e-03, 7.88997210e-01])  #qp[0] is x position, excluded by default. check hopper class defn
qv = np.array([2.51384158e+00, -2.31315405e+00, -1.39180762e-01, 8.47442020e-01, 3.59642776e-04, -1.92279412e-05])

device = 'cpu'
batch_size, n_max, m_max = 1, 40, 40
dx, dy, dg, dph, dpe = 1, 3, 1, 0, 27
t_steps = 400
# cnmp = CNMP(dx, dy, 20, 20, [128,128], decoder_hidden_dims=[128,128], batch_size=1, device=device)
model = CNMP(input_dim=dpe+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=[128,128], decoder_hidden_dims=[128,128], batch_size=1, device=device)

model_path = '/home/yigit/projects/pemp/outputs/sim/hopper/bare_pe/1733139349/saved_models/'
# cnmp.load_state_dict(torch.load(model_path + 'bare.pt', map_location='cpu'))
model.load_state_dict(torch.load(model_path + 'pe.pt', map_location='cpu'))

pe = generate_positional_encoding(t_steps, dpe)
g = 1.0 / 4  # 4 is max freq in demos
obs = torch.zeros((batch_size, n_max, dpe+dg+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dpe+dg), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)

obs.fill_(0)
tar_x.fill_(0)
obs_mask.fill_(False)
obs_mask[0, 0] = True

cont = 'y'
with torch.no_grad():
  time.sleep(1)
  while cont == 'y':
    observation, _ = env.reset()
    env.unwrapped.set_state(qp, qv)
    term, trunc = False, False

    obs[0, 0, dpe:dpe+dg] = g  # constant
    obs[0, 0, dpe+dg:] = torch.tensor([-0.02574568, 1., 1.])  # initial action (conditioning point)
    tar_x[0, 0, dpe:] = g  # constant
    
    step = 0
    while not (trunc):
      t = step%t_steps
      obs[0, 0, :dpe] = pe[t]
      tar_x[0, 0, :dpe] = pe[(t+1)%t_steps]
    
      pred = model(obs, tar_x, obs_mask)
      # clip action to env.action_space.low and env.action_space.high
      action = np.clip((pred[0, 0, :model.output_dim]).numpy(), env.action_space.low, env.action_space.high)

      observation, rewards, term, trunc, info = env.step(action)

      obs[0, 0, dpe+dg:] = torch.from_numpy(action)
      time.sleep(0.02)
      step += 1
      # print(step, action)
    cont = 'y'#input('Start over? (y/n)\n')
    time.sleep(1)

  env.close()



# import time
# import minari
# import gymnasium as gym
# import gymnasium_robotics
# import numpy as np

# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy


# gym.register_envs(gymnasium_robotics)

# env = gym.make('Hopper-v5', render_mode='human')
# env.reset()

# model = PPO.load("ppo_hopper", env=env, device='cpu')

# qp = np.array([0.00, 1.16241491e+00, 7.88451918e-02, -4.59125719e-01, -1.81820100e-02, -1.49593736e-01])  #qp[0] is x position, excluded by default. check hopper class defn
# qv = np.array([2.87365054e+00, 1.45815623e+00, 1.72433129e-02, -8.73125015e-03, -1.88852219e+00, -9.33476774e+00])

# for i in range(1):
#   obs, _ = env.reset()
#   env.unwrapped.set_state(qp, qv)
#   term, trunc = False, False
#   # step = 0
#   while not (term or trunc):
#       action, _states = model.predict(obs, deterministic=True)
#       time.sleep(10)
#       print(action)
#       break
#       obs, rewards, term, trunc, info = env.step(action)
#       time.sleep(0.01)
#       # if step == 135:
#       #    print(obs)
#       #    print(info)
#       #    time.sleep(100)
#       # step += 1
#   time.sleep(1)

# env.close()

