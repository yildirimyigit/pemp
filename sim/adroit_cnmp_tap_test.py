import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
import yaml
import torch

import adroit_hand_hammer_tap

import sys

folder_path = '../models/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

folder_path = '../data/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

from cnmp import CNMP
from positional_encoders import *


env = gym.make('AdroitHandHammer-vPEMPTap', render_mode='human')

out_folder = '/home/yigit/projects/pemp/outputs/sim/adroit/bare_pe/1780764441_nail_s1/'
model_path = out_folder + 'saved_models/'

# load y_test and g_test
y_test = torch.load(out_folder + 'y_test.pt', map_location=torch.device("cpu"))
g_test = torch.load(out_folder + 'g_test.pt', map_location=torch.device("cpu"))

# load out_folder/hyperparameters.yaml
with open(out_folder + 'hyperparameters.yaml', 'r') as f:
    hyperparameters = yaml.safe_load(f)

device = 'cpu'
t_steps = hyperparameters['t_steps']  # max length of demos
batch_size, n_max, m_max = 1, 1, t_steps
dx, dy, dg, dph, dpe = 1, y_test.shape[-1], g_test.shape[-1], 0, 27
model = CNMP(input_dim=dx+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=hyperparameters['enc_dims'], decoder_hidden_dims=hyperparameters['dec_dims'], batch_size=1, device=device)
model.load_state_dict(torch.load(model_path + 'bare.pt', map_location='cpu', weights_only=False))



test_index = 3

g = g_test[test_index]  # shape (dg,)

obs = torch.zeros((batch_size, n_max, dx+dg+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dx+dg), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)

obs_mask.fill_(False)
obs_mask[0, 0] = True


num_test = 50

nail_poses = np.zeros(num_test)
with torch.no_grad():
  times = 0
  while times<1:
    observation, _ = env.reset()
    term, trunc = False, False
    
    t = 0

    obs[0, 0, :dx] = 0  # x[0] is 0
    obs[0, 0, dx:dx+dg] = g  # constant
    obs[0, 0, dx+dg:] = torch.tensor(y_test[test_index, t])  # initial observation
    
    tar_x[0, :, dx:] = g  # constant
    tar_x[0, :, :dx] = torch.linspace(0, 1, t_steps).view(-1, 1) # time encoding as x
    pred = model(obs, tar_x, obs_mask)
    actions = pred[0, :, :model.output_dim]

    for lu in range(1):
      t = 0
      while t < t_steps:
        action = np.clip(actions[t].numpy(), env.action_space.low, env.action_space.high)
        t += 1
        observation, rewards, term, trunc, info = env.step(action)
        time.sleep(0.05)
      time.sleep(3)

    nail_poses[times] = observation[26]
    times += 1

  env.close()

np.save('nail_poses_cnmp.npy', nail_poses)