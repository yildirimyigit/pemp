import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
import yaml
import torch

import adroit_hand_hammer_updated

import sys

folder_path = '../models/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

folder_path = '../data/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

from cnmp import CNMP
from positional_encoders import *


env = gym.make('AdroitHandHammer-vPEMP', render_mode='human')

out_folder = '/home/yigit/projects/pemp/outputs/sim/adroit/bare_pe/1778753704/'
model_path = '/home/yigit/projects/pemp/outputs/sim/adroit/bare_pe/1778753704/saved_models/'

# load out_folder/hyperparameters.yaml
with open(out_folder + 'hyperparameters.yaml', 'r') as f:
    hyperparameters = yaml.safe_load(f)

device = 'cpu'
t_steps = hyperparameters['t_steps']  # max length of demos
batch_size, n_max, m_max = 1, 1, t_steps
dx, dy, dg, dph, dpe = 1, 26, 1, 0, 27
model = CNMP(input_dim=dpe+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=hyperparameters['enc_dims'], decoder_hidden_dims=hyperparameters['dec_dims'], batch_size=1, device=device)
model.load_state_dict(torch.load(model_path + 'pe.pt', map_location='cpu', weights_only=False))

pe = generate_positional_encoding(t_steps, dpe)
g = 1 #hyperparameters['min_freq']

obs = torch.zeros((batch_size, n_max, dpe+dg+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dpe+dg), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)

obs.fill_(0)
tar_x.fill_(0)
obs_mask.fill_(False)
obs_mask[0, 0] = True



num_test = 50

nail_poses = np.zeros(num_test)
with torch.no_grad():
  times = 0
  while times<1:
    observation, _ = env.reset()
    time.sleep(3)
    term, trunc = False, False
    
    t = 0

    obs[0, 0, :dpe] = pe[t]
    obs[0, 0, dpe:dpe+dg] = g  # constant
    obs[0, 0, dpe+dg:] = torch.tensor([1.0000, -0.7098,  1.0000, -0.9245, -1.0000,  1.0000,  1.0000,  1.0000,
         1.0000,  1.0000,  1.0000,  1.0000, -1.0000,  1.0000,  1.0000,  1.0000,
        -0.3020, -0.8480,  1.0000,  1.0000,  1.0000,  0.6940, -0.0790,  1.0000,
         1.0000,  1.0000])
    tar_x[0, 0, dpe:] = g  # constant
    tar_x[0, :, :dpe] = pe
    pred = model(obs, tar_x, obs_mask)
    actions = pred[0, :, :model.output_dim]

    for lu in range(1):
      t = 0
      while t < t_steps:
        print(actions[t].numpy())
        action = np.clip(actions[t].numpy(), env.action_space.low, env.action_space.high)
        print(action)
        t += 1
        observation, rewards, term, trunc, info = env.step(action)
        time.sleep(0.1)
      time.sleep(1)

    print(times)
    nail_poses[times] = observation[26]
    times += 1

  env.close()

np.save('nail_poses_pemp.npy', nail_poses)