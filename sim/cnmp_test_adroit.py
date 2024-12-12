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

env = gym.make('AdroitHandHammer-v1', render_mode='human')

# viewer = env.unwrapped.viewer
# viewer.cam.distance = 5.0

device = 'cpu'
t_steps = 100
batch_size, n_max, m_max = 1, 1, t_steps
dx, dy, dg, dph = 1, 26, 1, 0
model = CNMP(input_dim=dx+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=[512,512], decoder_hidden_dims=[512,512], batch_size=1, device=device)

model_path = '/home/yigit/projects/pemp/outputs/sim/adroit/bare_pe/1733759916/saved_models/'
model.load_state_dict(torch.load(model_path + 'bare.pt', map_location='cpu'))

g = 1.0 # 4 is max freq in demos
obs = torch.zeros((batch_size, n_max, dx+dg+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dx+dg), dtype=torch.float32, device=device)
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
    term, trunc = False, False
    
    t = 0

    obs[0, 0, :dx] = 0
    obs[0, 0, dx:dx+dg] = g  # constant
    obs[0, 0, dx+dg:] = torch.tensor([0.48420828580856323, -1.0 ,-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
 0.3672761023044586, 0.6314694881439209, -1.0, 1.0, 0.11585365235805511, 1.0,
 -0.9243438243865967, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.8393586874008179,
 0.024456113576889038])  # initial action (conditioning point)
    tar_x[0, :, :dx] = torch.linspace(0, 1, t_steps).unsqueeze(1)
    tar_x[0, :, dx:] = g  # constant
    pred = model(obs, tar_x, obs_mask)
    actions = pred[0, :, :model.output_dim]

    for lu in range(1):
      t = 0
      while t < t_steps:
        action = np.clip(actions[t].numpy(), env.action_space.low, env.action_space.high)
        t += 1
        observation, rewards, term, trunc, info = env.step(action)
        time.sleep(0.04)
      time.sleep(1)

    # step = 0
    # while not (trunc):
    #   t = step%t_steps
    #   obs[0, 0, :dpe] = pe[t]
    #   tar_x[0, 0, :dpe] = pe[(t+1)%t_steps]
    
    #   pred = model(obs, tar_x, obs_mask)
    #   # clip action to env.action_space.low and env.action_space.high
    #   action = np.clip((pred[0, 0, :model.output_dim]).numpy(), env.action_space.low, env.action_space.high)

    #   observation, rewards, term, trunc, info = env.step(action)

    #   obs[0, 0, dpe+dg:] = torch.from_numpy(action)
    #   time.sleep(0.01)
    #   step += 1
    #   # print(step, action)
    cont = input('Start over? (y/n)\n')
    # time.sleep(1)

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

