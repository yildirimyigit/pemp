import time
from pathlib import Path

import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(gymnasium_robotics)
import os
os.environ["MUJOCO_GL"] = "egl"

MODEL_PATH = Path(__file__).resolve().parents[1] / "ppo_adroithand_hammer_3_256_15m.zip"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find PPO model at {MODEL_PATH}")

env = gym.make('AdroitHandHammer-v1', render_mode='human')
o,i = env.reset()

model = PPO.load(str(MODEL_PATH), env=env, device='cpu')

renderer = getattr(env.unwrapped, "mujoco_renderer", None)
viewer = renderer._get_viewer(render_mode="human")
viewer._hide_menu = True

def reset():
    obs, info = env.reset()

    qp = np.array([0.05362812, -0.15676938,  0.1124559,  -0.19211253, -0.00677962,  0.94733362,
        0.68165027,  0.95151664, -0.04922544,  1.11051917,  0.84514704,  0.93555745,
        -0.07568273,  0.93582394,  0.93026161,  1.2665799,   0.14275029, -0.38195376,
        1.22411986,  1.04177562,  1.09699174,  0.10221559,  1.12782028,  0.14026537,
        -0.1667768,  -0.43379269,  0.,          0.02543423,  0.10667372, -0.01684074,
        -0.0044481,  -0.29020183,  0.25038974])
    qv = np.zeros(33)
    bp = np.array([0.05, 0., 0.18356179])
    tp = np.array([-0.04413594, -0.03659814, 0.18359105])

    init_state = dict(qpos=qp, qvel=qv, board_pos=bp)
    env.env.env.env.set_env_state(init_state)
    # settle simulator
    for _ in range(10):
        _ = env.step(np.zeros(env.action_space.shape))
    return obs, info

obs, info = reset()
input()
for i in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, term, trunc, info = env.step(np.zeros_like(action))
    time.sleep(0.03)

    if term or trunc:
        reset()
        print()

time.sleep(1)

env.close()
