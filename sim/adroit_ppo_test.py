from pathlib import Path
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
from stable_baselines3 import PPO

import adroit_hand_hammer_updated

if __name__=="__main__":

    MODEL_PATH = Path(__file__).resolve().parents[0] / "ppo_adroithand_hammer_3_512_15m_"

    env = gym.make('AdroitHandHammer-vPEMP', render_mode='human')
    model = PPO.load(str(MODEL_PATH), env=env, device='cpu')

    obs, info = env.reset()

    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, term, trunc, info = env.step(action)
        time.sleep(0.03)
        print(i)

    print("Observation after reset:", obs)
    print("Info after reset:", info)