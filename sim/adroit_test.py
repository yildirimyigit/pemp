import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

import adroit_hand_hammer_updated

if __name__=="__main__":
    env = gym.make('AdroitHandHammer-vPEMP', render_mode='human')
    obs, info = env.reset()

    for i in range(200):
        time.sleep(0.5)

    print("Observation after reset:", obs)
    print("Info after reset:", info)