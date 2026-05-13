import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import PPO


gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandHammer-v1')

def is_close(a, b, tol):
    return np.all(np.abs(a - b) < tol)

model = PPO.load("ppo_adroithand_hammer_", env=env, device='cpu')


actions = []
obs, _ = env.reset()

first_pose = obs[33:42]
# print(first_pose)

min_dist = 1000000

for i in range(10):
    term, trunc = False, False
    current_actions = []
    step = 0
    print("Episode", i)
    print(f'Init vel: {obs[27:30]}')
    first_pose = obs[33:42]
    while not (term or trunc):
        action, _states = model.predict(obs, deterministic=True)
        current_actions.append(action)
        obs, rewards, term, trunc, info = env.step(action)

        if step > 30: # Skip the first few steps
            pose = obs[33:42]

            dist = np.linalg.norm(first_pose - pose)
            # if is_close(first_pose, pose, 0.01):
            print(f'vel: {obs[27:30]}')
            if dist<min_dist:
                min_dist = dist
                print(step)
                print(pose)

        time.sleep(0.01)
        step += 1
    
        if term or trunc:
            obs, _ = env.reset()
    time.sleep(1)
    actions.append(current_actions)

actions = np.array(actions, dtype=object)
np.save("adroit_actions.npy", actions)


env.close()