import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np


gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandHammer-v1', render_mode='human')

actions = np.array(np.load("adroit_actions.npy", allow_pickle=True))[:, 1:24]
num_trajectories = actions.shape[0]
freqs = np.random.randint(1, 6, num_trajectories)
processed_actions = []

for i in range(num_trajectories):
    actions[i] = actions[i].astype(np.float32)

obs, _ = env.reset()
# env.start_recording('test')

for i in range(5):
    j = 0
    while j < 400:
        action = actions[j%23]
        if j == 23:
            print(env.env.env.env.get_env_state())
        obs, rewards, term, trunc, info = env.step(action)

        time.sleep(0.05)

        j += 1
    # if i == 23:
    #     print(env.env.env.env.get_env_state())

    env.reset()
    # env.stop_recording()

time.sleep(1)

# env.close()










# for i in range(10):
#     print(f'************************** Episode {i} **************************')
#     obss = []
#     obss.append(obs[33:42])

#     min_dist = 1000
#     for j in range(200):
#         action = actions[i][j, :]
#         obs, rewards, term, trunc, info = env.step(action)

#         obss.append(obs[33:42])
#         if j > 20:
#             for ind in range(j-10, 0, -1):
#                 dist = np.linalg.norm(obss[ind] - obs[33:42])
#                 if dist < min_dist:
#                     min_dist = dist
#                     print(j, ind)

#         time.sleep(0.01)
#     # if i == 23:
#     #     print(env.env.env.env.get_env_state())

#     env.reset()

# time.sleep(1)

# env.close()