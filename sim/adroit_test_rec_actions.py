import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np


gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandHammer-v1', render_mode='human')
# env = gym.wrappers.RecordVideo(env=tmp_env, video_folder="/home/yigit/Desktop/tmp/", name_prefix="periodic_test", episode_trigger=lambda x: x % 2 == 0)

actions = np.load("adroit_actions.npy", allow_pickle=True)[0][1:24]

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