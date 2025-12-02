import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


gym.register_envs(gymnasium_robotics)

# dataset = minari.load_dataset("D4RL/hammer/cloned-v2", download=True)#.filter_episodes(lambda episode: episode.rewards.mean() > 70)
# print("Total episodes:", dataset.total_episodes)
# print("Total steps:", dataset.total_steps)

# env = dataset.recover_environment(render_mode='human')
# env = gym.make('AdroitHandHammer-v1', render_mode='human')
if __name__=="__main__":
    env = gym.make('AdroitHandHammer-v1')#, render_mode='human')

    # env = make_vec_env('AdroitHandHammer-v1', n_envs=8, vec_env_cls=SubprocVecEnv)
    # env.reset()
    # env.render()

    # qp = np.array([0.05362812, -0.15676938,  0.1124559,  -0.19211253, -0.00677962,  0.94733362,
    #   0.68165027,  0.95151664, -0.04922544,  1.11051917,  0.84514704,  0.93555745,
    #  -0.07568273,  0.93582394,  0.93026161,  1.2665799,   0.14275029, -0.38195376,
    #   1.22411986,  1.04177562,  1.09699174,  0.10221559,  1.12782028,  0.14026537,
    #  -0.1667768,  -0.43379269,  0.,          0.02543423,  0.10667372, -0.01684074,
    #  -0.0044481,  -0.29020183,  0.25038974])
    # qv = np.zeros(33)
    # bp = np.array([0.05, 0., 0.18356179])
    # tp = np.array([-0.04413594, -0.03659814, 0.18359105])

    # init_state = dict(qpos=qp, qvel=qv, board_pos=bp)
    # env.env.env.env.set_env_state(init_state)

    model = PPO("MlpPolicy", env, verbose=0, batch_size=256, device='cpu', policy_kwargs=dict(net_arch=[512,512,512]))
    model.learn(total_timesteps=15_000_000, progress_bar=True)
    model.save("ppo_adroithand_hammer_3_256_15m")

    print("Training done.")

# for j in range(dataset.total_episodes):
#     env.env.env.env.set_env_state(init_state)
#     actions = dataset[j].actions
    
#     for i in range(200):
#         # init_state = env.env.env.env.get_env_state()
#         # print(i)
#         # print(init_state['qpos'])
#         # print(init_state['qvel'])
#         # print(init_state['board_pos'])
#         # print(init_state['target_pos'])
#         obs, rew, terminated, truncated, info = env.step(actions[i])
#         if terminated or truncated:
#             env.reset()
#         time.sleep(0.2)
#     env.reset()
#     time.sleep(1)

# env.close()