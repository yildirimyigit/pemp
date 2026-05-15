import gymnasium as gym
import gymnasium_robotics

import adroit_hand_hammer_updated

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__=="__main__":
    env = gym.make('AdroitHandHammer-vPEMP')#, render_mode='human')

    model = PPO("MlpPolicy", env, verbose=0, batch_size=256, device='cpu', policy_kwargs=dict(net_arch=[512,512,512]))
    model.learn(total_timesteps=15_000, progress_bar=True)
    model.save("ppo_adroithand_hammer_3_512_15m")

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