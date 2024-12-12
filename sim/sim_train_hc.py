import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


gym.register_envs(gymnasium_robotics)

if __name__=="__main__":
    env = make_vec_env("HalfCheetah-v5", n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, device="cpu")
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    model.save("ppo_hc_")


# import time
# import minari
# import gymnasium as gym
# import gymnasium_robotics
# import numpy as np

# from stable_baselines3 import SAC, PPO


# gym.register_envs(gymnasium_robotics)
# env = gym.make('HalfCheetah-v5')#, render_mode='human')
# # env.reset()
# # # env.render()

# model = PPO("MlpPolicy", env, verbose=0, batch_size=1024, device='cpu')
# model.learn(total_timesteps=1_000_000, progress_bar=True)
# model.save("ppo_hc")

# # print("Training done.")

# # model = SAC(
# #     "MultiInputPolicy",
# #     env,
# #     replay_buffer_class=HerReplayBuffer,
# #     replay_buffer_kwargs=dict(
# #       n_sampled_goal=n_sampled_goal,
# #       goal_selection_strategy="future",
# #     ),
# #     verbose=1,
# #     buffer_size=int(1e6),
# #     learning_rate=1e-3,
# #     gamma=0.95,
# #     batch_size=256,
# #     policy_kwargs=dict(net_arch=[256, 256, 256]),
# # )

# # model.learn(int(1_000_000))
# # model.save("sac_hc")