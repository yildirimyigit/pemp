import time
import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import PPO


gym.register_envs(gymnasium_robotics)
env = gym.make('Hopper-v5')#, render_mode='human')
env.reset()
# env.render()

model = PPO("MlpPolicy", env, verbose=1, batch_size=128, device='cpu')
model.learn(total_timesteps=10_000_000, progress_bar=True)
model.save("ppo_hopper_")

# print("Training done.")

# # for j in range(dataset.total_episodes):
# #     env.env.env.env.set_env_state(init_state)
# #     actions = dataset[j].actions
    
# for i in range(200):
#     obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
#     if terminated or truncated:
#         env.reset()
#     time.sleep(0.02)
# env.reset()
# time.sleep(1)

# env.close()