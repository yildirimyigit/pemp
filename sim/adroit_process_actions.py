# %%
import gymnasium as gym
import gymnasium_robotics
import numpy as np


gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandHammer-v1', render_mode='human')

actions = np.array(np.load("adroit_actions.npy", allow_pickle=True))[:, 1:24]
num_trajectories, period, dy = actions.shape
max_freq = 6
freqs = np.random.randint(1, max_freq, num_trajectories)
max_freq = np.max(freqs)

repeated_actions = []

# %%
for i in range(num_trajectories):
    repeated_actions.append(np.repeat(actions[i], freqs[i], axis=1))

# %%
t_steps = 200

# repeated_actions is a num_trajectories x ? x dy array
# interpolate each trajectory to t_steps so we have a (num_trajectories x t_steps x dy) array
interpolated_trajectories = np.zeros((num_trajectories, t_steps, dy))
for i in range(num_trajectories):
    cur_actions = np.array(repeated_actions[i], dtype=np.float64)
    for j in range(dy):
        interpolated_trajectories[i, :, j] = np.interp(np.linspace(0, 1, t_steps), np.linspace(0, 1, cur_actions.shape[0]), cur_actions[:, j])


# %%
# print(interpolated_trajectories)
print(interpolated_trajectories.shape)
# np.save("adroit_actions_10.npy", interpolated_trajectories)
# np.save("adroit_freqs_10.npy", freqs/max_freq)


