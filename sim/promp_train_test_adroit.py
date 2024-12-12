import time
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gmr import GMM
from movement_primitives.promp import ProMP

gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandHammer-v1', render_mode='human')

dx, dy, dg, dph, dpe = 1, 26, 1, 0, 27
n_weights_per_dim = 30

# data
Y = np.load("data/adroit_actions_10.npy", allow_pickle=True)  # y: (num_traj, t_steps, dy)
freqs = np.load("data/adroit_freqs_10.npy", allow_pickle=True)  # freqs: (num_traj,)
max_freq = np.max(freqs)
freqs = (freqs/max_freq)[:, np.newaxis]  # normalize freqs and newaxis to match Y

num_traj, t_steps, dy = Y.shape
dg = freqs.shape[-1]

x = np.linspace(0, 1, t_steps)
x = np.concatenate([x[np.newaxis, ...]]*num_traj, axis=0)  # x: (num_traj, t_steps)

promp = ProMP(n_dims=dy, n_weights_per_dim=n_weights_per_dim)
weights = np.empty((num_traj, dy * n_weights_per_dim))

for i in range(num_traj):
    T = np.linspace(0, 1, t_steps)  # Time
    weights[i] = promp.weights(T, Y[i])

X = np.hstack((freqs, weights))

# Train GMM
random_state = np.random.RandomState(0)
gmm = GMM(n_components=5, random_state=random_state)
gmm.from_samples(X)


freq_query = np.array([[0.75]])  # Example query frequency
conditional_weight_distribution = gmm.condition(np.arange(1), freq_query).to_mvn()

promp_query = ProMP(n_dims=dy, n_weights_per_dim=n_weights_per_dim)
promp_query.from_weight_distribution(conditional_weight_distribution.mean, conditional_weight_distribution.covariance)

cond_point = np.array([0.48420828580856323, -1.0 ,-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
0.3672761023044586, 0.6314694881439209, -1.0, 1.0, 0.11585365235805511, 1.0,
-0.9243438243865967, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.8393586874008179,
0.024456113576889038])

t = 0

promp_query = promp_query.condition_position(cond_point, t=t)
actions = promp_query.sample_trajectories(x[0], 1, np.random.RandomState(seed=1234))[0]
print(actions.shape)

cont = 'y'
while cont == 'y':
  observation, _ = env.reset()

  time.sleep(2)
  t = 0
  while t < t_steps:
    action = np.clip(actions[t], env.action_space.low, env.action_space.high)
    t += 1
    observation, rewards, term, trunc, info = env.step(action)
    time.sleep(0.04)
  time.sleep(1)

cont = input('Start over? (y/n)\n')

env.close()
