"""Pure-action hammer demonstrations for ``AdroitHandHammer-vPEMP``.

Unlike ``adroit_pemp_three_strike_controller.py``, this controller never edits
MuJoCo state during a demonstration.  Every saved sample is one action that was
actually sent through ``env.step(action)``, which makes the resulting files
suitable for action-only imitation learning.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np


gym.register_envs(gymnasium_robotics)

ENV_ID = "AdroitHandHammer-vPEMP"
import adroit_hand_hammer_updated

# Each row is one normalized action for AdroitHandHammer-v1:
# 0:4 are arm/wrist targets, 4:26 are finger/thumb targets.
RECORDED_ACTIONS = np.array(
    [
        [1.000, 0.353, 0.926, 1.000, 1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 0.378, 1.000, 1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -0.887, -0.080, -0.915, 1.000, 1.000, 0.600, 1.000, 1.000],
        [1.000, 0.112, 1.000, 1.000, 1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 0.589, 1.000, 1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -0.838, -0.190, -1.000, 1.000, 1.000, 0.581, 1.000, 1.000],
        [1.000, -0.254, 1.000, 1.000, 1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, 0.829, -1.000, -1.000, -1.000, -1.000, -1.000, -0.719, -0.292, -1.000, 1.000, 1.000, 0.623, 1.000, 1.000],
        [1.000, -0.641, 1.000, 1.000, 1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, 0.442, -0.866, -1.000, -1.000, -1.000, -1.000, -0.555, -0.353, -1.000, 1.000, 1.000, 0.676, 1.000, 1.000],
        [1.000, -0.985, 1.000, 1.000, 1.000, -1.000, -1.000, -1.000, -0.972, 1.000, 1.000, 1.000, -0.030, -0.610, -1.000, -0.933, -1.000, -1.000, -0.387, -0.361, -1.000, 1.000, 1.000, 0.705, 1.000, 1.000],
        [1.000, -1.000, 0.604, -0.433, 1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, -0.254, -0.001, 1.000, 0.222, -1.000, -1.000, 0.068, 0.104, -1.000, 1.000, 1.000, 0.004, 1.000, 1.000],
        [1.000, -0.016, -0.881, -1.000, 1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 0.912, -0.909, 0.967, 0.796, 0.919, -1.000, -1.000, 0.070, 1.000, 0.643, 1.000, 1.000, 0.253, 0.626, 1.000],
        [-1.000, 1.000, -1.000, -1.000, -0.498, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 0.652, -1.000, 1.000, -0.671, 1.000, 0.274, -1.000, -0.029, 1.000, 1.000, 1.000, 1.000, 0.853, -0.929, -0.011],
        [1.000, 0.455, -1.000, -1.000, -0.773, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, -0.140, 0.789, -0.549, 0.433, -1.000, -1.000, -0.021, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 0.072, -1.000, -1.000, -1.000, -0.817, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, -0.160, 1.000, -0.469, 1.000, -0.345, -1.000, 0.432, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, -0.884, -1.000, -1.000, -0.433, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, -0.111, 1.000, 0.335, 1.000, -1.000, -1.000, 0.101, 1.000, 0.901, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, -0.386, -1.000, -1.000, 0.450, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, -0.739, 0.729, 0.721, 1.000, -1.000, -1.000, -0.319, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, -0.372, -1.000, -1.000, 0.891, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.478, 0.937, 1.000, -1.000, -1.000, -0.490, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [0.624, -0.227, -1.000, -1.000, 1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 0.954, -1.000, 0.340, 1.000, 1.000, -1.000, -1.000, -0.577, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.904],
        [0.521, -0.354, -1.000, -1.000, 1.000, -1.000, -0.922, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.391, 0.971, 1.000, -1.000, -1.000, -0.524, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.760],
        [0.451, -0.599, -1.000, -1.000, 1.000, -1.000, -0.746, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.397, 0.978, 1.000, -1.000, -1.000, -0.466, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.599],
        [0.404, -0.865, -1.000, -1.000, 1.000, -1.000, -0.562, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.388, 1.000, 1.000, -1.000, -1.000, -0.444, 1.000, 1.000, 1.000, 1.000, 1.000, 0.921, 0.462],
        [0.398, -1.000, -1.000, -1.000, 1.000, -1.000, -0.436, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.398, 1.000, 1.000, -1.000, -1.000, -0.440, 1.000, 1.000, 1.000, 1.000, 1.000, 0.875, 0.364],
        [0.385, -1.000, -1.000, -1.000, 1.000, -1.000, -0.339, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.428, 1.000, 1.000, -1.000, -1.000, -0.439, 1.000, 1.000, 1.000, 1.000, 1.000, 0.845, 0.273],
        [0.343, -1.000, -1.000, -1.000, 1.000, -1.000, -0.250, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.466, 1.000, 1.000, -1.000, -1.000, -0.429, 1.000, 1.000, 1.000, 1.000, 1.000, 0.820, 0.168],
        [0.225, -1.000, -1.000, -1.000, 1.000, -1.000, -0.067, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.477, 1.000, 1.000, -1.000, -1.000, -0.428, 1.000, 1.000, 1.000, 1.000, 1.000, 0.776, 0.029],
        [0.248, -1.000, -1.000, -1.000, 1.000, -1.000, -0.074, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.529, 1.000, 1.000, -1.000, -1.000, -0.445, 1.000, 1.000, 1.000, 1.000, 1.000, 0.817, 0.012],
        [0.355, -1.000, -1.000, -1.000, 1.000, -1.000, -0.098, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.531, 1.000, 1.000, -1.000, -1.000, -0.452, 1.000, 1.000, 1.000, 1.000, 1.000, 0.859, 0.004],
        [1.000, -1.000, -1.000, -1.000, 0.415, -1.000, -0.295, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.896, 0.708, 1.000, -1.000, -1.000, -0.168, 1.000, 1.000, 1.000, 1.000, 1.000, 0.861, 0.243],
        [1.000, -1.000, -0.720, -1.000, -0.707, -1.000, -0.268, -1.000, -1.000, 1.000, 1.000, 1.000, -0.834, 1.000, -0.308, 1.000, -1.000, -1.000, 0.348, 1.000, 0.119, 1.000, 1.000, 1.000, 0.898, 0.490],
        [1.000, -1.000, -0.341, -1.000, -1.000, 0.084, 0.067, -1.000, -1.000, 1.000, 1.000, 1.000, -0.796, 1.000, -1.000, 1.000, -0.505, -1.000, 0.704, 1.000, -0.013, 1.000, 1.000, 1.000, 0.308, 0.174],
        [0.934, -1.000, -0.530, -1.000, -1.000, 0.538, 0.212, -1.000, -1.000, 1.000, 1.000, 1.000, -0.940, 1.000, -1.000, 1.000, 0.118, -1.000, 0.897, 1.000, 0.353, 1.000, 1.000, 1.000, -0.384, -0.249],
        [1.000, -1.000, -0.875, -1.000, -1.000, 0.058, 0.123, -1.000, -1.000, 1.000, 1.000, 1.000, -0.952, 1.000, -1.000, 1.000, -0.121, -1.000, 0.861, 1.000, 0.407, 1.000, 1.000, 1.000, -0.156, -0.231],
        [1.000, -1.000, -0.956, -1.000, -1.000, -0.608, 0.017, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -0.407, -1.000, 0.824, 1.000, 0.422, 1.000, 1.000, 1.000, -0.155, -0.391],
        [1.000, -1.000, -0.820, -1.000, -1.000, -1.000, -0.091, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -0.770, -1.000, 0.715, 1.000, 0.232, 1.000, 1.000, 1.000, -0.130, -0.707],
        [1.000, -1.000, -0.825, -1.000, -1.000, -1.000, -0.076, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -1.000, -1.000, 0.767, 1.000, 0.009, 1.000, 1.000, 1.000, 0.158, -0.638],
        [1.000, -1.000, -0.107, -1.000, -1.000, -1.000, -0.360, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.894, -1.000, 1.000, -1.000, -1.000, 0.442, 1.000, -0.596, 1.000, 1.000, 1.000, 0.238, -1.000],
        [1.000, -1.000, -0.565, -1.000, -1.000, -1.000, -0.052, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -1.000, -1.000, 0.751, 1.000, -0.207, 1.000, 1.000, 1.000, -0.084, -1.000],
        [1.000, -1.000, -0.403, -1.000, -1.000, -1.000, -0.017, -1.000, -0.573, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -1.000, -1.000, 0.700, 1.000, -0.152, 1.000, 1.000, 1.000, -0.513, -1.000],
        [0.583, -1.000, -0.587, -1.000, -1.000, -1.000, 0.123, -1.000, -0.265, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -0.799, -1.000, 0.804, 1.000, 0.088, 1.000, 1.000, 1.000, -0.799, -1.000],
        [0.584, -1.000, -0.682, -1.000, -1.000, -0.959, 0.177, -1.000, -0.156, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -0.813, -1.000, 0.836, 1.000, 0.085, 1.000, 1.000, 1.000, -0.688, -1.000],
        [0.903, -1.000, -0.323, -1.000, -1.000, -1.000, 0.058, -1.000, 0.114, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 1.000, -1.000, -1.000, 0.628, 1.000, -0.211, 1.000, 1.000, 1.000, -0.538, -1.000],
        [0.826, -1.000, -0.205, -1.000, -1.000, -1.000, 0.014, -1.000, 0.367, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, 0.957, -1.000, -1.000, 0.531, 1.000, -0.221, 1.000, 1.000, 1.000, -0.627, -1.000],
        [0.800, -1.000, 0.486, -1.000, -1.000, -1.000, -0.152, -1.000, 0.977, 1.000, 1.000, 1.000, -1.000, 0.937, -1.000, 0.414, -1.000, -1.000, 0.164, 1.000, -0.749, 1.000, 0.605, 1.000, -0.538, -1.000],
        [-0.868, -0.877, 0.357, -1.000, -1.000, 0.509, -0.235, -1.000, 1.000, 0.821, 1.000, 1.000, -1.000, 1.000, -1.000, -0.084, -0.295, -0.796, 0.120, 1.000, -0.235, 1.000, -0.016, 1.000, -1.000, -1.000],
        [-0.785, 0.266, -0.762, -1.000, -1.000, 1.000, -1.000, -1.000, 1.000, -0.319, 0.692, 1.000, -0.903, 1.000, -1.000, -0.109, 0.657, -0.526, 0.310, 1.000, 0.875, 1.000, -0.202, 1.000, -1.000, -1.000],
        [1.000, -1.000, 0.390, -1.000, -1.000, -1.000, -1.000, -1.000, 1.000, 1.000, 1.000, 1.000, -0.151, 0.029, -1.000, -0.269, -1.000, -0.875, 0.181, 0.934, -1.000, 1.000, 0.010, 1.000, 0.726, -1.000],
        [-0.301, -0.454, -0.836, -1.000, -1.000, -0.925, -0.370, -1.000, 0.854, 0.512, 1.000, 1.000, -1.000, 1.000, -0.926, 1.000, 0.364, -1.000, 0.904, 1.000, 1.000, 1.000, 0.382, 1.000, -1.000, -1.000],
        [1.000, -1.000, 0.798, -1.000, -0.160, -1.000, -1.000, -1.000, 0.966, 1.000, 1.000, 1.000, -0.387, -0.518, -0.706, -0.388, -1.000, -1.000, -0.308, 0.149, -1.000, 1.000, -0.668, 1.000, 0.838, -1.000],
        [0.889, -1.000, 1.000, -1.000, 0.089, -1.000, -0.302, -1.000, 0.950, 1.000, 1.000, 0.780, -1.000, -0.198, -1.000, -0.070, -1.000, -1.000, -0.608, -0.183, -1.000, 1.000, -0.578, 1.000, -0.226, -1.000],
        [1.000, -1.000, 0.984, -1.000, 1.000, -1.000, 0.639, -1.000, -0.957, 1.000, 1.000, 1.000, -1.000, 1.000, 1.000, 1.000, -1.000, -1.000, 0.462, 0.105, -0.978, 1.000, 1.000, 0.763, -0.212, -0.721],
        [1.000, -1.000, 1.000, -1.000, 1.000, -1.000, 0.848, -1.000, -0.976, 1.000, 1.000, 1.000, -1.000, 1.000, 1.000, 1.000, -1.000, -1.000, 0.502, -0.157, -1.000, 1.000, 1.000, 0.435, -0.231, -0.616],
        [1.000, -1.000, 1.000, -1.000, 0.152, -1.000, 1.000, -1.000, -0.503, 1.000, 1.000, 1.000, -1.000, 1.000, 0.771, 1.000, -1.000, -1.000, 0.673, 0.770, -0.712, 1.000, 1.000, 0.680, -1.000, -1.000],
        [-0.571, -1.000, 0.848, -1.000, -1.000, 0.055, 1.000, -1.000, 0.088, 1.000, 1.000, 1.000, -1.000, 1.000, -0.601, 1.000, 0.074, -1.000, 0.878, 1.000, 0.355, 1.000, 1.000, 1.000, -1.000, -1.000],
        [-0.512, -0.745, 1.000, -1.000, -1.000, 1.000, -0.156, -1.000, 1.000, 1.000, 1.000, 1.000, -1.000, 1.000, -1.000, -0.433, -0.302, -0.848, -0.218, 1.000, -0.962, 0.694, -0.079, 1.000, -1.000, -1.000],
        [1.000, -0.742, 1.000, -1.000, -1.000, 0.184, -1.000, -1.000, 1.000, 1.000, 1.000, 1.000, -0.628, 1.000, -1.000, -0.678, -1.000, -1.000, -0.405, 1.000, -1.000, 1.000, -0.505, 1.000, -0.127, -1.000],
        [-1.000, 0.171, 1.000, -1.000, -1.000, 1.000, -0.342, -1.000, 1.000, 0.841, 1.000, 1.000, -1.000, 1.000, -1.000, -0.233, 0.508, -0.798, 0.006, 1.000, 0.337, 1.000, -0.457, 1.000, -1.000, -1.000],
        [0.081, 0.441, 0.929, -1.000, -1.000, 1.000, -1.000, -1.000, 1.000, 0.235, 0.733, 1.000, -0.513, 1.000, -1.000, -0.569, 0.144, -0.727, 0.179, 1.000, 0.498, 1.000, -0.976, 1.000, -1.000, -1.000],
        [1.000, -0.315, 1.000, -1.000, -1.000, 0.969, -1.000, -1.000, 1.000, 0.568, 0.723, 1.000, -0.232, 1.000, -1.000, -0.359, -0.342, -0.823, 0.490, 1.000, 0.144, 1.000, -0.714, 1.000, -0.787, -1.000],
        [-0.025, -0.853, 0.470, -1.000, -1.000, 0.903, -0.258, -1.000, 1.000, 0.767, 1.000, 1.000, -0.913, 1.000, -1.000, 1.000, 0.803, -0.986, 1.000, 1.000, 1.000, 1.000, 0.610, 1.000, -1.000, -1.000],
        [-0.384, -0.792, 0.087, -1.000, -1.000, -0.067, -0.042, -1.000, 0.586, 0.684, 1.000, 1.000, -1.000, 1.000, -0.849, 1.000, 0.511, -1.000, 1.000, 1.000, 1.000, 1.000, 0.611, 1.000, -1.000, -1.000],
        [-0.801, 0.170, -0.318, -1.000, -1.000, -0.873, -0.532, -1.000, 0.473, -0.062, 1.000, 1.000, -1.000, 1.000, -0.992, 1.000, 0.096, -1.000, 0.566, 1.000, 1.000, 1.000, -0.161, 1.000, -1.000, -1.000],
        [1.000, -1.000, 0.293, -1.000, -0.227, -1.000, -0.643, -1.000, -0.406, 1.000, 1.000, 1.000, -0.723, 1.000, 0.832, 1.000, -1.000, -1.000, 0.565, 1.000, 0.610, 1.000, 0.631, 1.000, -0.415, -0.848],
        [-0.987, 0.300, -0.936, -1.000, -1.000, -1.000, -0.289, -1.000, -0.243, 0.274, 1.000, 0.980, -1.000, 1.000, -0.207, 1.000, 0.207, -1.000, 0.599, 1.000, 1.000, 1.000, 0.451, 1.000, -1.000, -1.000],
        [1.000, -1.000, -1.000, -1.000, -0.617, -1.000, -0.593, -1.000, -1.000, 1.000, 1.000, 1.000, -0.400, 1.000, 0.202, 1.000, -1.000, -1.000, 0.885, 1.000, 1.000, 1.000, 1.000, 1.000, 0.499, 0.002],
        [0.187, -0.481, -1.000, -1.000, -0.217, -1.000, -0.193, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 1.000, 0.120, 1.000, -0.637, -1.000, 0.553, 1.000, 1.000, 1.000, 1.000, 1.000, -0.764, -1.000],
        [1.000, -1.000, -1.000, -1.000, 1.000, -1.000, -0.219, -1.000, -1.000, 1.000, 1.000, 1.000, -0.983, 0.424, 1.000, 1.000, -1.000, -1.000, 0.267, 1.000, 1.000, 1.000, 1.000, 0.814, 0.586, -0.001],
        [0.891, -1.000, -1.000, -1.000, 1.000, -1.000, 0.462, -1.000, -1.000, 1.000, 1.000, 0.824, -1.000, 0.214, 1.000, 1.000, -1.000, -1.000, 0.003, 1.000, 0.978, 1.000, 1.000, 0.272, 0.241, -0.309],
        [0.622, -1.000, -1.000, -1.000, 1.000, -1.000, 0.933, -1.000, -1.000, 1.000, 1.000, 0.835, -1.000, 0.335, 0.931, 1.000, -1.000, -1.000, -0.032, 0.765, 0.490, 1.000, 1.000, -0.083, -0.013, -0.441],
        [0.795, -1.000, -1.000, -1.000, 1.000, -1.000, 0.891, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.566, 0.401, 1.000, -1.000, -1.000, -0.032, 0.773, 0.321, 1.000, 1.000, -0.131, -0.154, -0.359],
        [1.000, -1.000, -1.000, -1.000, 1.000, -1.000, 0.849, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.607, -0.110, 1.000, -1.000, -1.000, 0.281, 0.960, 0.215, 1.000, 1.000, -0.241, -0.245, -0.366],
        [-1.000, -0.978, -1.000, -1.000, -1.000, 0.309, 1.000, -1.000, -1.000, 1.000, 1.000, 0.661, -1.000, 1.000, -1.000, 1.000, 1.000, -0.817, 0.869, 1.000, 1.000, 1.000, 1.000, 0.516, -1.000, -1.000],
        [1.000, -1.000, -0.512, -1.000, -0.731, -1.000, -0.454, -1.000, -0.773, 1.000, 1.000, 1.000, -1.000, -0.706, -1.000, 0.035, -1.000, -1.000, -0.748, 0.422, -1.000, 1.000, 1.000, 1.000, 0.718, -1.000],
        [1.000, -1.000, -0.652, -1.000, 0.420, -1.000, 0.672, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 0.753, 0.246, 1.000, -1.000, -1.000, 0.311, 0.817, -0.375, 1.000, 1.000, 0.884, 0.210, -0.585],
        [1.000, -1.000, 0.183, -1.000, 0.742, -1.000, 0.948, -1.000, -1.000, 1.000, 1.000, 1.000, -1.000, 1.000, 0.848, 1.000, -1.000, -1.000, 0.375, 0.449, -0.711, 1.000, 1.000, 0.636, -0.063, -0.763],
    ],
    dtype=np.float32,
)

def stretch_trajectory(actions, target_steps):
    orig_steps = len(actions)
    new_actions = np.zeros((target_steps, actions.shape[1]))
    for i in range(actions.shape[1]):
        new_actions[:, i] = np.interp(np.linspace(0, orig_steps - 1, target_steps), np.arange(orig_steps), actions[:, i])
    return np.float32(new_actions)

PICKUP_ACTIONS = RECORDED_ACTIONS[:50].copy()

# These timings keep the cycle executable with real actions only.  Longer
# action-only settle phases let the hammer sag and make later contacts less
# reliable in the PEMP start state.
RETRACT_STEPS = 15
IMPACT_STEPS = 8
SETTLE_STEPS = 2
POST_CONTACT_FOLLOW_THROUGH_STEPS = 2
NAIL_IMPACT_THRESHOLD = 0.5
RETRACT_ACTIONS = stretch_trajectory(RECORDED_ACTIONS[50:60].copy(), RETRACT_STEPS)
IMPACT_ACTIONS = stretch_trajectory(RECORDED_ACTIONS[60:70].copy(), IMPACT_STEPS)

# Enforce a stable, tightly clamped grasp after the pickup phase
stable_grasp = PICKUP_ACTIONS[-1, 4:].copy()
# Clamping flexion joints to 1.0
stable_grasp[1:4] = 1.0   # FF
stable_grasp[5:8] = 1.0   # MF
stable_grasp[9:12] = 1.0  # RF
stable_grasp[14:17] = 1.0 # LF
stable_grasp[19:22] = 1.0 # TH

RETRACT_ACTIONS[:, 4:] = stable_grasp
IMPACT_ACTIONS[:, 4:] = stable_grasp

SETTLE_ACTIONS = np.tile(RETRACT_ACTIONS[-1], (SETTLE_STEPS, 1))

NUM_STRIKES = 3
STRIKE_TEMPLATE_ACTIONS = np.vstack([RETRACT_ACTIONS, IMPACT_ACTIONS])

# Sequence: Pickup -> Retract -> Settle -> (Swing -> Retract -> Settle) * 3
trajectory_parts = [PICKUP_ACTIONS, RETRACT_ACTIONS, SETTLE_ACTIONS]
for _ in range(NUM_STRIKES):
    trajectory_parts.append(IMPACT_ACTIONS)
    trajectory_parts.append(RETRACT_ACTIONS)
    trajectory_parts.append(SETTLE_ACTIONS)

SCRIPTED_ACTIONS = np.vstack(trajectory_parts)



STRIKE_CYCLE_STEPS = RETRACT_STEPS + IMPACT_STEPS
IMPACT_PHASE_STEP = RETRACT_STEPS + 4  # scale roughly from 14/20
CONTACT_DISTANCE = 0.03
OFFSET_ERROR_WEIGHTS = np.array([2.0, 2.0, 1.0])
HAMMER_HEAD_GEOM_NAME = "head"
NAIL_BODY_NAME = "nail"

def build_arm_delta_grid() -> np.ndarray:
    candidates = []
    for d0 in (-0.8, -0.4, 0.0, 0.4, 0.8):
        for d1 in (-0.5, 0.0, 0.5):
            for d2 in (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6):
                for d3 in (0.0, 0.4, 0.8, 1.2):
                    candidates.append((d0, d1, d2, d3))
    return np.array(candidates, dtype=np.float32)

ARM_DELTA_CANDIDATES = build_arm_delta_grid()

def get_task_sites(env: gym.Env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_env = env.unwrapped
    palm = base_env.data.site_xpos[base_env.S_grasp_site_id].copy()
    tool = base_env.data.site_xpos[base_env.tool_site_id].copy()
    nail = base_env.data.site_xpos[base_env.target_obj_site_id].copy()
    return palm, tool, nail


def goal_distance(env: gym.Env) -> float:
    base_env = env.unwrapped
    nail = base_env.data.site_xpos[base_env.target_obj_site_id].copy()
    goal = base_env.data.site_xpos[base_env.goal_site_id].copy()
    return float(np.linalg.norm(nail - goal))


def object_nail_contact_names(env: gym.Env) -> list[str]:
    base_env = env.unwrapped
    model = base_env.model
    data = base_env.data

    object_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == int(base_env.obj_body_id)
    }
    nail_body_id = next(
        body_id for body_id in range(model.nbody) if model.body(body_id).name == "nail"
    )
    nail_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == nail_body_id
    }

    contacts: list[str] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        object_hits_nail = geom1 in object_geom_ids and geom2 in nail_geom_ids
        nail_hits_object = geom2 in object_geom_ids and geom1 in nail_geom_ids
        if object_hits_nail or nail_hits_object:
            name1 = model.geom(geom1).name or f"geom_{geom1}"
            name2 = model.geom(geom2).name or f"geom_{geom2}"
            contacts.append(f"{name1}:{name2}")
    return contacts


def strike_index_from_phase(phase: str) -> int | None:
    if not phase.startswith("strike_") or not phase.endswith("_swing"):
        return None
    try:
        return int(phase.split("_", maxsplit=2)[1]) - 1
    except (IndexError, ValueError):
        return None

def normalized_action_to_ctrl(base_env, action: np.ndarray) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    return base_env.act_mean + action * base_env.act_rng

def save_sim_state(base_env) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return (
        base_env.data.qpos.copy(),
        base_env.data.qvel.copy(),
        base_env.data.ctrl.copy(),
        float(base_env.data.time),
    )

def restore_sim_state(base_env, state: tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> None:
    qpos, qvel, ctrl, sim_time = state
    base_env.set_state(qpos, qvel)
    base_env.data.ctrl[:] = ctrl
    base_env.data.time = sim_time

def simulate_unrendered_step(base_env, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_env.do_simulation(normalized_action_to_ctrl(base_env, action), base_env.frame_skip)
    tool = base_env.data.site_xpos[base_env.tool_site_id].copy()
    nail = base_env.data.site_xpos[base_env.target_obj_site_id].copy()
    return tool, nail

def select_closed_loop_strike_action(
    env: gym.Env,
    base_action: np.ndarray,
    reference_offsets: list[np.ndarray],
    cycle_step: int,
    lookahead_steps: int,
) -> np.ndarray:
    base_env = env.unwrapped
    initial_state = save_sim_state(base_env)
    best_score = np.inf
    best_action = base_action.copy()
    horizon = max(1, min(lookahead_steps, STRIKE_CYCLE_STEPS - cycle_step))

    for arm_delta in ARM_DELTA_CANDIDATES:
        action = base_action.copy()
        action[:4] = np.clip(action[:4] + arm_delta, -1.0, 1.0)
        score = 0.0
        closest_impact_distance = np.inf

        for horizon_step in range(horizon):
            if horizon_step == 0:
                rollout_action = action
            else:
                template_step = min(cycle_step + horizon_step, STRIKE_CYCLE_STEPS - 1)
                rollout_action = IMPACT_ACTIONS[template_step - RETRACT_STEPS] if template_step >= RETRACT_STEPS else RETRACT_ACTIONS[template_step]

            tool, nail = simulate_unrendered_step(base_env, rollout_action)
            reference_step = min(cycle_step + horizon_step, len(reference_offsets) - 1)
            offset_error = (tool - nail) - reference_offsets[reference_step]
            score += (horizon_step + 1) * np.linalg.norm(OFFSET_ERROR_WEIGHTS * offset_error)

            if reference_step >= IMPACT_PHASE_STEP:
                closest_impact_distance = min(closest_impact_distance, np.linalg.norm(tool - nail))

        if cycle_step >= 10:
            score += 0.7 * closest_impact_distance

        restore_sim_state(base_env, initial_state)
        if score < best_score:
            best_score = score
            best_action = action

    restore_sim_state(base_env, initial_state)
    return best_action


def load_action_trajectory(path: Path) -> tuple[np.ndarray, list[str] | None, int | None]:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if "actions" not in loaded:
                raise ValueError(f"{path} does not contain an 'actions' array")
            actions = loaded["actions"].astype(np.float32)
            phases = loaded["phases"].astype(str).tolist() if "phases" in loaded else None
            seed = int(loaded["seed"].item()) if "seed" in loaded else None
    else:
        actions = loaded.astype(np.float32)
        phases = None
        seed = None

    if actions.ndim != 2 or actions.shape[1] != 26:
        raise ValueError(f"expected actions with shape (N, 26), got {actions.shape}")
    return np.clip(actions, -1.0, 1.0), phases, seed


def save_action_trajectory(
    path: Path,
    actions: np.ndarray,
    phases: list[str],
    seed: int,
    strike_count: int,
    final_goal_distance: float,
    success: bool,
    strike_hits: list[bool],
    strike_contact_steps: list[int],
    home_tool_positions: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        actions=actions.astype(np.float32),
        phases=np.array(phases, dtype="U48"),
        seed=np.array(seed, dtype=np.int64),
        env_id=np.array(ENV_ID),
        controller=np.array("pemp_pure_action_rhythmic"),
        pure_action_trajectory=np.array(True, dtype=bool),
        state_driven=np.zeros(len(actions), dtype=bool),
        num_strikes=np.array(strike_count, dtype=np.int64),
        retract_steps=np.array(RETRACT_STEPS, dtype=np.int64),
        impact_steps=np.array(IMPACT_STEPS, dtype=np.int64),
        settle_steps=np.array(SETTLE_STEPS, dtype=np.int64),
        post_contact_follow_through_steps=np.array(
            POST_CONTACT_FOLLOW_THROUGH_STEPS, dtype=np.int64
        ),
        nail_impact_threshold=np.array(NAIL_IMPACT_THRESHOLD, dtype=np.float32),
        final_goal_distance=np.array(final_goal_distance, dtype=np.float32),
        success=np.array(success, dtype=bool),
        strike_hits=np.array(strike_hits, dtype=bool),
        strike_contact_steps=np.array(strike_contact_steps, dtype=np.int64),
        home_tool_positions=(
            np.asarray(home_tool_positions, dtype=np.float64)
            if home_tool_positions
            else np.empty((0, 3), dtype=np.float64)
        ),
    )
    print(f"saved {len(actions)} actions to {path}")


def episode_save_path(base_path: Path, seed: int, episode_count: int) -> Path:
    if episode_count == 1:
        return base_path
    suffix = base_path.suffix or ".npz"
    return base_path.with_name(f"{base_path.stem}_seed{seed}{suffix}")


def normalize_save_path(path: Path) -> Path:
    if path.suffix:
        return path
    return path.with_suffix(".npz")


def resolve_load_path(path: Path) -> Path:
    if path.exists() or path.suffix:
        return path
    npz_path = path.with_suffix(".npz")
    return npz_path if npz_path.exists() else path


def hide_mujoco_info_pane(env: gym.Env) -> None:
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    if renderer is None:
        return

    viewer = renderer._get_viewer(render_mode="human")
    if hasattr(viewer, "_hide_menu"):
        viewer._hide_menu = True





def run_episode(
    seed: int,
    render: bool,
    sleep_s: float,
    verbose: bool,
    hide_info_pane: bool,
    replay_actions: np.ndarray | None,
    replay_phases: list[str] | None,
    save_actions_path: Path | None,
    strike_count: int,
) -> bool:
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=400)
    env.reset(seed=seed)

    if render and hide_info_pane:
        hide_mujoco_info_pane(env)

    success = False
    first_success_step: int | None = None
    executed_actions: list[np.ndarray] = []
    phases: list[str] = []
    strike_hits = [False for _ in range(strike_count)]
    strike_contact_steps = [-1 for _ in range(strike_count)]
    home_tool_positions: list[np.ndarray] = []

    def step_env(action: np.ndarray, phase: str) -> tuple[bool, bool]:
        nonlocal success, first_success_step

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        executed_actions.append(action.copy())
        phases.append(phase)
        obs, reward, terminated, truncated, info = env.step(action)
        action_step = len(executed_actions) - 1
        step_success = bool(info.get("success", False))
        if step_success and first_success_step is None:
            first_success_step = action_step
        success = success or step_success

        contact_names = object_nail_contact_names(env)
        nail_impact = float(obs[-1])
        hit = bool(contact_names) or nail_impact >= NAIL_IMPACT_THRESHOLD
        strike_index = strike_index_from_phase(phase)
        if (
            hit
            and strike_index is not None
            and 0 <= strike_index < strike_count
            and not strike_hits[strike_index]
        ):
            strike_hits[strike_index] = True
            strike_contact_steps[strike_index] = action_step

        should_log = (
            verbose
            or action_step % 10 == 0
            or hit
            or step_success
            or terminated
            or truncated
        )
        if should_log:
            _, tool, nail = get_task_sites(env)
            print(
                f"{action_step:03d} {phase:24s} "
                f"goal_dist={goal_distance(env):.4f} "
                f"tool={np.round(tool, 3)} "
                f"nail={np.round(nail, 3)} "
                f"impact={nail_impact:.2f} "
                f"contact={contact_names} "
                f"reward={reward:.3f} success={step_success}"
            )

        if render and sleep_s > 0:
            time.sleep(sleep_s)

        return terminated or truncated, hit

    if replay_actions is not None:
        for action_index, action in enumerate(replay_actions):
            phase = (
                replay_phases[action_index]
                if replay_phases is not None and action_index < len(replay_phases)
                else "replay"
            )
            done, _ = step_env(action, phase)
            if done:
                break
    else:
        for action in RETRACT_ACTIONS:
            done, _ = step_env(action, "go_to_x")
            if done:
                break

        settle_action = RETRACT_ACTIONS[-1]

        episode_done = False
        for strike_index in range(strike_count):
            strike_number = strike_index + 1
            home_tool_positions.append(get_task_sites(env)[1])

            first_hit_swing_step: int | None = None
            for swing_step, action in enumerate(IMPACT_ACTIONS):
                done, hit = step_env(action, f"strike_{strike_number}_swing")
                if hit and first_hit_swing_step is None:
                    first_hit_swing_step = swing_step
                if done:
                    episode_done = True
                    break
                if (
                    first_hit_swing_step is not None
                    and swing_step - first_hit_swing_step
                    >= POST_CONTACT_FOLLOW_THROUGH_STEPS
                ):
                    break
            if episode_done:
                break

            for action in RETRACT_ACTIONS:
                done, _ = step_env(action, f"strike_{strike_number}_retract_to_x")
                if done:
                    episode_done = True
                    break
            if episode_done:
                break

            for _ in range(SETTLE_STEPS):
                done, _ = step_env(settle_action, f"strike_{strike_number}_settle_at_x")
                if done:
                    episode_done = True
                    break
            if episode_done:
                break

    final_goal_distance = goal_distance(env)
    env.close()

    print(
        f"finished seed={seed} success={success} "
        f"first_success_step={first_success_step} "
        f"final_goal_distance={final_goal_distance:.4f} "
        f"strike_hits={strike_hits} "
        f"strike_contact_steps={strike_contact_steps}"
    )
    if save_actions_path is not None:
        save_action_trajectory(
            save_actions_path,
            np.asarray(executed_actions, dtype=np.float32),
            phases,
            seed=seed,
            strike_count=strike_count,
            final_goal_distance=final_goal_distance,
            success=success,
            strike_hits=strike_hits,
            strike_contact_steps=strike_contact_steps,
            home_tool_positions=home_tool_positions,
        )
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--show-info-pane", action="store_true")
    parser.add_argument("--save-actions", type=Path)
    parser.add_argument("--load-actions", type=Path)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    replay_actions = None
    replay_phases = None
    if args.load_actions is not None:
        load_path = resolve_load_path(args.load_actions)
        replay_actions, replay_phases, saved_seed = load_action_trajectory(load_path)
        print(f"loaded {len(replay_actions)} actions from {load_path}")
        if saved_seed is not None and saved_seed != args.seed:
            print(
                f"warning: trajectory was saved with seed={saved_seed}, "
                f"but this run uses seed={args.seed}"
            )

    save_actions_path = (
        normalize_save_path(args.save_actions) if args.save_actions is not None else None
    )

    if not args.no_render:
        time.sleep(3)
    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        run_episode(
            seed=episode_seed,
            render=not args.no_render,
            sleep_s=args.sleep,
            verbose=args.verbose,
            hide_info_pane=not args.show_info_pane,
            replay_actions=replay_actions,
            replay_phases=replay_phases,
            save_actions_path=(
                episode_save_path(save_actions_path, episode_seed, args.episodes)
                if save_actions_path is not None
                else None
            ),
            strike_count=NUM_STRIKES,
        )
