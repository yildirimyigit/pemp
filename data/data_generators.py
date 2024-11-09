import torch
import numpy as np


def generate_trajectory(x, peak_x, peak_width, peak_height, noise_level=0.0005):
    y = torch.zeros_like(x)

    y += peak_height * torch.exp(-((x - peak_x)**2) / (2 * peak_width**2))
    y += noise_level * torch.randn_like(x)
    return y.unsqueeze(-1)

# def generate_trajectory(x, peak_x, peak_width, peak_height, noise_level=0.0005):
#     y = torch.zeros_like(x)

#     y += peak_height * torch.exp(-((x - peak_x)**2) / (2 * peak_width**2))
#     y += noise_level * torch.randn_like(x)
#     return y.unsqueeze(-1)


def one_peak(num_traj=4, peak_width=0.06, peak_height=0.2, t_steps=200, interp=True):
    offset = 2 * peak_width

    peak_positions = (torch.linspace(offset, 1 - offset, num_traj+2) + 0.00 * torch.randn(num_traj+2))[1:-1]

    y = torch.zeros(num_traj, t_steps, 1)
    x = torch.linspace(0, 1, t_steps)

    for i, pp in enumerate(peak_positions):
        y[i] = generate_trajectory(x, pp, peak_width, peak_height)
    
    return y, peak_positions.unsqueeze(-1)

def two_peaks(num_traj=4, peak_width=0.035, peak_height=0.2, t_steps=200):
    y = torch.zeros(num_traj, t_steps, 1)
    x = torch.linspace(0, 1, t_steps)

    peak_positions = torch.zeros(num_traj, 2)
    
    for i in range(num_traj):
        peak_positions[i, 0] = torch.rand(1)*0.3+0.2 #torch.linspace(0.2, 0.8, 2) + 0.02 * torch.randn(2)
        peak_positions[i, 1] = torch.rand(1)*0.3+0.6 #torch.linspace(0.2, 0.8, 2) + 0.02 * torch.randn(2)
    
    for i in range(num_traj):
        y[i] += generate_trajectory(x, peak_positions[i, 0], peak_width, peak_height)
        y[i] += generate_trajectory(x, peak_positions[i, 1], peak_width, peak_height)
    
    return y, peak_positions

def n_peaks(num_peaks, num_trajs=4, peak_pos_noise=0.03, t_steps=200, noise_level=0.0015):
    x = torch.linspace(0, 1, t_steps).view(-1, 1)
    
    # Calculate the width and height of the peaks to ensure they don't overlap and are broad
    peak_width = 0.55 / num_peaks  # Spread the peaks across 55% of the x range
    peak_height = 0.4 / (num_peaks)  # Adjust peak height to be less pointy

    y_all = torch.zeros(num_trajs, t_steps, 1)
    pps = torch.zeros(num_trajs, num_peaks)

    for i in range(num_trajs):
        peak_positions = torch.linspace(peak_width/1.1, 1-peak_width/1.2, num_peaks) + peak_pos_noise * torch.randn(num_peaks)
        y = torch.zeros(t_steps, 1)

        for pp in peak_positions:
            y += peak_height * torch.exp(-((x - pp)**2) / (2 * (peak_width/8)**2))

        y += noise_level * torch.randn_like(x)  # Add some noise to the signal
        y_all[i] = y
        pps[i] = peak_positions
    
    return x.unsqueeze(0).repeat(num_trajs, 1, 1), y_all, pps


def generate_cyclic_trajectory(num_cycles=5, num_points_per_cycle=100):
    """
    Generate a 1D cyclic trajectory with corresponding phase values.
    
    Args:
        num_cycles (int): Number of cycles to generate.
        num_points_per_cycle (int): Number of points per cycle.
        # amplitude (float): Amplitude of the sine wave.
        # frequency (float): Frequency of the sine wave.

    Returns:
        trajectory (torch.Tensor): 1D trajectory points.
        phase (torch.Tensor): Phase values in range [0, 1] for each point.
    """

    amplitude=0.98
    frequency=1.0

    # Total number of points
    num_points = num_cycles * num_points_per_cycle

    # Generate phase values for each point in [0, 1]
    phase = torch.linspace(0, 1, num_points_per_cycle).repeat(num_cycles)
    
    # Generate time values for the full trajectory
    offset = torch.rand(1).item() * 2 * np.pi
    time = torch.linspace(offset, offset + num_cycles * 2 * np.pi, num_points)
    
    # Generate 1D sinusoidal trajectory
    trajectory = amplitude * torch.sin(frequency * time) + (1-amplitude) * torch.randn(num_points)
    
    return trajectory.unsqueeze(-1), phase.unsqueeze(-1)


def generate_cyclic_trajectories(num_trajs=10, num_cycles=5, num_points_per_cycle=100):
    """
    Generate num_trajs 1D cyclic trajectories with corresponding phase values.
    
    Args:
        num_trajs (int): Number of trajectories to generate.
        num_cycles (int): Number of cycles to generate.
        num_points_per_cycle (int): Number of points per cycle.

    Returns:
        trajectories (torch.Tensor): Cyclic trajectories in the shape (num_trajs, num_points, 1).
        phases (torch.Tensor): Phase values in range [0, 1] for each point. Shape (num_trajs, num_points, 1).
    """
    # Total number of points
    num_points = num_cycles * num_points_per_cycle
    trajectories, phases = torch.zeros(num_trajs, num_points, 1), torch.zeros(num_trajs, num_points, 1)

    for i in range(num_trajs):
        trajectories[i], phases[i] = generate_cyclic_trajectory(num_cycles, num_points_per_cycle)
    
    return trajectories, phases


def generate_cyclic_trajectories_with_random_cycles(num_trajs=10, t_steps=1200):
    """
    Generate num_trajs 1D cyclic trajectories with random number of cycles and corresponding phase values.
    
    Args:
        num_trajs (int): Number of trajectories to generate.
        t_steps (int): Number of time steps per trajectory.

    Returns:
        trajectories (torch.Tensor): Cyclic trajectories in the shape (num_trajs, t_steps, 1).
        phases (torch.Tensor): Phase values in range [0, 1] for each point. Shape (num_trajs, t_steps, 1).
    """
    trajectories, phases = torch.zeros(num_trajs, t_steps, 1), torch.zeros(num_trajs, t_steps, 1)

    for i in range(num_trajs):
        num_cycles = np.random.randint(1, 5)
        num_points_per_cycle = t_steps // num_cycles
        trajectories[i], phases[i] = generate_cyclic_trajectory(num_cycles, num_points_per_cycle)
    
    return trajectories, phases