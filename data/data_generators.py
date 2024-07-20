import torch


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

# Three peaks method
def three_peaks(num_traj=4, peak_width=0.06, peak_height=0.2, t_steps=200):
    y = torch.zeros(num_traj, t_steps, 1)
    x = torch.linspace(0, 1, t_steps)
    
    for i in range(num_traj):
        peak_positions = torch.linspace(0.2, 0.8, 3) + 0.02 * torch.randn(3)
        for pp in peak_positions:
            y[i] += generate_trajectory(x, pp, peak_width, peak_height)
    
    return y, peak_positions.unsqueeze(-1)

# Four peaks method
def four_peaks(num_traj=4, peak_width=0.06, peak_height=0.2, t_steps=200):
    y = torch.zeros(num_traj, t_steps, 1)
    x = torch.linspace(0, 1, t_steps)
    
    for i in range(num_traj):
        peak_positions = torch.linspace(0.2, 0.8, 4) + 0.02 * torch.randn(4)
        for pp in peak_positions:
            y[i] += generate_trajectory(x, pp, peak_width, peak_height)
    
    return y, peak_positions.unsqueeze(-1)
