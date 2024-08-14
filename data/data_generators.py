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
