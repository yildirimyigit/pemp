# %%
import sys
import torch
from matplotlib import pyplot as plt
import numpy as np

folder_path = '../models/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

folder_path = '../data/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

from cnmp import CNMP
from positional_encoders import *

torch.set_float32_matmul_precision('high')

def get_free_gpu():
    gpu_util = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch GPU
#        gpu_util.append((i, torch.cuda.memory_stats()['reserved_bytes.all.current'] / (1024 ** 2)))
        gpu_util.append((i, torch.cuda.utilization()))
    gpu_util.sort(key=lambda x: x[1])
    return gpu_util[0][0]

if torch.cuda.is_available():
    available_gpu = get_free_gpu()
    if available_gpu == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{available_gpu}")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
print("Device :", device)

# %%
dx, dy, dg, dph, dpe = 1, 26, 1, 0, 27
num_demos, num_test = 8, 2
num_trajs = num_demos + num_test
t_steps = 200
n_max, m_max = 20, 20

trajectories, freqs = torch.from_numpy(np.load('../sim/data/adroit_actions_10.npy')), torch.from_numpy(np.load('../sim/data/adroit_freqs_10.npy'))
max_freq = max(freqs)

perm_ids = torch.randperm(num_trajs)
train_ids, test_ids = perm_ids[:num_demos], perm_ids[num_demos:]

all_x = torch.linspace(0, 1, t_steps).unsqueeze(-1).unsqueeze(0).repeat(num_trajs,1,1)

x_train, x_test = all_x[train_ids], all_x[test_ids]
y_train, y_test = trajectories[train_ids], trajectories[test_ids]
g_train, g_test = freqs[train_ids], freqs[test_ids]

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, g_train shape: {g_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}, g_test shape: {g_test.shape}")

# %%
pe = generate_positional_encoding(t_steps, dpe)

# %%
batch_size = 2

enc_dims = [256,256,256]
dec_dims = [256,256,256]

m0_ = CNMP(input_dim=dx+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=enc_dims, decoder_hidden_dims=dec_dims, batch_size=batch_size, device=device)
opt0 = torch.optim.Adam(lr=3e-4, params=m0_.parameters())

m1_ = CNMP(input_dim=dpe+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=enc_dims, decoder_hidden_dims=dec_dims, batch_size=batch_size, device=device)
opt1 = torch.optim.Adam(lr=3e-4, params=m1_.parameters())

pytorch_total_params = sum(p.numel() for p in m0_.parameters())
print('Bare: ', pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in m1_.parameters())
print('PE: ', pytorch_total_params)

if torch.__version__ >= "2.0":
    m0, m1 = torch.compile(m0_), torch.compile(m1_)
else:
    m0, m1 = m0_, m1_

# %%
obs0 = torch.zeros((batch_size, n_max, dx+dg+dy), dtype=torch.float32, device=device)
tar_x0 = torch.zeros((batch_size, m_max, dx+dg), dtype=torch.float32, device=device)

obs1 = torch.zeros((batch_size, n_max, dpe+dg+dy), dtype=torch.float32, device=device)
tar_x1 = torch.zeros((batch_size, m_max, dpe+dg), dtype=torch.float32, device=device)

tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)

def prepare_masked_batch(t: list, traj_ids: list):
    global obs0, tar_x0, obs1, tar_x1, tar_y, obs_mask, tar_mask
    obs0.fill_(0)
    tar_x0.fill_(0)
    obs1.fill_(0)
    tar_x1.fill_(0)
    tar_y.fill_(0)
    obs_mask.fill_(False)
    tar_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]

        n = torch.randint(1, n_max+1, (1,)).item()
        m = torch.randint(1, m_max+1, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = permuted_ids[n:n+m]

        obs0[i, :n, :dx] = x_train[traj_id, n_ids]  # t
        obs0[i, :n, dx:dx+dg] = g_train[traj_id]  # gamma
        obs0[i, :n, dx+dg:] = traj[n_ids]  # SM(t)

        obs1[i, :n, :dpe] = pe[n_ids]  # PE(t)
        obs1[i, :n, dpe:dpe+dg] = g_train[traj_id]  # gamma
        obs1[i, :n, dpe+dg:] = traj[n_ids]  # SM(t)

        obs_mask[i, :n] = True
        
        tar_x0[i, :m, :dx] = x_train[traj_id, m_ids]
        tar_x0[i, :m, dx:] = g_train[traj_id]
        tar_x1[i, :m, :dpe] = pe[m_ids]
        tar_x1[i, :m, dpe:] = g_train[traj_id]        
        
        tar_y[i, :m] = traj[m_ids]
        tar_mask[i, :m] = True


test_obs0 = torch.zeros((batch_size, n_max, dx+dg+dy), dtype=torch.float32, device=device)
test_tar_x0 = torch.zeros((batch_size, t_steps, dx+dg), dtype=torch.float32, device=device)

test_obs1 = torch.zeros((batch_size, n_max, dpe+dg+dy), dtype=torch.float32, device=device)
test_tar_x1 = torch.zeros((batch_size, t_steps, dpe+dg), dtype=torch.float32, device=device)

test_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
test_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
last_obs_vals = torch.zeros((batch_size, n_max, dx), dtype=torch.int32, device=device)  # only for plotting

def prepare_masked_test_batch(t: list, traj_ids: list, fixed_ind=None):
    global test_obs0, test_tar_x0, test_obs1, test_tar_x1, test_tar_y, test_obs_mask, last_obs_vals
    test_obs0.fill_(0)
    test_tar_x0.fill_(0)
    test_obs1.fill_(0)
    test_tar_x1.fill_(0)
    test_tar_y.fill_(0)
    test_obs_mask.fill_(False)
    last_obs_vals.fill_(0)

    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]

        # n = num_peaks #torch.randint(5, n_max, (1,)).item()
        n = torch.randint(1, n_max+1, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = torch.arange(t_steps)

        if fixed_ind != None:
            for p in range(n):
                n_ids[p] = fixed_ind[i, p]
            # n_ids[-1] = fixed_ind[i]

        test_obs0[i, :n, :dx] = x_test[traj_id, n_ids]  # t
        test_obs0[i, :n, dx:dx+dg] = g_test[traj_id]
        test_obs0[i, :n, dx+dg:] = traj[n_ids]  # SM(t)

        test_obs1[i, :n, :dpe] = pe[n_ids]  # PE(t)
        test_obs1[i, :n, dpe:dpe+dg] = g_test[traj_id]
        test_obs1[i, :n, dpe+dg:] = traj[n_ids]

        last_obs_vals[i, :n] = n_ids.unsqueeze(-1)
        test_obs_mask[i, :n] = True
        
        test_tar_x0[i, :, :dx] = x_test[traj_id, m_ids]
        test_tar_x1[i, :, :dpe] = pe[m_ids]

        test_tar_y[i] = traj[m_ids]

# %%
import time
import os


timestamp = int(time.time())
root_folder = f'../outputs/sim/adroit/bare_pe/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_models/'):
    os.makedirs(f'{root_folder}saved_models/')

img_folder = f'{root_folder}img/'
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

torch.save(y_train, f'{root_folder}y.pt')


epochs = 10_000_000
epoch_iter = num_demos // batch_size
test_epoch_iter = num_test//batch_size
avg_loss0, avg_loss1 = 0, 0
loss_report_interval = 500
test_per_epoch = 1000
min_test_loss0, min_test_loss1 = 1000000, 1000000
mse_loss = torch.nn.MSELoss()

plot_test = True

l0, l1 = [], []


# if plot_test == True

if dy > 1:
    bare_plot_start, bare_plot_end = dx+dg+1, dx+dg+2
    pe_plot_start, pe_plot_end = dpe+dg+1, dpe+dg+2
else:
    bare_plot_start, bare_plot_end = dx+dg, dx+dg+1
    pe_plot_start, pe_plot_end = dpe+dg, dpe+dg+1



for epoch in range(epochs):
    epoch_loss0, epoch_loss1 = 0, 0

    traj_ids = torch.randperm(num_demos)[:batch_size * epoch_iter].chunk(epoch_iter)

    for i in range(epoch_iter):
        prepare_masked_batch(y_train, traj_ids[i])

        opt0.zero_grad()
        pred0 = m0(obs0, tar_x0, obs_mask)
        loss0 = m0.loss(pred0, tar_y, tar_mask)
        loss0.backward()
        opt0.step()

        epoch_loss0 += loss0.item()


        opt1.zero_grad()
        pred1 = m1(obs1, tar_x1, obs_mask)
        loss1 = m1.loss(pred1, tar_y, tar_mask)
        loss1.backward()
        opt1.step()

        epoch_loss1 += loss1.item()


    if epoch % test_per_epoch == 0:# and epoch > 0:
        test_traj_ids = torch.randperm(num_test)[:batch_size*test_epoch_iter].chunk(test_epoch_iter)
        test_loss0, test_loss1 = 0, 0

        for j in range(test_epoch_iter):
            prepare_masked_test_batch(y_test, test_traj_ids[j])

            pred0 = m0.val(test_obs0, test_tar_x0, test_obs_mask)
            pred1 = m1.val(test_obs1, test_tar_x1, test_obs_mask)
            
            if plot_test:
                for k in range(batch_size):
                    current_n = test_obs_mask[k].sum().item()
                    plt.scatter(last_obs_vals[k, :current_n, :dx].cpu().numpy(), test_obs0[k, :current_n, bare_plot_start:bare_plot_end].cpu().numpy(), label='Condition')
                    plt.plot(test_tar_y[k, :, 0].cpu().numpy(), label=f"Groundtruth")
                    plt.plot(pred0[k, :, 0].cpu().numpy(), label=f"Prediction")
                    
                    plt.legend(loc='upper left')
                    plt.title(f'Epoch: {epoch}', fontsize=20)
                    plt.savefig(f'{img_folder}{epoch}_{test_traj_ids[j][k]}_bare.png')
                    plt.clf()

                    plt.scatter(last_obs_vals[k, :current_n, :dx].cpu().numpy(), test_obs1[k, :current_n, pe_plot_start:pe_plot_end].cpu().numpy(), label='Condition')
                    plt.plot(test_tar_y[k, :, 0].cpu().numpy(), label=f"Groundtruth")
                    plt.plot(pred1[k, :, 0].cpu().numpy(), label=f"Prediction")
                    
                    plt.legend(loc='upper left')
                    plt.title(f'Epoch: {epoch}', fontsize=20)
                    plt.savefig(f'{img_folder}{epoch}_{test_traj_ids[j][k]}_ph.png')
                    plt.clf()
                    

            test_loss0 += mse_loss(pred0[:, :, :m0.output_dim], test_tar_y).item()
            test_loss1 += mse_loss(pred1[:, :, :m1.output_dim], test_tar_y).item()
        
        test_loss0 /= test_epoch_iter
        test_loss1 /= test_epoch_iter
            
        if test_loss0 < min_test_loss0:
            min_test_loss0 = test_loss0
            print(f'New BARE best: {min_test_loss0}, PE best: {min_test_loss1}')
            torch.save(m0_.state_dict(), f'{root_folder}saved_models/bare.pt')

        if test_loss1 < min_test_loss1:
            min_test_loss1 = test_loss1
            print(f'New PE best: {min_test_loss1}, BARE best: {min_test_loss0}')
            torch.save(m1_.state_dict(), f'{root_folder}saved_models/pe.pt')


    epoch_loss0 /= epoch_iter
    epoch_loss1 /= epoch_iter

    avg_loss0 += epoch_loss0
    avg_loss1 += epoch_loss1

    l0.append(epoch_loss0)
    l1.append(epoch_loss1)

    if epoch % loss_report_interval == 0:
        print("Epoch: {}, Losses: BARE: {}, PE: {}".format(epoch, avg_loss0/loss_report_interval, avg_loss1/loss_report_interval))
        avg_loss0, avg_loss1 = 0, 0


# %%
# last_obs_vals.shape
test_obs0[k, current_n, dx:].shape

# %%
torch.save(l0, f'{root_folder}losses_bare.pt')
torch.save(l1, f'{root_folder}losses_pe.pt')


