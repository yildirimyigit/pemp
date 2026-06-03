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
seed='hostile_s1'

data_folder = f'../sim/data/fluidlab_mixing_{seed}/processed/'

x_train = torch.from_numpy(np.load(f'{data_folder}x.npy'))
x_test = torch.from_numpy(np.load(f'{data_folder}x_test.npy'))
y_train = torch.from_numpy(np.load(f'{data_folder}y.npy'))
y_test = torch.from_numpy(np.load(f'{data_folder}y_test.npy'))
g_train = torch.from_numpy(np.load(f'{data_folder}g.npy'))
g_test = torch.from_numpy(np.load(f'{data_folder}g_test.npy'))


dx, dy, dg, dpe = x_test.shape[-1], y_test.shape[-1], 1, 27
num_demos, num_test = x_train.shape[0], x_test.shape[0]
t_steps = x_test.shape[1]
n_max, m_max = 60, 60

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, g_train shape: {g_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}, g_test shape: {g_test.shape}")

# %%
pe = generate_positional_encoding(t_steps, dpe, frequency_scaler=0.3).to(device)

# %%
batch_size = num_test

enc_dims = [256, 256, 256]
dec_dims = [256, 256, 256]

m0_ = CNMP(input_dim=dx+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=enc_dims, decoder_hidden_dims=dec_dims, batch_size=batch_size, device=device, importance_weighting=True)
opt0 = torch.optim.Adam(lr=3e-4, params=m0_.parameters())

m1_ = CNMP(input_dim=dpe+dg, output_dim=dy, n_max=n_max, m_max=m_max, encoder_hidden_dims=enc_dims, decoder_hidden_dims=dec_dims, batch_size=batch_size, device=device, importance_weighting=True)
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


def prepare_masked_train_batch(traj_ids: list):
    global obs0, tar_x0, obs1, tar_x1, tar_y, obs_mask, tar_mask
    obs0.fill_(0)
    tar_x0.fill_(0)
    obs1.fill_(0)
    tar_x1.fill_(0)
    tar_y.fill_(0)
    obs_mask.fill_(False)
    tar_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        # Random number of context and target points
        n = torch.randint(1, n_max+1, (1,)).item()
        m = torch.randint(1, m_max+1, (1,)).item()

        # Random points for context and target
        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = permuted_ids[n:n+m]

        # Fill obs0 with t,g,SM(t)
        obs0[i, :n, :dx] = x_train[traj_id, n_ids]
        obs0[i, :n, dx:dx+dg] = g_train[traj_id]
        obs0[i, :n, dx+dg:] = y_train[traj_id, n_ids]

        # Fill tar_x0 with t,g
        tar_x0[i, :m, :dx] = x_train[traj_id, m_ids]
        tar_x0[i, :m, dx:] = g_train[traj_id]

        # Fill obs1 with PE(t),g,SM(t)
        obs1[i, :n, :dpe] = pe[n_ids]
        obs1[i, :n, dpe:dpe+dg] = g_train[traj_id]
        obs1[i, :n, dpe+dg:] = y_train[traj_id, n_ids]

        # Fill tar_x1 with PE(t),g
        tar_x1[i, :m, :dpe] = pe[m_ids]
        tar_x1[i, :m, dpe:] = g_train[traj_id]

        # Fill tar_y with SM(t)
        tar_y[i, :m] = y_train[traj_id, m_ids]

        # n/n_max and m/m_max are True and rest are masked
        obs_mask[i, :n] = True
        tar_mask[i, :m] = True


test_obs0 = torch.zeros((batch_size, n_max, dx+dg+dy), dtype=torch.float32, device=device)
test_tar_x0 = torch.zeros((batch_size, t_steps, dx+dg), dtype=torch.float32, device=device)

test_obs1 = torch.zeros((batch_size, n_max, dpe+dg+dy), dtype=torch.float32, device=device)
test_tar_x1 = torch.zeros((batch_size, t_steps, dpe+dg), dtype=torch.float32, device=device)

test_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
test_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)


def prepare_masked_test_batch(traj_ids: list):
    global test_obs0, test_tar_x0, test_obs1, test_tar_x1, test_tar_y, test_obs_mask
    test_obs0.fill_(0)
    test_tar_x0.fill_(0)
    test_obs1.fill_(0)
    test_tar_x1.fill_(0)
    test_tar_y.fill_(0)
    test_obs_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        n = torch.randint(1, n_max+1, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]

        test_obs0[i, :n, :dx] = x_test[traj_id, n_ids]
        test_obs0[i, :n, dx:dx+dg] = g_test[traj_id]
        test_obs0[i, :n, dx+dg:] = y_test[traj_id, n_ids]

        test_tar_x0[i, :, :dx] = x_test[traj_id]
        test_tar_x0[i, :, dx:] = g_test[traj_id]

        test_obs1[i, :n, :dpe] = pe[n_ids]
        test_obs1[i, :n, dpe:dpe+dg] = g_test[traj_id]
        test_obs1[i, :n, dpe+dg:] = y_test[traj_id, n_ids]

        test_tar_x1[i, :, :dpe] = pe
        test_tar_x1[i, :, dpe:] = g_test[traj_id]

        test_tar_y[i] = y_test[traj_id]
        test_obs_mask[i, :n] = True

# %%
import time
import os


timestamp = int(time.time())
root_folder = f'../outputs/sim/mixing/{str(timestamp)}_{seed}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_models/'):
    os.makedirs(f'{root_folder}saved_models/')

img_folder = f'{root_folder}img/'
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

torch.save(x_train, f'{root_folder}x.pt')
torch.save(x_test, f'{root_folder}x_test.pt')
torch.save(y_train, f'{root_folder}y.pt')
torch.save(y_test, f'{root_folder}y_test.pt')
torch.save(g_train, f'{root_folder}g.pt')
torch.save(g_test, f'{root_folder}g_test.pt')

epochs = 1_000_000
epoch_iter = num_demos // batch_size
test_epoch_iter = num_test//batch_size
avg_loss0, avg_loss1 = 0, 0
loss_report_interval = 500
test_per_epoch = 1000
min_test_loss0, min_test_loss1 = 1000000, 1000000
mse_loss = torch.nn.MSELoss()

plot_test = False

l0, l1 = [], []

for epoch in range(epochs):
    epoch_loss0, epoch_loss1 = 0, 0

    traj_ids = torch.randperm(num_demos)[:batch_size * epoch_iter].chunk(epoch_iter)

    for i in range(epoch_iter):
        prepare_masked_train_batch(traj_ids[i])

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

    if epoch % test_per_epoch == 0:
        test_traj_ids = torch.randperm(num_test)[:batch_size*test_epoch_iter].chunk(test_epoch_iter)
        test_loss0, test_loss1 = 0, 0

        for j in range(test_epoch_iter):
            prepare_masked_test_batch(test_traj_ids[j])

            pred0 = m0.val(test_obs0, test_tar_x0, test_obs_mask)
            pred1 = m1.val(test_obs1, test_tar_x1, test_obs_mask)

            test_loss0 += mse_loss(pred0[:, :, :m0.output_dim], test_tar_y).item()
            test_loss1 += mse_loss(pred1[:, :, :m1.output_dim], test_tar_y).item()

            # TODO: Plot predictions vs ground truth for the first test batch

        test_loss0 /= test_epoch_iter  # Average MSE test loss over test batches
        test_loss1 /= test_epoch_iter
            
        if test_loss0 < min_test_loss0:
            min_test_loss0 = test_loss0
            print(f'New BARE best: {min_test_loss0}, PE best: {min_test_loss1}')
            torch.save(m0_.state_dict(), f'{root_folder}saved_models/bare.pt')

        if test_loss1 < min_test_loss1:
            min_test_loss1 = test_loss1
            print(f'New PE best: {min_test_loss1}, BARE best: {min_test_loss0}')
            torch.save(m1_.state_dict(), f'{root_folder}saved_models/pe.pt')
        
    epoch_loss0 /= epoch_iter  # Average NLL train loss over training batches
    epoch_loss1 /= epoch_iter

    avg_loss0 += epoch_loss0
    avg_loss1 += epoch_loss1

    l0.append(epoch_loss0)
    l1.append(epoch_loss1)

    if epoch % loss_report_interval == 0:
        print("Epoch: {}, Losses: BARE: {}, PE: {}".format(epoch, avg_loss0/loss_report_interval, avg_loss1/loss_report_interval))
        avg_loss0, avg_loss1 = 0, 0

# %%
torch.save(l0, f'{root_folder}losses_bare.pt')
torch.save(l1, f'{root_folder}losses_pe.pt')

# %%
# write neural network topology and hyperparameters to a yaml file for later reference
import yaml

hyperparameters = {
    "enc_dims": enc_dims,
    "dec_dims": dec_dims,
    "batch_size": batch_size,
    "learning_rate": 3e-4,
    "epochs": epochs,
    "t_steps": t_steps,
    "min_freq": min(g_train).item(),
}

with open(f'{root_folder}hyperparameters.yaml', 'w') as f:
    yaml.dump(hyperparameters, f)

# %%



