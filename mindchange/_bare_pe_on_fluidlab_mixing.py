# PEMP (bare-PE CNMP) training on the FluidLab Mixing periodic-stir dataset.
# Mirrors bare_pe_on_adroit_actions_with_freq.py but with dy=3 (stirrer xyz) and
# t_steps=200, loading sim/data/fluidlab_mixing/.  Run from the mindchange/ dir:
#   PEMP_EPOCHS=2 PEMP_TEST_EVERY=1 python bare_pe_on_fluidlab_mixing.py   # smoke
#   python bare_pe_on_fluidlab_mixing.py                                   # full
import sys
import os
import time
import torch
import numpy as np
from matplotlib import pyplot as plt

for p in ('../models/', '../data/'):
    if p not in sys.path:
        sys.path.append(p)

from cnmp import CNMP
from positional_encoders import *

torch.set_float32_matmul_precision('high')


def get_free_gpu():
    try:
        gpu_util = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            gpu_util.append((i, torch.cuda.utilization()))
        gpu_util.sort(key=lambda x: x[1])
        return gpu_util[0][0]
    except Exception:
        return 0  # pynvml unavailable -> just use cuda:0


if torch.cuda.is_available():
    available_gpu = get_free_gpu()
    device = torch.device("cuda:0" if available_gpu == 0 else f"cuda:{available_gpu}")
else:
    device = torch.device("cpu")
print("Device :", device)

# ---- dims / dataset ----
dx, dy, dg, dph, dpe = 1, 3, 1, 0, 27
t_steps = 200
n_max, m_max = 12, 12

# Actions live in [-ACT_LIMIT, ACT_LIMIT] (signal std ~0.0046).  The CNMP trains a
# Gaussian NLL whose predicted std starts at softplus(0)=0.69 -- ~150x the signal
# scale -- so the oscillation is "explained away as noise" and the mean collapses
# to a flat line.  Normalize y to ~[-1, 1] so the learning signal is well-scaled.
# At rollout, multiply model outputs by ACT_LIMIT to get back real sim actions.
ACT_LIMIT = 0.007
DATA = '../sim/data/fluidlab_mixing/'
y_train = (torch.from_numpy(np.load(DATA + 'y.npy')).float() / ACT_LIMIT).to(device)    # (num_demos,T,3) normalized
g_train = torch.from_numpy(np.load(DATA + 'g.npy')).float().to(device)                  # (num_demos,)
y_test = (torch.from_numpy(np.load(DATA + 'y_test.npy')).float() / ACT_LIMIT).to(device)  # (num_test,T,3) normalized
g_test = torch.from_numpy(np.load(DATA + 'g_test.npy')).float().to(device)
num_demos, num_test = y_train.shape[0], y_test.shape[0]

x_train = torch.linspace(0, 1, t_steps, device=device).reshape(1, -1, 1).repeat(num_demos, 1, 1)
x_test = torch.linspace(0, 1, t_steps, device=device).reshape(1, -1, 1).repeat(num_test, 1, 1)
print(f"x_train {tuple(x_train.shape)} y_train {tuple(y_train.shape)} g_train {tuple(g_train.shape)}")
print(f"x_test  {tuple(x_test.shape)} y_test  {tuple(y_test.shape)} g_test  {tuple(g_test.shape)}")

# Data freqs (loops 2-6 over 200 steps) are angular freq ~0.063..0.19 rad/step.
# frequency_scaler=0.3 places the top PE band (~0.3) just above the fastest signal
# and, with dpe=27, spreads the geometric bands densely across that range (the old
# dpe=10/scaler=0.2 left a gap right where the data frequencies sit).
pe_freq_scaler = 0.3
pe = generate_positional_encoding(t_steps, dpe, frequency_scaler=pe_freq_scaler)
if torch.is_tensor(pe):
    pe = pe.to(device)
else:
    pe = torch.as_tensor(pe, dtype=torch.float32, device=device)

# ---- models ----
batch_size = 4
enc_dims, dec_dims = [256, 256], [256, 256]
m0_ = CNMP(input_dim=dx + dg, output_dim=dy, n_max=n_max, m_max=m_max,
           encoder_hidden_dims=enc_dims, decoder_hidden_dims=dec_dims,
           batch_size=batch_size, device=device)
opt0 = torch.optim.Adam(lr=3e-4, params=m0_.parameters())
m1_ = CNMP(input_dim=dpe + dg, output_dim=dy, n_max=n_max, m_max=m_max,
           encoder_hidden_dims=enc_dims, decoder_hidden_dims=dec_dims,
           batch_size=batch_size, device=device)
opt1 = torch.optim.Adam(lr=3e-4, params=m1_.parameters())
print('Bare params:', sum(p.numel() for p in m0_.parameters()))
print('PE   params:', sum(p.numel() for p in m1_.parameters()))

m0, m1 = (torch.compile(m0_), torch.compile(m1_)) if torch.__version__ >= "2.0" else (m0_, m1_)

# ---- batch buffers ----
obs0 = torch.zeros((batch_size, n_max, dx + dg + dy), dtype=torch.float32, device=device)
tar_x0 = torch.zeros((batch_size, m_max, dx + dg), dtype=torch.float32, device=device)
obs1 = torch.zeros((batch_size, n_max, dpe + dg + dy), dtype=torch.float32, device=device)
tar_x1 = torch.zeros((batch_size, m_max, dpe + dg), dtype=torch.float32, device=device)
tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)


def prepare_masked_batch(t, traj_ids):
    obs0.fill_(0); tar_x0.fill_(0); obs1.fill_(0); tar_x1.fill_(0)
    tar_y.fill_(0); obs_mask.fill_(False); tar_mask.fill_(False)
    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]
        n = torch.randint(1, n_max + 1, (1,)).item()
        m = torch.randint(1, m_max + 1, (1,)).item()
        perm = torch.randperm(t_steps)
        n_ids, m_ids = perm[:n], perm[n:n + m]

        obs0[i, :n, :dx] = x_train[traj_id, n_ids]
        obs0[i, :n, dx:dx + dg] = g_train[traj_id]
        obs0[i, :n, dx + dg:] = traj[n_ids]
        obs1[i, :n, :dpe] = pe[n_ids]
        obs1[i, :n, dpe:dpe + dg] = g_train[traj_id]
        obs1[i, :n, dpe + dg:] = traj[n_ids]
        obs_mask[i, :n] = True

        tar_x0[i, :m, :dx] = x_train[traj_id, m_ids]
        tar_x0[i, :m, dx:] = g_train[traj_id]
        tar_x1[i, :m, :dpe] = pe[m_ids]
        tar_x1[i, :m, dpe:] = g_train[traj_id]
        tar_y[i, :m] = traj[m_ids]
        tar_mask[i, :m] = True


test_obs0 = torch.zeros((batch_size, n_max, dx + dg + dy), dtype=torch.float32, device=device)
test_tar_x0 = torch.zeros((batch_size, t_steps, dx + dg), dtype=torch.float32, device=device)
test_obs1 = torch.zeros((batch_size, n_max, dpe + dg + dy), dtype=torch.float32, device=device)
test_tar_x1 = torch.zeros((batch_size, t_steps, dpe + dg), dtype=torch.float32, device=device)
test_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
test_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
last_obs_vals = torch.zeros((batch_size, n_max, dx), dtype=torch.int32, device=device)


def prepare_masked_test_batch(t, traj_ids):
    test_obs0.fill_(0); test_tar_x0.fill_(0); test_obs1.fill_(0); test_tar_x1.fill_(0)
    test_tar_y.fill_(0); test_obs_mask.fill_(False); last_obs_vals.fill_(0)
    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]
        n = torch.randint(1, n_max + 1, (1,)).item()
        perm = torch.randperm(t_steps)
        n_ids, m_ids = perm[:n], torch.arange(t_steps)

        test_obs0[i, :n, :dx] = x_test[traj_id, n_ids]
        test_obs0[i, :n, dx:dx + dg] = g_test[traj_id]
        test_obs0[i, :n, dx + dg:] = traj[n_ids]
        test_obs1[i, :n, :dpe] = pe[n_ids]
        test_obs1[i, :n, dpe:dpe + dg] = g_test[traj_id]
        test_obs1[i, :n, dpe + dg:] = traj[n_ids]
        last_obs_vals[i, :n] = n_ids.unsqueeze(-1).to(device).to(torch.int32)
        test_obs_mask[i, :n] = True

        test_tar_x0[i, :, :dx] = x_test[traj_id, m_ids]
        test_tar_x1[i, :, :dpe] = pe[m_ids]
        test_tar_y[i] = traj[m_ids]


# ---- output dirs ----
timestamp = int(time.time())
root_folder = f'../outputs/sim/mixing/bare_pe/{timestamp}/'
img_folder = f'{root_folder}img/'
os.makedirs(f'{root_folder}saved_models/', exist_ok=True)
os.makedirs(img_folder, exist_ok=True)
torch.save(y_train, f'{root_folder}y.pt')

# ---- train ----
epochs = int(os.environ.get('PEMP_EPOCHS', 2_000_000))
test_per_epoch = int(os.environ.get('PEMP_TEST_EVERY', 1000))
loss_report_interval = int(os.environ.get('PEMP_REPORT_EVERY', 1000))
epoch_iter = max(1, num_demos // batch_size)
test_epoch_iter = max(1, num_test // batch_size)
mse_loss = torch.nn.MSELoss()
plot_test = os.environ.get('PEMP_PLOT', '1') == '1'
min_test_loss0, min_test_loss1 = float('inf'), float('inf')
avg_loss0, avg_loss1 = 0.0, 0.0
l0, l1 = [], []

bare_ps, bare_pe_ = (dx + dg + 1, dx + dg + 2) if dy > 1 else (dx + dg, dx + dg + 1)
pe_ps, pe_pe = (dpe + dg + 1, dpe + dg + 2) if dy > 1 else (dpe + dg, dpe + dg + 1)

for epoch in range(epochs):
    epoch_loss0 = epoch_loss1 = 0.0
    traj_ids = torch.randperm(num_demos)[:batch_size * epoch_iter].chunk(epoch_iter)
    for i in range(epoch_iter):
        prepare_masked_batch(y_train, traj_ids[i])

        opt0.zero_grad()
        loss0 = m0.loss(m0(obs0, tar_x0, obs_mask), tar_y, tar_mask)
        loss0.backward(); opt0.step(); epoch_loss0 += loss0.item()

        opt1.zero_grad()
        loss1 = m1.loss(m1(obs1, tar_x1, obs_mask), tar_y, tar_mask)
        loss1.backward(); opt1.step(); epoch_loss1 += loss1.item()

    if epoch % test_per_epoch == 0:
        test_traj_ids = torch.randperm(num_test)[:batch_size * test_epoch_iter].chunk(test_epoch_iter)
        test_loss0 = test_loss1 = 0.0
        for j in range(test_epoch_iter):
            prepare_masked_test_batch(y_test, test_traj_ids[j])
            pred0 = m0.val(test_obs0, test_tar_x0, test_obs_mask)
            pred1 = m1.val(test_obs1, test_tar_x1, test_obs_mask)
            if plot_test:
                for k in range(batch_size):
                    cn = test_obs_mask[k].sum().item()
                    plt.plot(test_tar_y[k, :, 0].cpu().numpy(), label="Groundtruth")
                    plt.plot(pred1[k, :, 0].detach().cpu().numpy(), label="Prediction")
                    plt.legend(loc='upper left'); plt.title(f'Epoch: {epoch}')
                    plt.savefig(f'{img_folder}{epoch}_{int(test_traj_ids[j][k])}_pe.png'); plt.clf()
            test_loss0 += mse_loss(pred0[:, :, :m0_.output_dim], test_tar_y).item()
            test_loss1 += mse_loss(pred1[:, :, :m1_.output_dim], test_tar_y).item()
        test_loss0 /= test_epoch_iter; test_loss1 /= test_epoch_iter
        if test_loss0 < min_test_loss0:
            min_test_loss0 = test_loss0
            torch.save(m0_.state_dict(), f'{root_folder}saved_models/bare.pt')
        if test_loss1 < min_test_loss1:
            min_test_loss1 = test_loss1
            print(f'New PE best: {min_test_loss1:.6f} (bare best: {min_test_loss0:.6f})')
            torch.save(m1_.state_dict(), f'{root_folder}saved_models/pe.pt')

    epoch_loss0 /= epoch_iter; epoch_loss1 /= epoch_iter
    avg_loss0 += epoch_loss0; avg_loss1 += epoch_loss1
    l0.append(epoch_loss0); l1.append(epoch_loss1)
    if epoch % loss_report_interval == 0:
        print(f"Epoch {epoch}: BARE {avg_loss0/loss_report_interval:.6f} PE {avg_loss1/loss_report_interval:.6f}")
        avg_loss0 = avg_loss1 = 0.0

torch.save(l0, f'{root_folder}losses_bare.pt')
torch.save(l1, f'{root_folder}losses_pe.pt')
print('saved models + losses to', root_folder)

import yaml

hyperparameters = {
    "enc_dims": enc_dims,
    "dec_dims": dec_dims,
    "batch_size": batch_size,
    "learning_rate": 3e-4,
    "epochs": epochs,
    "t_steps": t_steps,
    "min_freq": min(g_train).item(),
    "dpe": dpe,
    "pe_freq_scaler": pe_freq_scaler,
    "act_limit": ACT_LIMIT,  # y was normalized by this; multiply preds by it for sim rollout
}

with open(f'{root_folder}hyperparameters.yaml', 'w') as f:
    yaml.dump(hyperparameters, f)