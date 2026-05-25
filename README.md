# Position-Enhanced Movement Primitives

<img width="900" height="360" alt="mixing_comparison_low_res" src="https://github.com/user-attachments/assets/1cce3ba8-a7ee-4e1b-9682-e9dcf922f4cd" />

(Too messy now. I'm going to tidy up after paper submission.)

# Setup & quickstart — PEMP synthetic comparison
---

## 1. Environment

Requires conda (Anaconda/Miniconda). The env spec is [`conda_environment.yml`](conda_environment.yml).

```bash
cd ~/projects/pemp
conda env create -f conda_environment.yml      # creates env "pemp-gpu" (Python 3.10)
conda activate pemp-gpu
```

What it pulls in: `pytorch>=2.0` (CUDA 12.4 build), `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, and via pip `gmr` + `movement-primitives` (the ProMP/GMM baselines).
(The gymnasium/mujoco/SB3 pins in the file are only for the `sim/` MuJoCo scripts and
are **not** needed for this notebook.)

**GPU vs CPU.** The notebook auto-selects a free CUDA GPU and falls back to CPU if none
is available, so it runs either way — but `t_steps=1200`, `n_max=m_max=100` makes CPU
training slow; a GPU is recommended. On a CPU-only machine, replace `pytorch-cuda=12.4`
in the yml with the CPU build (`cpuonly` from the `pytorch` channel) before creating the
env.

Sanity check:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import gmr, movement_primitives; print('baselines OK')"
```

---

## 2. Run the notebook (Example Run)

The notebook uses paths relative to `mindchange/` (`../models/`, `../data/`,
`../outputs/`), so launch Jupyter **from that directory** (or open it in the IDE with the
repo as the workspace — the relative paths resolve the same way):

```bash
cd ~/projects/pemp/mindchange
jupyter notebook bare_pe_promp_gmm_with_freq.ipynb
```

Then run all cells. The flow is:

1. Imports + device pick (`models/cnmp.py`, `data/{data_generators,positional_encoders,plotters}.py`).
2. Generate synthetic cyclic trajectories (`generate_cyclic_trajectories_with_random_cycles`,
   `max_freq=5`, 180 train / 20 test) and the `PE(t)` encoding.
3. Build the bare CNMP (`m0_`) and PE-CNMP (`m1_`), train both (Adam, `lr=3e-4`).
4. Fit the ProMP / ProMP+GMM baselines and produce the comparison plots.

### Outputs
Models, losses, and comparison images are written to:
```
outputs/comparison/mind_change/freq/bare_pe_promp_gmm/<timestamp>/
  saved_models/{bare,pe}.pt
  losses_{bare,pe}.pt
  img/<epoch>_<test_traj>_{bare,pe}.png
```

---

## Notes
- This is the comparison on the pure-synthetic, simple data. The Adroit Hand Hammer, and FluidLab Mixing experiments (sim rollouts,
  matplotlib renders) live under `sim/` and may use separate environments.

### Experiments

##### Synthetic (Simple)

% Will be added soon

##### Synthetic (Complex)

![cnmpVSpemp](https://github.com/user-attachments/assets/20bf474e-5150-411a-aa34-20ac532140c6)

##### Simulation (Adroit Hand Hammer)

<p align="center">
  <img src="https://github.com/user-attachments/assets/bff83218-2b9d-45b2-9d49-972cc4055e2f" width="250">
  <img src="https://github.com/user-attachments/assets/fe97dabd-73b7-4c86-8eef-9be35d37e715" width="250">
  <img src="https://github.com/user-attachments/assets/5ca7a1ed-bb67-48e9-a9b1-d8c58669d549" width="250">
</p>


##### Simulation (Fluidlab Mixing)

<img width="900" height="360" alt="mixing_comparison_low_res" src="https://github.com/user-attachments/assets/1cce3ba8-a7ee-4e1b-9682-e9dcf922f4cd" />

##### Real Robot

% Will be added soon
