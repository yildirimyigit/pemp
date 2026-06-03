"""Generate Adroit hammer action trajectories from PEMP / CNMP / WMP, each
conditioned on the INITIAL POINT of every test demonstration in the dataset.

Inputs:
  --run   training run with saved_models/{pe,bare}.pt + hyperparameters.yaml
  --data  processed dataset folder with {x,y,g}{,_test}.npy

For each test demo i:
  context  = single (t=0, y_test[i, 0]) point + g_target = g_test[i]
  output   = predicted (T, 26) action trajectory clipped to [-1, 1]
  saved to <run>/adroit_rollout/<filekey>_n<strikes>.npz
           with `actions` (T, 26) and `num_strikes` = round(g_target * MAX_LOOPS),
           ready for sim/adroit_hammer_compare.py.

Run in pemp-gpu (needs torch + PyWavelets):
  ~/sw/anaconda3/envs/pemp-gpu/bin/python sim/adroit_hammer_generate.py \\
      --run outputs/sim/adroit/bare_pe/1779883452_s1 \\
      --data sim/data/clean_taps_s1/processed
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "models"), os.path.join(ROOT, "data"),
          os.path.join(ROOT, "mindchange")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cnmp import CNMP
from positional_encoders import generate_positional_encoding
from wmp_baseline import WaveletMovementPrimitive

DEFAULT_RUN = "/home/yigit/projects/pemp/outputs/sim/adroit/bare_pe/1779883452_s1"
DEFAULT_DATA = "/home/yigit/projects/pemp/sim/data/clean_taps_s1/processed"
MAX_LOOPS = 6
DPE = 27
DG = 1
DX = 1


# ----------------------------- model loaders --------------------------------- #
def _load_cnmp(state_path, input_dim, dy, hp):
    model = CNMP(input_dim=input_dim, output_dim=dy, n_max=1, m_max=hp["t_steps"],
                 encoder_hidden_dims=hp["enc_dims"], decoder_hidden_dims=hp["dec_dims"],
                 batch_size=1, device="cpu")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    model.eval()
    return model


# ----------------------------- trajectory generators ------------------------- #
def gen_pe(model, pe_basis, g_target, y0, t_steps, dy):
    """PEMP: condition on (PE(t=0), g, y0); query PE for all t."""
    obs = torch.zeros(1, 1, DPE + DG + dy)
    obs[0, 0, :DPE] = pe_basis[0]
    obs[0, 0, DPE:DPE + DG] = float(g_target)
    obs[0, 0, DPE + DG:] = torch.as_tensor(y0, dtype=torch.float32)
    tar_x = torch.zeros(1, t_steps, DPE + DG)
    tar_x[0, :, :DPE] = pe_basis
    tar_x[0, :, DPE:] = float(g_target)
    mask = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        return model(obs, tar_x, mask)[0, :, :dy].cpu().numpy()


def gen_bare(model, x_grid_t, g_target, y0, t_steps, dy):
    """Bare CNMP: condition on (x=0, g, y0); query x in [0,1] for all t."""
    obs = torch.zeros(1, 1, DX + DG + dy)
    obs[0, 0, :DX] = x_grid_t[0]
    obs[0, 0, DX:DX + DG] = float(g_target)
    obs[0, 0, DX + DG:] = torch.as_tensor(y0, dtype=torch.float32)
    tar_x = torch.zeros(1, t_steps, DX + DG)
    tar_x[0, :, :DX] = x_grid_t
    tar_x[0, :, DX:] = float(g_target)
    mask = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        return model(obs, tar_x, mask)[0, :, :dy].cpu().numpy()


def gen_wmp(wmp, g_target, y0):
    """WMP phase-adaptive prediction with the single (t=0, y0) observation."""
    return wmp.predict(
        context=np.atleast_2d(g_target),
        t_cond=[0.0],
        y_cond=np.asarray(y0, dtype=float).reshape(1, -1),
        method="phase_adaptive", batched=True,
    )


# ----------------------------------- main ------------------------------------ #
def _parse_approaches(items):
    return dict(it.split("=", 1) for it in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--pe-scaler", type=float, default=0.2,
                    help="PE frequency_scaler -- must match training "
                         "(Adroit notebook uses the default 0.2)")
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--approaches", nargs="+",
                    default=["PEMP=pe", "CNMP=bare", "WMP=wmp"])
    args = ap.parse_args()

    hp = yaml.safe_load(open(os.path.join(args.run, "hyperparameters.yaml")))
    t_steps = int(hp["t_steps"])
    approaches = _parse_approaches(args.approaches)
    print(f"[cfg] run={os.path.basename(args.run)}  t_steps={t_steps}  pe_scaler={args.pe_scaler}  "
          f"approaches={approaches}")

    # Data (processed/)
    y_train = np.load(os.path.join(args.data, "y.npy"))       # (n_train, T, dy)
    g_train = np.load(os.path.join(args.data, "g.npy"))       # (n_train,)
    y_test = np.load(os.path.join(args.data, "y_test.npy"))   # (n_test, T, dy)
    g_test = np.load(os.path.join(args.data, "g_test.npy"))   # (n_test,)
    dy = int(y_train.shape[-1])
    if y_train.shape[1] != t_steps or y_test.shape[1] != t_steps:
        raise RuntimeError(f"t_steps mismatch: hp={t_steps}, data y_train T={y_train.shape[1]}")
    print(f"[data] train y{y_train.shape}  test y{y_test.shape}  dy={dy}  "
          f"g_test={[round(float(g), 3) for g in g_test]}")

    pe_basis = generate_positional_encoding(t_steps, DPE, args.pe_scaler)  # (T, dpe)
    x_grid_t = torch.linspace(0.0, 1.0, t_steps).reshape(-1, 1)            # (T, dx)

    # Load / fit models lazily per approach
    models = {}
    if "pe" in approaches.values():
        models["pe"] = _load_cnmp(os.path.join(args.run, "saved_models", "pe.pt"),
                                  input_dim=DPE + DG, dy=dy, hp=hp)
        print(f"[load] PE model: {sum(p.numel() for p in models['pe'].parameters())} params")
    if "bare" in approaches.values():
        models["bare"] = _load_cnmp(os.path.join(args.run, "saved_models", "bare.pt"),
                                    input_dim=DX + DG, dy=dy, hp=hp)
        print(f"[load] bare model: {sum(p.numel() for p in models['bare'].parameters())} params")
    if "wmp" in approaches.values():
        models["wmp"] = WaveletMovementPrimitive(
            wavelet=args.wavelet, mode="periodization", obs_noise=1e-3,
        ).fit(y_train, contexts=g_train)
        print(f"[fit ] WMP: K_per_dim={models['wmp'].k_per_dim}  "
              f"total_K={models['wmp'].k_per_dim * dy}")

    out_dir = os.path.join(args.run, "adroit_rollout")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(g_test)):
        g_t = float(g_test[i])
        strikes = int(round(g_t * MAX_LOOPS))
        y0 = np.asarray(y_test[i, 0, :], dtype=np.float32)
        print(f"[gen ] test demo {i}  g={g_t:.3f}  strikes={strikes}  "
              f"||y0||={float(np.linalg.norm(y0)):.3f}")
        for label, key in approaches.items():
            if key == "pe":
                a = gen_pe(models["pe"], pe_basis, g_t, y0, t_steps, dy)
            elif key == "bare":
                a = gen_bare(models["bare"], x_grid_t, g_t, y0, t_steps, dy)
            elif key == "wmp":
                a = gen_wmp(models["wmp"], g_t, y0)
            else:
                raise ValueError(f"unknown approach key {key!r}")
            a = np.clip(a, -1.0, 1.0).astype(np.float32)
            out_path = os.path.join(out_dir, f"{key}_n{strikes}.npz")
            np.savez_compressed(out_path, actions=a,
                                num_strikes=np.int64(strikes),
                                g=np.float32(g_t))
            rel = os.path.relpath(out_path, args.run)
            print(f"        wrote {rel}  shape={a.shape}  "
                  f"range[{float(a.min()):.3f},{float(a.max()):.3f}]")


if __name__ == "__main__":
    main()
