"""Generate FluidLab Mixing action trajectories from the trained bare CNMP model
(bare.pt), roll them out, and save matplotlib animations.

Identical to sim/fluidlab_mixing_pemp_test.py except the positional input is the
scalar normalized time x in [0,1] (input_dim = dx+dg) instead of the positional
encoding PE(t).  This is the CNMP baseline PEMP is compared against; on the
mixing stir it tends to under-reconstruct the higher frequencies (an MLP on
scalar time has a spectral bias against multi-period signals).

Run in the `fluidlab` conda env:

  FLUIDLAB_TI_MEM_GB=4.0 PYTHONPATH=$HOME/projects/FluidLab \
    $HOME/sw/anaconda3/envs/fluidlab/bin/python sim/fluidlab_mixing_bare_test.py
"""
import os
import sys
import argparse

import numpy as np
import yaml
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "models"), os.path.join(ROOT, "data"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cnmp import CNMP
from fluidlab_mixing_rollout import make_env, rollout_and_render, rollout_and_render_ggui, ACT_LIMIT

DX, DG = 1, 1             # must match training (dx=1 scalar time, dg=1)
ALL_FREQS = [0.333, 0.5, 0.667, 0.833, 1.0]
DEFAULT_RUN = "/home/yigit/projects/pemp/outputs/sim/mixing/1779529990"


def generate_trajectory(run_dir, g_target, n_ctx, steps, ctx_seed=None):
    """Bare-CNMP action trajectory conditioned on stir frequency g_target.
    ctx_seed=None -> evenly-spaced context; an int -> random context (for eval draws)."""
    with open(os.path.join(run_dir, "hyperparameters.yaml")) as f:
        hp = yaml.safe_load(f)
    t_steps = hp["t_steps"]
    steps = min(steps or t_steps, t_steps)

    y_train = torch.load(os.path.join(run_dir, "y.pt"), map_location="cpu").float()
    g_train = torch.load(os.path.join(run_dir, "g.pt"), map_location="cpu").float().reshape(-1)
    dy = y_train.shape[-1]
    denorm = ACT_LIMIT / float(y_train.abs().max())

    x_time = torch.linspace(0, 1, t_steps).reshape(-1, 1)  # the scalar positional coord
    model = CNMP(input_dim=DX + DG, output_dim=dy, n_max=max(n_ctx, 1), m_max=t_steps,
                 encoder_hidden_dims=hp["enc_dims"], decoder_hidden_dims=hp["dec_dims"],
                 batch_size=1, device="cpu")
    model.load_state_dict(torch.load(os.path.join(run_dir, "saved_models", "bare.pt"),
                                     map_location="cpu", weights_only=False))
    model.eval()

    demo = int((g_train - g_target).abs().argmin())
    g_used = float(g_train[demo])
    if ctx_seed is None:
        ctx_ids = torch.linspace(0, t_steps - 1, n_ctx).round().long()
    else:
        gen = torch.Generator().manual_seed(int(ctx_seed))
        ctx_ids = torch.randperm(t_steps, generator=gen)[:n_ctx].sort().values

    obs = torch.zeros(1, n_ctx, DX + DG + dy)
    obs[0, :, :DX] = x_time[ctx_ids]
    obs[0, :, DX:DX + DG] = g_used
    obs[0, :, DX + DG:] = y_train[demo, ctx_ids]
    obs_mask = torch.ones(1, n_ctx, dtype=torch.bool)

    tar_x = torch.zeros(1, t_steps, DX + DG)
    tar_x[0, :, :DX] = x_time
    tar_x[0, :, DX:] = g_used

    with torch.no_grad():
        pred = model(obs, tar_x, obs_mask)
    actions = np.clip(pred[0, :, :dy].cpu().numpy() * denorm, -ACT_LIMIT, ACT_LIMIT).astype(np.float32)
    print(f"[gen] g={g_used:.3f} (demo {demo}); action range "
          f"[{actions.min():.4f},{actions.max():.4f}]  |a| per dim={np.abs(actions).max(0).round(4)}")
    return actions[:steps], g_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--g", type=float, nargs="+", default=ALL_FREQS, help="stir freqs to render")
    ap.add_argument("--n-ctx", type=int, default=10)
    ap.add_argument("--steps", type=int, default=int(os.environ.get("MIX_STEPS", 0)) or None)
    ap.add_argument("--stride", type=int, default=int(os.environ.get("MIX_STRIDE", 2)))
    ap.add_argument("--render", choices=["matplotlib", "ggui"], default="matplotlib",
                    help="ggui = FluidLab native 3D (needs a Vulkan<=1.3 device; check with ggui_smoke.py)")
    args = ap.parse_args()
    print(f"[cfg] render = {args.render}")

    out_dir = os.path.join(args.run, "rollout")
    os.makedirs(out_dir, exist_ok=True)
    render_fn = rollout_and_render_ggui if args.render == "ggui" else rollout_and_render
    suffix = "_ggui" if args.render == "ggui" else ""

    jobs = []
    for g in args.g:
        actions, g_used = generate_trajectory(args.run, g, args.n_ctx, args.steps)
        tag = f"mixing_bare_g{g_used:.3f}".replace(".", "p")
        np.save(os.path.join(out_dir, tag + "_actions.npy"), actions)
        jobs.append((actions, g_used, os.path.join(out_dir, tag + suffix + ".mp4")))

    env = make_env()
    for actions, g_used, out_path in jobs:
        render_fn(env, actions, out_path, stride=args.stride, title=f"CNMP g={g_used:.3f}")
    env.close()


if __name__ == "__main__":
    main()
