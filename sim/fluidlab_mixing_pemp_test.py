"""Generate FluidLab Mixing action trajectories from a trained PEMP (PE-CNMP) model,
roll them out in the Mixing env, and save matplotlib animations of the latte.

Mirrors sim/adroit_pemp_tap_test.py (load model -> build conditioned context ->
predict the whole trajectory -> step the env), but the Mixing env lives in the
FluidLab repo / `fluidlab` conda env, so run it there:

  FLUIDLAB_TI_MEM_GB=4.0 PYTHONPATH=$HOME/projects/FluidLab \
    $HOME/sw/anaconda3/envs/fluidlab/bin/python sim/fluidlab_mixing_pemp_test.py

By default it renders all five stir frequencies, reusing one env (reset between
each).  Pass --g to pick specific ones.  The bare CNMP counterpart that loads
bare.pt is sim/fluidlab_mixing_bare_test.py.

Smoke test (few steps) :  MIX_STEPS=20 MIX_STRIDE=1 ... python ... --g 1.0
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
from positional_encoders import generate_positional_encoding
from fluidlab_mixing_rollout import make_env, rollout_and_render, rollout_and_render_ggui, ACT_LIMIT

DPE, DG = 27, 1            # must match training (notebook: dpe=27, dg=1)
PE_FREQ_SCALER = 0.2       # must match training (bigR run 1779633078 trained with 0.2)
ALL_FREQS = [0.333, 0.5, 0.667, 0.833, 1.0]
DEFAULT_RUN = "/home/yigit/projects/pemp/outputs/sim/mixing/1779529990"


def generate_trajectory(run_dir, g_target, n_ctx, steps, pe_scaler=PE_FREQ_SCALER, ctx_seed=None):
    """PEMP (PE-CNMP) action trajectory conditioned on stir frequency g_target.
    ctx_seed=None -> evenly-spaced context; an int -> random context (for eval draws)."""
    with open(os.path.join(run_dir, "hyperparameters.yaml")) as f:
        hp = yaml.safe_load(f)
    t_steps = hp["t_steps"]
    steps = min(steps or t_steps, t_steps)

    y_train = torch.load(os.path.join(run_dir, "y.pt"), map_location="cpu").float()
    g_train = torch.load(os.path.join(run_dir, "g.pt"), map_location="cpu").float().reshape(-1)
    dy = y_train.shape[-1]
    denorm = ACT_LIMIT / float(y_train.abs().max())  # maps dataset peak -> action range

    pe = generate_positional_encoding(t_steps, DPE, pe_scaler)
    model = CNMP(input_dim=DPE + DG, output_dim=dy, n_max=max(n_ctx, 1), m_max=t_steps,
                 encoder_hidden_dims=hp["enc_dims"], decoder_hidden_dims=hp["dec_dims"],
                 batch_size=1, device="cpu")
    model.load_state_dict(torch.load(os.path.join(run_dir, "saved_models", "pe.pt"),
                                     map_location="cpu", weights_only=False))
    model.eval()

    demo = int((g_train - g_target).abs().argmin())  # context from nearest-g demo
    g_used = float(g_train[demo])
    if ctx_seed is None:
        ctx_ids = torch.linspace(0, t_steps - 1, n_ctx).round().long()
    else:
        gen = torch.Generator().manual_seed(int(ctx_seed))
        ctx_ids = torch.randperm(t_steps, generator=gen)[:n_ctx].sort().values

    obs = torch.zeros(1, n_ctx, DPE + DG + dy)
    obs[0, :, :DPE] = pe[ctx_ids]
    obs[0, :, DPE:DPE + DG] = g_used
    obs[0, :, DPE + DG:] = y_train[demo, ctx_ids]
    obs_mask = torch.ones(1, n_ctx, dtype=torch.bool)

    tar_x = torch.zeros(1, t_steps, DPE + DG)
    tar_x[0, :, :DPE] = pe
    tar_x[0, :, DPE:] = g_used

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
    ap.add_argument("--pe-scaler", type=float, default=PE_FREQ_SCALER,
                    help="PE frequency_scaler -- MUST match training (not stored in hyperparameters.yaml)")
    ap.add_argument("--render", choices=["matplotlib", "ggui"], default="matplotlib",
                    help="ggui = FluidLab native 3D (needs a Vulkan<=1.3 device; check with ggui_smoke.py)")
    args = ap.parse_args()
    print(f"[cfg] pe frequency_scaler = {args.pe_scaler} (must match training); render = {args.render}")

    out_dir = os.path.join(args.run, "rollout")
    os.makedirs(out_dir, exist_ok=True)
    render_fn = rollout_and_render_ggui if args.render == "ggui" else rollout_and_render
    suffix = "_ggui" if args.render == "ggui" else ""

    # generate every trajectory first (cheap, torch CPU), then build the env once
    jobs = []
    for g in args.g:
        actions, g_used = generate_trajectory(args.run, g, args.n_ctx, args.steps, args.pe_scaler)
        tag = f"mixing_pe_g{g_used:.3f}".replace(".", "p")
        np.save(os.path.join(out_dir, tag + "_actions.npy"), actions)
        jobs.append((actions, g_used, os.path.join(out_dir, tag + suffix + ".mp4")))

    env = make_env()
    for actions, g_used, out_path in jobs:
        render_fn(env, actions, out_path, stride=args.stride, title=f"PEMP g={g_used:.3f}")
    env.close()


if __name__ == "__main__":
    main()
