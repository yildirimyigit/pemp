"""Generate FluidLab Mixing action trajectories from a fitted WMP (Wavelet Movement
Primitive) baseline, roll them out in the Mixing env, and save matplotlib animations.

Same shape / rollout API as sim/fluidlab_mixing_pemp_test.py and
sim/fluidlab_mixing_bare_test.py, but WMP is fit IN-PROCESS from the run's y.pt/g.pt
(there's no separately saved 'model file' for WMP -- the wavelet/Gaussian fit is
fast and reproducible from the training tensors).

Naming convention for outputs:
    <run>/rollout/mixing_wmp_g<freq>[_ggui]_actions.npy   (real, denormalized)
    <run>/rollout/mixing_wmp_g<freq>[_ggui].mp4

Run in the fluidlab conda env:
    FLUIDLAB_TI_MEM_GB=4.0 PYTHONPATH=$HOME/projects/FluidLab \
      ~/sw/anaconda3/envs/fluidlab/bin/python sim/fluidlab_mixing_wmp_test.py \
        --run outputs/sim/mixing/1779720624_s1
"""
import os
import sys
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "mindchange"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from wmp_baseline import WaveletMovementPrimitive
from fluidlab_mixing_rollout import (
    make_env, rollout_and_render, rollout_and_render_ggui, ACT_LIMIT,
)

ALL_FREQS = [0.333, 0.5, 0.667, 0.833, 1.0]
DEFAULT_RUN = "/home/yigit/projects/pemp/outputs/sim/mixing/1779720624_s1"


def _fit_wmp(run_dir, wavelet="db4"):
    """Fit one WMP on the run's TRAINING tensors, but return the HELD-OUT test tensors
    as the context source (so generate_trajectory conditions on an unseen orientation,
    matching the PEMP/CNMP change).  denorm uses the training normalisation."""
    y = torch.load(os.path.join(run_dir, "y.pt"), map_location="cpu",
                   weights_only=False).float().numpy()                  # (n, T, dy) train
    g = torch.load(os.path.join(run_dir, "g.pt"), map_location="cpu",
                   weights_only=False).float().numpy()
    if g.ndim == 1:
        g = g[:, None]                                                  # (n, dg)
    wmp = WaveletMovementPrimitive(wavelet=wavelet, mode="periodization",
                                   obs_noise=1e-3).fit(y, contexts=g)
    denorm = ACT_LIMIT / float(np.abs(y).max())     # maps the dataset peak -> action range
    # held-out test demos for the conditioning context (unseen orientations)
    y_ctx = torch.load(os.path.join(run_dir, "y_test.pt"), map_location="cpu",
                       weights_only=False).float().numpy()
    g_ctx = torch.load(os.path.join(run_dir, "g_test.pt"), map_location="cpu",
                       weights_only=False).float().numpy().reshape(-1)
    return wmp, y_ctx, g_ctx, denorm


def generate_trajectory(wmp, y_train, g_train, denorm, g_target,
                        n_ctx=10, ctx_seed=None,
                        method="phase_adaptive", batched=True, snap_g=True):
    """WMP action trajectory conditioned on stir frequency g_target.

    Context = `n_ctx` evenly-spaced (default) or seeded-random points lifted from the
    nearest-g training demo, matching what fluidlab_mixing_pemp_test does for PEMP/CNMP.
    snap_g=False conditions on the REQUESTED g_target (off-training-grid evaluation).
    """
    T = y_train.shape[1]
    demo = int(np.argmin(np.abs(g_train - g_target)))
    g_used = float(g_train[demo]) if snap_g else float(g_target)
    # Context window: whole trajectory ("even", default) or just the FIRST CYCLE of the
    # commanded frequency ("initial", via MIX_CTX_MODE=initial).  g = n_loops/6 by convention.
    win = T
    if os.environ.get("MIX_CTX_MODE", "even") == "initial":
        n_loops = max(1, int(round(g_used * 6)))
        win = max(n_ctx, T // n_loops)
    if ctx_seed is None:
        ctx_ids = np.linspace(0, win - 1, n_ctx).round().astype(int)
    else:
        ctx_ids = np.sort(np.random.default_rng(int(ctx_seed)).permutation(win)[:n_ctx])
    t_norm = ctx_ids / float(T - 1)
    y_ctx = y_train[demo, ctx_ids, :]
    pred = wmp.predict(
        context=np.atleast_2d(g_used),
        t_cond=t_norm, y_cond=y_ctx,
        method=method, batched=batched,
    )                                                                   # (T, dy) normalized
    actions = np.clip(pred * denorm, -ACT_LIMIT, ACT_LIMIT).astype(np.float32)
    return actions, g_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--g", type=float, nargs="+", default=ALL_FREQS, help="stir freqs to render")
    ap.add_argument("--n-ctx", type=int, default=5)
    ap.add_argument("--steps", type=int, default=int(os.environ.get("MIX_STEPS", 0)) or None,
                    help="action steps to roll out (default: full t_steps)")
    ap.add_argument("--stride", type=int, default=int(os.environ.get("MIX_STRIDE", 2)))
    ap.add_argument("--render", choices=["matplotlib", "ggui"], default="matplotlib",
                    help="ggui = FluidLab native 3D (needs a Vulkan<=1.3 device)")
    ap.add_argument("--wavelet", default="db4", help="PyWavelets family (default db4)")
    ap.add_argument("--method", choices=["gaussian", "phase_adaptive"],
                    default="phase_adaptive",
                    help="WMP conditioning: paper-faithful 'phase_adaptive' (default) "
                         "or pure Gaussian via-point conditioning")
    ap.add_argument("--no-batched", action="store_true",
                    help="paper-strict per-observation iteration (very slow at high n_ctx)")
    ap.add_argument("--no-snap-g", action="store_true",
                    help="condition on the REQUESTED g instead of snapping to nearest training g "
                         "(use for off-grid evaluation, e.g. g=0.4 or g=0.9)")
    args = ap.parse_args()
    snap_g = not args.no_snap_g
    print(f"[cfg] wavelet={args.wavelet}  method={args.method}  "
          f"batched={not args.no_batched}  n_ctx={args.n_ctx}  render={args.render}")

    out_dir = os.path.join(args.run, "rollout")
    os.makedirs(out_dir, exist_ok=True)
    render_fn = rollout_and_render_ggui if args.render == "ggui" else rollout_and_render
    suffix = "_ggui" if args.render == "ggui" else ""

    wmp, y_train, g_train, denorm = _fit_wmp(args.run, wavelet=args.wavelet)
    print(f"[fit] WMP: n={y_train.shape[0]} demos  T={y_train.shape[1]}  dy={y_train.shape[2]}  "
          f"K_per_dim={wmp.k_per_dim}  denorm={denorm:.5f}")

    # Generate every action trajectory first (cheap once WMP is fit), then build the env once.
    jobs = []
    for g in args.g:
        actions, g_used = generate_trajectory(
            wmp, y_train, g_train, denorm, g,
            n_ctx=args.n_ctx, method=args.method, batched=not args.no_batched,
            snap_g=snap_g,
        )
        if args.steps:
            actions = actions[:args.steps]
        tag = f"mixing_wmp_g{g_used:.3f}".replace(".", "p")
        np.save(os.path.join(out_dir, tag + "_actions.npy"), actions)
        print(f"[gen] g={g_used:.3f}: action range [{actions.min():.4f},{actions.max():.4f}] "
              f"|a| per dim={np.abs(actions).max(0).round(4)}")
        jobs.append((actions, g_used, os.path.join(out_dir, tag + suffix + ".mp4")))

    env = make_env()
    for actions, g_used, out_path in jobs:
        render_fn(env, actions, out_path, stride=args.stride, title=f"WMP g={g_used:.3f}")
    env.close()


if __name__ == "__main__":
    main()
