"""Render hostile-mixing demonstrations as FluidLab particle-mixing videos.

Feeds each saved demo's actions through the Mixing-v0 sim and renders the top-down
coffee/milk particle scatter (same renderer the *_test.py rollouts use), one mp4 per
demo -- i.e. the videos like the CNMP/PEMP rollout frames, not an action plot.

MUST run in the `fluidlab` conda env (taichi sim).  Build ONE env and reset between
demos (taichi accumulates GPU fields across gym.make calls -> OOM).  Example:

  FLUIDLAB_TI_MEM_GB=4.0 PYTHONPATH=$HOME/projects/FluidLab \\
    ~/sw/anaconda3/envs/fluidlab/bin/python sim/data/viz_hostile_demos.py --seed 1

Options:
  --indices 0 12 24    explicit demo indices (default: first demo of each g)
  --out-dir <dir>      default sim/data/fluidlab_mixing_hostile_s<seed>/videos
  --fps 25  --stride 2 --subsample 5     render cadence (matplotlib renderer)
  --render matplotlib|ggui               GGUI needs a Vulkan<=1.3 device (see memory)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent          # sim/data
sys.path.insert(0, str(BASE.parent))             # sim/  (fluidlab_mixing_rollout lives here)


def dataset_dir(seed: int) -> Path:
    return BASE / ("fluidlab_mixing_hostile" if seed == 0 else
                   f"fluidlab_mixing_hostile_s{seed}")


def list_raw(d: Path):
    return sorted(d.glob("raw/*.npz"), key=lambda p: int(p.stem))


def default_indices(raw_files):
    """First demo index of each distinct g (sorted by g)."""
    by_g = {}
    for p in raw_files:
        g = round(float(np.load(p)["g"]), 4)
        by_g.setdefault(g, int(p.stem))
    return [by_g[g] for g in sorted(by_g)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="demo indices to render (default: first demo of each g)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=2, help="capture every Nth sim step")
    ap.add_argument("--subsample", type=int, default=5, help="particle subsampling (matplotlib)")
    ap.add_argument("--render", choices=["matplotlib", "ggui"], default="matplotlib")
    args = ap.parse_args()

    d = dataset_dir(args.seed)
    raw = list_raw(d)
    if not raw:
        raise SystemExit(f"no demos at {d}/raw -- run make_fluidlab_mixing_hostile_dataset.py "
                         f"--seed {args.seed} first")
    out_dir = Path(args.out_dir) if args.out_dir else (d / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = args.indices if args.indices is not None else default_indices(raw)
    by_stem = {int(p.stem): p for p in raw}

    # import the FluidLab rollout tooling (needs the fluidlab env on PYTHONPATH)
    from fluidlab_mixing_rollout import make_env, rollout_and_render, rollout_and_render_ggui
    render_fn = rollout_and_render_ggui if args.render == "ggui" else rollout_and_render

    print(f"[viz] {d.name}: rendering {len(idxs)} demo(s) via FluidLab sim -> {out_dir}")
    env = make_env()                                      # ONE env, reused per demo
    for i in idxs:
        if i not in by_stem:
            print(f"  [skip] index {i}: no raw/{i}.npz"); continue
        z = np.load(by_stem[i])
        actions = z["actions"].astype(np.float32)
        g = float(z["g"]); n = int(z["n_loops"])
        out = str(out_dir / f"hostile_s{args.seed}_demo{i}_g{g:.2f}_n{n}.mp4")
        title = f"demo{i}  g={g:.2f}  n={n}"
        kw = dict(stride=args.stride, fps=args.fps, title=title)
        if args.render == "matplotlib":
            kw["subsample"] = args.subsample
        saved = render_fn(env, actions, out, **kw)
        print(f"  [out] {saved}")


if __name__ == "__main__":
    main()
