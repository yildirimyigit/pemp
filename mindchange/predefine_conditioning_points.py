"""Predefine the conditioning protocol per experiment for the bare/PEMP/ProMP
comparison on frequency-conditioned periodic data (synthetic 1D or the more
complex 'combined' set).

Reads each timestamped experiment dir under one of the known base dirs
(`--dataset simple` or `--dataset complex`, or `--base-dir <path>` for any
other), and writes a single self-contained protocol file
  <ts>/conditioning.npz
that any internal or external model will read to (a) get the exact context
inputs to condition on, and (b) get the target indices/values to evaluate against.

Design choices:
  * Use each experiment's existing held-out test set (`x_test.pt`/`y_test.pt`/`g_test.pt`).
  * Random context indices, seeded deterministically per (timestamp, test_traj_idx)
    so every model uses the SAME context -> paired comparison.
  * NESTED across n_ctx values: one random permutation per test trajectory, and
    context_n{N} = perm[:N], target_n{N} = perm[N:].  So  n=3 ⊂ n=10 ⊂ n=30 ⊂ n=100.
    The error-vs-n_ctx curve then reads as "what does each method gain from MORE
    observations" rather than "what does it gain from a different random draw."
  * One npz per experiment with every n_ctx inside (small data, one load downstream).
  * Also saves the actual context_x/y and target_x/y values (not just indices) so
    consumers can load the npz without needing the .pt tensors.

Run (single or many; pick dataset or override base):
  python mindchange/predefine_conditioning_points.py --dataset simple  --timestamp <ts>
  python mindchange/predefine_conditioning_points.py --dataset complex --all
  python mindchange/predefine_conditioning_points.py --base-dir <path> --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Known experiment sets.  Pick one via `--dataset {simple,complex}`, or override the
# path entirely with `--base-dir <path>` (still useful for any future set).
BASE_DIRS = {
    "simple":  Path("/home/yigit/projects/pemp/outputs/comparison/mind_change/freq/bare_pe_promp_gmm"),
    "complex": Path("/home/yigit/projects/pemp/outputs/comparison/mind_change/freq/combined/bare_pe_promp_gmm"),
}
BASE_DIR = BASE_DIRS["simple"]  # default if neither --dataset nor --base-dir is given
DEFAULT_N_CTX = (3, 10, 30, 100)
BASE_SEED = 0  # combined with test_traj_idx for the per-trajectory permutation


def _load_test_tensors(ts_dir: Path):
    """Return (x_test, y_test, g_test) as numpy arrays from <ts>/x_test.pt etc."""
    def _np(name):
        return torch.load(ts_dir / name, map_location="cpu", weights_only=False).numpy()
    x = _np("x_test.pt")     # (n, T, dx)
    y = _np("y_test.pt")     # (n, T, dy)
    g = _np("g_test.pt")     # (n,) or (n, dg)
    if g.ndim == 1:
        g = g[:, None]       # -> (n, dg=1) for a consistent shape
    return x, y, g


def _build_protocol(x_test, y_test, g_test, n_ctx_values, base_seed):
    """Compute nested random context perms + slice x/y at the chosen indices."""
    n, T = x_test.shape[:2]
    perms = np.empty((n, T), dtype=np.int64)
    for i in range(n):
        # seeded per (base_seed, test_traj_idx); identical regardless of n_ctx (nested)
        perms[i] = np.random.default_rng(base_seed + i).permutation(T)

    out: dict[str, np.ndarray] = {
        "t_steps": np.int64(T),
        "num_test": np.int64(n),
        "n_ctx_values": np.asarray(list(n_ctx_values), dtype=np.int64),
        "base_seed": np.int64(base_seed),
        "g": g_test.astype(np.float32),
        "x_test_full": x_test.astype(np.float32),   # full trajectory for convenience
        "y_test_full": y_test.astype(np.float32),
        "notes": np.array(
            "Nested random context per (base_seed + test_traj_idx). "
            "For each n_ctx N: context_indices_n{N} = perm[:N], target_indices_n{N} = perm[N:]. "
            "context_{x,y}_n{N} are the gathered values at context_indices_n{N}; "
            "target_{x,y}_n{N} likewise. g[i] is the frequency-conditioning for traj i."
        ),
    }
    for N in n_ctx_values:
        if N >= T:
            print(f"  warn: n_ctx={N} >= t_steps={T}, skipping")
            continue
        ctx_idx = perms[:, :N]
        tgt_idx = perms[:, N:]
        out[f"context_indices_n{N}"] = ctx_idx
        out[f"target_indices_n{N}"]  = tgt_idx
        out[f"context_x_n{N}"] = np.take_along_axis(x_test, ctx_idx[..., None], axis=1).astype(np.float32)
        out[f"context_y_n{N}"] = np.take_along_axis(y_test, ctx_idx[..., None], axis=1).astype(np.float32)
        out[f"target_x_n{N}"]  = np.take_along_axis(x_test, tgt_idx[..., None], axis=1).astype(np.float32)
        out[f"target_y_n{N}"]  = np.take_along_axis(y_test, tgt_idx[..., None], axis=1).astype(np.float32)
    return out, n, T


def process_timestamp(ts_dir: Path, n_ctx_values, base_seed, out_name="conditioning.npz"):
    x, y, g = _load_test_tensors(ts_dir)
    proto, n, T = _build_protocol(x, y, g, n_ctx_values, base_seed)
    out_path = ts_dir / out_name
    np.savez_compressed(out_path, **proto)
    saved_n_ctx = [N for N in n_ctx_values if N < T]
    print(f"[ok] {ts_dir.name}: wrote {out_path.name}  num_test={n} T={T} "
          f"n_ctx={saved_n_ctx} g_shape={g.shape}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timestamp", nargs="+", help="one or more experiment timestamp dirs")
    ap.add_argument("--all", action="store_true", help="process every dir under --base-dir")
    ap.add_argument("--dataset", choices=sorted(BASE_DIRS), default=None,
                    help="experiment set: simple (1D synthetic) or complex (combined). "
                         "Sets --base-dir if given; --base-dir overrides this if also set.")
    ap.add_argument("--base-dir", default=None,
                    help=f"override base dir (default: simple set = {BASE_DIR})")
    ap.add_argument("--n-ctx", type=int, nargs="+", default=list(DEFAULT_N_CTX),
                    help="nested context sizes; default 3 10 30 100")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--out-name", default="conditioning.npz",
                    help="output filename inside each ts dir (use conditioning_v2.npz "
                         "to keep the old conditioning.npz untouched alongside it)")
    args = ap.parse_args()

    if args.base_dir is not None:
        base = Path(args.base_dir)
    elif args.dataset is not None:
        base = BASE_DIRS[args.dataset]
    else:
        base = BASE_DIR
    print(f"[base] {base}")
    if args.all:
        timestamps = [d.name for d in sorted(base.iterdir()) if d.is_dir()]
    elif args.timestamp:
        timestamps = args.timestamp
    else:
        ap.error("specify --timestamp <ts> [ts ...] or --all")

    for ts in timestamps:
        ts_dir = base / ts
        if not ts_dir.is_dir():
            print(f"[skip] {ts}: not a dir at {ts_dir}"); continue
        try:
            process_timestamp(ts_dir, args.n_ctx, args.seed, args.out_name)
        except FileNotFoundError as e:
            print(f"[skip] {ts}: missing input ({e})")
        except Exception as e:
            print(f"[err]  {ts}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
