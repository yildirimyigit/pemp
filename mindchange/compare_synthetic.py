"""Compare CNMP / PEMP / WMP / 30 ProMP+GMM variants on a single synthetic-experiment
folder, using the predefined evaluation protocol from <ts>/conditioning.npz.

For each test trajectory i and each n_ctx in the protocol's `n_ctx_values`, every
approach is conditioned on (context_x_n{N}[i], context_y_n{N}[i], g[i]) and asked
to predict at the corresponding target_x_n{N}[i].  MSE is computed on the TARGET
(non-conditioning) points only and aggregated.

Expected layout of <ts>:
  saved_models/bare.pt              -> CNMP
  saved_models/pe.pt                -> PEMP
  saved_models/promp_{0..29}/{promp.pkl, gmm.pkl}  -> 30 ProMP+GMM variants
  x.pt, y.pt, g.pt                   -> training data (WMP fitted on these)
  conditioning.npz                   -> shared eval protocol
                                       (built by predefine_conditioning_points.py)

Run in pemp-gpu (torch + PyWavelets + movement_primitives + gmr):
  ~/sw/anaconda3/envs/pemp-gpu/bin/python mindchange/compare_synthetic.py \\
      --ts outputs/comparison/mind_change/freq/bare_pe_promp_gmm/1776959250
"""
from __future__ import annotations

import os
import sys
import math
import csv
import pickle
import argparse
import re

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "models"), os.path.join(ROOT, "data"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cnmp import CNMP
from positional_encoders import generate_positional_encoding
from wmp_baseline import WaveletMovementPrimitive
from movement_primitives.promp import ProMP  # noqa: F401  (registered for pickle)

# ----- model / data dimensions, matching the synthetic notebook ----- #
DX = 1
DG = 1
DY = 1
DPE = 27
PE_SCALER = 0.2          # notebook uses the default 0.2
ENC_DIMS = [256, 256]    # notebook value
DEC_DIMS = [256, 256]


# ============================================================================
# model loading / fitting
# ============================================================================
def _load_cnmp(state_path, input_dim, dy, m_max):
    m = CNMP(input_dim=input_dim, output_dim=dy, n_max=1, m_max=m_max,
             encoder_hidden_dims=ENC_DIMS, decoder_hidden_dims=DEC_DIMS,
             batch_size=1, device="cpu")
    m.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def _load_promp_gmm_dir(path):
    """Load (promp, gmm) from a saved_models/promp_<k>/ folder."""
    with open(os.path.join(path, "promp.pkl"), "rb") as f:
        promp = pickle.load(f)
    with open(os.path.join(path, "gmm.pkl"), "rb") as f:
        gmm = pickle.load(f)
    return promp, gmm


def _fit_wmp(y_train, g_train, wavelet="db4"):
    """y_train: (n, T, dy); g_train: (n,) or (n, dg)."""
    g = g_train.reshape(-1, 1) if g_train.ndim == 1 else g_train
    return WaveletMovementPrimitive(wavelet=wavelet, mode="periodization",
                                    obs_noise=1e-3).fit(y_train, contexts=g)


# ============================================================================
# per-approach prediction at target points (n_target, dy)
# ============================================================================
def pred_cnmp(model, ctx_x, ctx_y, g_val, target_x):
    n_ctx, n_q = len(ctx_x), len(target_x)
    obs = torch.zeros(1, n_ctx, DX + DG + DY)
    obs[0, :, :DX] = torch.as_tensor(ctx_x, dtype=torch.float32)
    obs[0, :, DX:DX + DG] = float(g_val)
    obs[0, :, DX + DG:] = torch.as_tensor(ctx_y, dtype=torch.float32)
    tar_x = torch.zeros(1, n_q, DX + DG)
    tar_x[0, :, :DX] = torch.as_tensor(target_x, dtype=torch.float32)
    tar_x[0, :, DX:] = float(g_val)
    mask = torch.ones(1, n_ctx, dtype=torch.bool)
    # use .val() not forward(): forward repeats r to fixed self.m_max so it can't
    # accept variable-length target sequences; .val() uses tar.shape[1] directly.
    return model.val(obs, tar_x, mask)[0, :, :DY].cpu().numpy()


def pred_pemp(model, pe, ctx_idx, ctx_y, g_val, target_idx):
    n_ctx, n_q = len(ctx_idx), len(target_idx)
    obs = torch.zeros(1, n_ctx, DPE + DG + DY)
    obs[0, :, :DPE] = pe[ctx_idx]
    obs[0, :, DPE:DPE + DG] = float(g_val)
    obs[0, :, DPE + DG:] = torch.as_tensor(ctx_y, dtype=torch.float32)
    tar_x = torch.zeros(1, n_q, DPE + DG)
    tar_x[0, :, :DPE] = pe[target_idx]
    tar_x[0, :, DPE:] = float(g_val)
    mask = torch.ones(1, n_ctx, dtype=torch.bool)
    return model.val(obs, tar_x, mask)[0, :, :DY].cpu().numpy()


def pred_wmp(wmp, ctx_x, ctx_y, g_val, target_idx):
    """Return WMP prediction at target indices only.  WMP returns the full
    trajectory; we slice it at the target indices."""
    full = wmp.predict(
        context=np.atleast_2d(g_val).reshape(1, -1),
        t_cond=np.asarray(ctx_x).reshape(-1),
        y_cond=np.asarray(ctx_y).reshape(len(ctx_x), DY),
        method="phase_adaptive", batched=True,
    )  # (T, dy)
    return full[np.asarray(target_idx, dtype=np.int64)]


PROMP_OBS_NOISE = 1e-3   # y_cov for position conditioning (see note in pred_promp_gmm)


def pred_promp_gmm(promp, gmm, ctx_x, ctx_y, g_val, target_x):
    """ProMP+GMM: 1) condition GMM on g -> conditional weight distribution.
                  2) condition the ProMP on each observation point.
                  3) return mean_trajectory at the target times.

    NOTE: ProMP.condition_position RETURNS a new conditioned ProMP; it does *not*
    mutate in place.  Each step must rebind `cp` -- the previous version discarded
    the return value, so the context points had zero effect (ProMP collapsed to the
    GMM prior mean, ~0.48 MSE regardless of n_ctx).  A small observation noise
    (PROMP_OBS_NOISE) regularizes the otherwise ill-conditioned sequential updates;
    with zero noise the posterior blows up (|y| ~ 1e2) once several points are added."""
    g_q = np.atleast_2d(float(g_val)).reshape(1, -1)  # (1, dg)
    cond = gmm.condition(np.arange(g_q.shape[1]), g_q).to_mvn()
    promp.from_weight_distribution(cond.mean, cond.covariance)
    cp = promp
    y_cov = np.array([[PROMP_OBS_NOISE]])
    for t_c, y_c in zip(np.asarray(ctx_x).reshape(-1).tolist(),
                        np.asarray(ctx_y).reshape(-1).tolist()):
        cp = cp.condition_position(np.array([float(y_c)]), y_cov=y_cov,
                                   t=float(t_c), t_max=1.0)
    return cp.mean_trajectory(np.asarray(target_x).reshape(-1)).reshape(-1, DY)


# ============================================================================
# evaluation
# ============================================================================
def _approach_list(saved_models_dir):
    """Build the ordered list of (label, kind, payload) approaches.  Payload is
    a model for cnmp/pemp/wmp and (promp, gmm) for each promp_N."""
    items = []
    if os.path.exists(os.path.join(saved_models_dir, "bare.pt")):
        items.append(("CNMP", "cnmp", os.path.join(saved_models_dir, "bare.pt")))
    if os.path.exists(os.path.join(saved_models_dir, "pe.pt")):
        items.append(("PEMP", "pemp", os.path.join(saved_models_dir, "pe.pt")))
    items.append(("WMP", "wmp", None))                  # fit later
    promp_dirs = sorted(
        (d for d in os.listdir(saved_models_dir) if d.startswith("promp_")),
        key=lambda d: int(re.search(r"promp_(\d+)", d).group(1)),
    )
    for d in promp_dirs:
        items.append((f"ProMP_{d.split('_')[1]}", "promp",
                      os.path.join(saved_models_dir, d)))
    return items


def evaluate(ts_dir, eval_subset=None, cond_file="conditioning.npz"):
    """Evaluate every approach against every (n_ctx, test_traj) in the protocol file.
    Returns a list of sample dicts and the n_ctx_values list.

    `cond_file` is the protocol filename inside <ts> (default conditioning.npz;
    pass conditioning_v2.npz to run against the alternate {1,3,10,20} protocol)."""
    ts_dir = os.path.abspath(ts_dir)
    sm = os.path.join(ts_dir, "saved_models")
    # ---- training data (for WMP) ----
    y_train = torch.load(os.path.join(ts_dir, "y.pt"), map_location="cpu",
                         weights_only=False).float().numpy()  # (n, T, 1)
    g_train = torch.load(os.path.join(ts_dir, "g.pt"), map_location="cpu",
                         weights_only=False).float().numpy().reshape(-1)
    t_steps = y_train.shape[1]
    pe = generate_positional_encoding(t_steps, DPE, PE_SCALER)   # (T, dpe)
    print(f"[data] t_steps={t_steps}  train y{y_train.shape}  g unique={sorted(set(g_train.round(3)))[:6]}")

    # ---- protocol ----
    proto = np.load(os.path.join(ts_dir, cond_file), allow_pickle=False)
    n_ctx_values = [int(v) for v in proto["n_ctx_values"]]
    num_test = int(proto["num_test"])
    g_test = proto["g"]                                         # (num_test, dg)
    print(f"[proto] n_ctx_values={n_ctx_values}  num_test={num_test}")

    # ---- approaches ----
    approaches = _approach_list(sm)
    if eval_subset is not None:
        keep = set(eval_subset)
        approaches = [a for a in approaches if a[0] in keep or a[1] in keep]
    print(f"[approaches] {len(approaches)} models: "
          f"{[a[0] for a in approaches[:6]] + (['...'] if len(approaches) > 6 else [])}")

    # ---- load / fit ----
    loaded = {}
    for label, kind, payload in approaches:
        if kind == "cnmp":
            loaded[label] = ("cnmp", _load_cnmp(payload, DX + DG, DY, t_steps))
        elif kind == "pemp":
            loaded[label] = ("pemp", _load_cnmp(payload, DPE + DG, DY, t_steps))
        elif kind == "wmp":
            loaded[label] = ("wmp", _fit_wmp(y_train, g_train))
        elif kind == "promp":
            loaded[label] = ("promp", _load_promp_gmm_dir(payload))
    print(f"[loaded] {len(loaded)} approaches ready")

    # ---- evaluate ----
    samples = []
    for N in n_ctx_values:
        ctx_idx_all = proto[f"context_indices_n{N}"]            # (num_test, N)
        tgt_idx_all = proto[f"target_indices_n{N}"]             # (num_test, T-N)
        ctx_x_all = proto[f"context_x_n{N}"]                    # (num_test, N, dx)
        ctx_y_all = proto[f"context_y_n{N}"]                    # (num_test, N, dy)
        tgt_x_all = proto[f"target_x_n{N}"]                     # (num_test, T-N, dx)
        tgt_y_all = proto[f"target_y_n{N}"]                     # (num_test, T-N, dy) ground truth
        for i in range(num_test):
            ctx_x = ctx_x_all[i].reshape(-1)                    # (N,)
            ctx_y = ctx_y_all[i].reshape(-1, DY)                # (N, dy)
            ctx_idx = ctx_idx_all[i]
            tgt_idx = tgt_idx_all[i]
            tgt_x = tgt_x_all[i].reshape(-1)
            tgt_y = tgt_y_all[i].reshape(-1, DY)
            g_i = float(g_test[i].reshape(-1)[0])
            for label, (kind, m) in loaded.items():
                try:
                    if kind == "cnmp":
                        pred = pred_cnmp(m, ctx_x.reshape(-1, DX), ctx_y, g_i,
                                          tgt_x.reshape(-1, DX))
                    elif kind == "pemp":
                        pred = pred_pemp(m, pe, ctx_idx, ctx_y, g_i, tgt_idx)
                    elif kind == "wmp":
                        pred = pred_wmp(m, ctx_x, ctx_y, g_i, tgt_idx)
                    elif kind == "promp":
                        promp, gmm = m
                        pred = pred_promp_gmm(promp, gmm, ctx_x, ctx_y, g_i, tgt_x)
                    else:
                        continue
                    mse = float(np.mean((pred - tgt_y) ** 2))
                except Exception as e:
                    mse = float("nan")
                    print(f"  ! {label} n={N} traj={i}: {type(e).__name__}: {str(e)[:120]}")
                samples.append(dict(approach=label, n_ctx=N, traj=i, mse=mse,
                                     timestamp=os.path.basename(ts_dir)))
        print(f"[done] n_ctx={N}: {num_test} test trajs x {len(loaded)} approaches")
    return samples, n_ctx_values


# ============================================================================
# aggregation + table writing
# ============================================================================
def aggregate_by_approach_nctx(samples):
    """Group samples by (approach, n_ctx); return rows with mean/std MSE."""
    groups = {}
    for s in samples:
        groups.setdefault((s["approach"], s["n_ctx"]), []).append(s["mse"])
    rows = []
    for (label, N), vs in groups.items():
        v = np.array([x for x in vs if not (isinstance(x, float) and math.isnan(x))], float)
        rows.append({
            "approach": label, "n_ctx": N, "n": len(v),
            "mse_mean": float(v.mean()) if len(v) else float("nan"),
            "mse_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        })
    rows.sort(key=lambda r: (r["approach"], r["n_ctx"]))
    return rows


def _print_table(rows, n_ctx_values, samples):
    """Wide table: rows = approach, cols = n_ctx; cells = mean+-std."""
    by = {(r["approach"], r["n_ctx"]): r for r in rows}
    approaches_seen = sorted({r["approach"] for r in rows}, key=_sort_key)
    head = ["approach", "n_demos"] + [f"n={N}" for N in n_ctx_values]
    cells = []
    for a in approaches_seen:
        row = [a, str(by[(a, n_ctx_values[0])]["n"])]
        for N in n_ctx_values:
            r = by.get((a, N), None)
            if r is None or math.isnan(r["mse_mean"]):
                row.append("-")
            elif r["n"] > 1:
                row.append(f"{r['mse_mean']:.4f}+-{r['mse_std']:.4f}")
            else:
                row.append(f"{r['mse_mean']:.4f}")
        cells.append(row)
    widths = [max(len(head[i]), max(len(r[i]) for r in cells)) for i in range(len(head))]
    line = "  ".join(f"{head[i]:>{widths[i]}}" for i in range(len(head)))
    print("\n=== Test-target MSE: mean+-std over test trajectories per (approach, n_ctx) ===")
    print(line)
    print("-" * len(line))
    for r in cells:
        print("  ".join(f"{r[i]:>{widths[i]}}" for i in range(len(head))))


def _sort_key(label):
    """Approach ordering: CNMP, PEMP, WMP, ProMP_<k> by k."""
    if label == "CNMP": return (0,)
    if label == "PEMP": return (1,)
    if label == "WMP":  return (2,)
    m = re.match(r"ProMP_(\d+)", label)
    return (3, int(m.group(1))) if m else (4, label)


def _write_csv(rows, path, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"[out] wrote {path}")


def _write_latex_byapproach_table(rows, n_ctx_values, path):
    by = {(r["approach"], r["n_ctx"]): r for r in rows}
    approaches_seen = sorted({r["approach"] for r in rows}, key=_sort_key)
    head = ["Approach"] + [f"$n_{{ctx}}={N}$" for N in n_ctx_values]
    lines = ["\\begin{tabular}{l" + "r" * len(n_ctx_values) + "}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]
    for a in approaches_seen:
        cells = [a]
        for N in n_ctx_values:
            r = by.get((a, N), None)
            if r is None or math.isnan(r["mse_mean"]):
                cells.append("-")
            elif r["n"] > 1:
                cells.append(f"${r['mse_mean']:.4f}{{\\scriptstyle\\,\\pm {r['mse_std']:.4f}}}$")
            else:
                cells.append(f"${r['mse_mean']:.4f}$")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[out] wrote {path}")


# ============================================================================
# CLI
# ============================================================================
def tag_from_cond(cond_file):
    """`conditioning.npz` -> ``; `conditioning_v2.npz` -> `_v2`; arbitrary names
    fall back to `_<basename>`.  Used to namespace per-protocol output files so
    v1 and v2 artifacts coexist in the same <ts> folder."""
    base = os.path.splitext(os.path.basename(cond_file))[0]
    if base == "conditioning":
        return ""
    if base.startswith("conditioning_"):
        return base[len("conditioning"):]   # leaves the leading underscore in place
    return "_" + base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True, help="path to a single experiment timestamp folder")
    ap.add_argument("--cond-file", default="conditioning.npz",
                    help="protocol filename inside <ts> (e.g. conditioning_v2.npz "
                         "for the {1,3,10,20} protocol)")
    ap.add_argument("--out-prefix", default=None,
                    help="prefix for output files; default: <ts>/comparison_synth{tag} "
                         "where tag comes from the cond filename")
    ap.add_argument("--approaches", nargs="+", default=None,
                    help="optional subset (labels like CNMP PEMP WMP ProMP_0 ...)")
    args = ap.parse_args()

    samples, n_ctx_values = evaluate(args.ts, eval_subset=args.approaches,
                                     cond_file=args.cond_file)
    rows = aggregate_by_approach_nctx(samples)
    _print_table(rows, n_ctx_values, samples)
    tag = tag_from_cond(args.cond_file)
    prefix = args.out_prefix or os.path.join(args.ts, f"comparison_synth{tag}")
    _write_csv(samples, prefix + "_samples.csv",
               ["timestamp", "approach", "n_ctx", "traj", "mse"])
    _write_csv(rows, prefix + "_byapproach.csv",
               ["approach", "n_ctx", "n", "mse_mean", "mse_std"])
    _write_latex_byapproach_table(rows, n_ctx_values, prefix + "_byapproach.tex")


if __name__ == "__main__":
    main()
