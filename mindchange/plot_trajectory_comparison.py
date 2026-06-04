"""Trajectory-reconstruction comparison plots (one Simple, one Complex), styled to
match test_bare_pe_promp_gmm_on_complex_random_with_freq.ipynb (cell 10).

Recipe (per dataset):
  * pick one training run,
  * pick one test trajectory from y_test (default: g closest to 0.6),
  * take the n_ctx=20 context from that run's conditioning_v2.npz,
  * reconstruct the FULL trajectory with CNMP, PEMP, best-ProMP+GMM, WMP,
  * overlay ground truth + the 20 condition points, save as SVG.

Predictions reuse compare_synthetic.py's functions verbatim, so the curves are the
same computation as the comparison tables (correct fixed ProMP conditioning, WMP
fit on the run's training data).  The best ProMP variant is read from the base's
comparison_synth_v2_aggregated_byapproach.csv (same selection as the tables).

Colours (paper-canonical): GT=gray(alpha), PEMP=orange, CNMP=red, ProMP=blue,
WMP=green.  Run:
  ~/sw/anaconda3/envs/pemp-gpu/bin/python mindchange/plot_trajectory_comparison.py
  # optional overrides:
  #   --simple-run <dir> --complex-run <dir> --simple-traj <i> --complex-traj <i>
  #   --n-ctx 20 --out-dir <dir> --target-g 0.6
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_synthetic as cs   # reuse the exact prediction code used by the tables

# ---- style (from the notebook cell 10 + requested colour mapping) ----
COLORS = {
    "GT":    "#000000",   # black, drawn semi-transparent
    "PEMP":  "#E8420E",   # orange (our method)
    "CNMP":  "#0072B2",   # red
    "ProMP": "#9B59B6",   # blue
    "WMP":   "#00B3C4",   # vibrant green
}
LINESTYLES = {
    "GT":    "-",
    "PEMP":  "--",
    "CNMP":  (0, (1, 1)),
    "ProMP": (0, (3, 1, 1, 1, 1, 1)),
    "WMP":   "--",
}
LINEWIDTHS = {"GT": 1.0, "PEMP": 2.5, "CNMP": 2.0, "ProMP": 2.0, "WMP": 2.0}

DEF_SIMPLE = "outputs/comparison/mind_change/sawtooth/1780303718"
DEF_COMPLEX = "outputs/comparison/mind_change/freq/combined/bare_pe_promp_gmm/1777148322"
AGG_CSV = "comparison_synth_v2_aggregated_byapproach.csv"


def best_promp_dirname(base: Path) -> str:
    """Lowest mean-MSE-across-n_ctx ProMP_<k> from the base's aggregated CSV
    (identical selection to make_v2_tables.py).  Returns 'promp_<k>'."""
    means: dict[str, list[float]] = {}
    with (base / AGG_CSV).open() as f:
        for r in csv.DictReader(f):
            if r["approach"].startswith("ProMP_") and r["mse_mean"] not in ("", "nan"):
                means.setdefault(r["approach"], []).append(float(r["mse_mean"]))
    best = min(means, key=lambda k: sum(means[k]) / len(means[k]))
    return f"promp_{best.split('_')[1]}"


def pick_traj(g_arr: np.ndarray, target_g: float, override) -> int:
    if override is not None:
        return int(override)
    return int(np.argmin(np.abs(g_arr.reshape(-1) - target_g)))


def load_run(run_dir: Path, target_g: float, traj_override):
    """Load models + WMP + best ProMP once for a run, pick the example trajectory.
    Returns a context dict reused across n_ctx values (nested contexts share the run)."""
    proto = np.load(run_dir / "conditioning_v2.npz", allow_pickle=False)
    g_test = proto["g"].reshape(-1)
    traj = pick_traj(g_test, target_g, traj_override)
    g_val = float(g_test[traj])
    x_full = proto["x_test_full"][traj].reshape(-1)        # (T,)
    y_full = proto["y_test_full"][traj].reshape(-1)        # (T,)
    T = x_full.shape[0]

    pe = cs.generate_positional_encoding(T, cs.DPE, cs.PE_SCALER)
    sm = run_dir / "saved_models"
    cnmp = cs._load_cnmp(sm / "bare.pt", cs.DX + cs.DG, cs.DY, T)
    pemp = cs._load_cnmp(sm / "pe.pt", cs.DPE + cs.DG, cs.DY, T)

    y_train = torch.load(run_dir / "y.pt", map_location="cpu",
                         weights_only=False).float().numpy()
    g_train = torch.load(run_dir / "g.pt", map_location="cpu",
                         weights_only=False).float().numpy().reshape(-1)
    wmp = cs._fit_wmp(y_train, g_train)

    promp_dir = best_promp_dirname(run_dir.parent)
    promp, gmm = cs._load_promp_gmm_dir(str(sm / promp_dir))

    return dict(proto=proto, traj=traj, g_val=g_val, x_full=x_full, y_full=y_full,
                T=T, full_idx=np.arange(T, dtype=np.int64), pe=pe,
                cnmp=cnmp, pemp=pemp, wmp=wmp, promp=promp, gmm=gmm, promp_dir=promp_dir)


def panel_for_nctx(R, n_ctx: int):
    """Reconstruct the full trajectory with every approach for one n_ctx value."""
    proto, traj, g_val, T = R["proto"], R["traj"], R["g_val"], R["T"]
    x_full, y_full = R["x_full"], R["y_full"]
    ctx_x = proto[f"context_x_n{n_ctx}"][traj].reshape(-1, cs.DX)
    ctx_y = proto[f"context_y_n{n_ctx}"][traj].reshape(-1, cs.DY)
    ctx_idx = proto[f"context_indices_n{n_ctx}"][traj]
    preds = {
        "CNMP":  cs.pred_cnmp(R["cnmp"], ctx_x, ctx_y, g_val, x_full.reshape(-1, cs.DX)),
        "PEMP":  cs.pred_pemp(R["pemp"], R["pe"], ctx_idx, ctx_y, g_val, R["full_idx"]),
        "WMP":   cs.pred_wmp(R["wmp"], ctx_x.reshape(-1), ctx_y, g_val, R["full_idx"]),
        "ProMP": cs.pred_promp_gmm(R["promp"], R["gmm"], ctx_x.reshape(-1), ctx_y, g_val, x_full),
    }
    preds = {k: np.asarray(v).reshape(-1) for k, v in preds.items()}
    mses = {k: float(np.mean((v - y_full) ** 2)) for k, v in preds.items()}
    return dict(x=x_full, y=y_full, ctx_x=ctx_x.reshape(-1), ctx_y=ctx_y.reshape(-1),
                preds=preds, mses=mses, g=g_val, traj=traj, promp_dir=R["promp_dir"], T=T)


def draw(panel, title, out_svg, n_ctx):
    plt.figure(figsize=(8, 6))
    # ground truth: black, semi-transparent, behind everything
    plt.plot(panel["x"], panel["y"], color=COLORS["GT"], alpha=0.5,
             linewidth=LINEWIDTHS["GT"], linestyle=LINESTYLES["GT"], label="Ground truth")
    # model reconstructions
    for name in ("ProMP", "WMP", "CNMP", "PEMP"):
        plt.plot(panel["x"], panel["preds"][name], color=COLORS[name],
                 linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name],
                 label=("ProMP+GMM" if name == "ProMP" else name))
    # condition points
    plt.scatter(panel["ctx_x"], panel["ctx_y"], color="black", s=80, zorder=5,
                label=f"Condition points")

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Time", fontsize=14, fontweight="bold")
    plt.ylabel("Position", fontsize=14, fontweight="bold")
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="best", frameon=True, framealpha=0.75, fontsize=12, ncol=2)
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()
    print(f"[out] wrote {out_svg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simple-run", default=DEF_SIMPLE)
    ap.add_argument("--complex-run", default=DEF_COMPLEX)
    ap.add_argument("--simple-traj", type=int, default=None)
    ap.add_argument("--complex-traj", type=int, default=None)
    ap.add_argument("--n-ctx", type=int, nargs="+", default=[3, 10, 20],
                    help="one SVG per value (default 3 10 20); same example traj across them")
    ap.add_argument("--target-g", type=float, default=0.6,
                    help="pick the test traj whose g is closest to this (if --*-traj unset)")
    ap.add_argument("--out-dir", default="outputs/comparison/mind_change/freq/plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, run, override in (("Simple", args.simple_run, args.simple_traj),
                                 ("Complex", args.complex_run, args.complex_traj)):
        run_dir = Path(run)
        R = load_run(run_dir, args.target_g, override)
        print(f"[{label}] run={run_dir.name} traj={R['traj']} g={R['g_val']:.3f} "
              f"bestProMP={R['promp_dir']}")
        for N in args.n_ctx:
            panel = panel_for_nctx(R, N)
            print(f"   n_ctx={N:<3d} MSE: "
                  + "  ".join(f"{k}={panel['mses'][k]:.4f}" for k in ("PEMP", "CNMP", "WMP", "ProMP")))
            title = f"{label} synthetic — Reconstruction"
            draw(panel, title, out_dir / f"{label.lower()}_trajcomp_n{N}.svg", N)


if __name__ == "__main__":
    main()
