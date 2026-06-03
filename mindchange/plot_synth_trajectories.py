"""Figure script for the synthetic-comparison section of the paper.

Four single-panel PDFs sized for a 2x2 subfigure block in a 2-column layout:

  simple_examples.pdf            -- demonstration family: one trajectory per g
  complex_examples.pdf           -- same, complex (combined) dataset
  simple_comparison_g0.4_n3.pdf  -- GT (all g faint + g=0.4 bold) + CNMP/PEMP/WMP/
                                    ProMP+GMM at n_ctx=3, with +-1 sigma bands
  complex_comparison_g0.4_n3.pdf -- same on the complex dataset

Palette is the colorblind-safe Okabe-Ito set, fixed module-level so every other
figure in the paper can `from plot_synth_trajectories import COLORS, STYLES`.

Run:
  ~/sw/anaconda3/envs/pemp-gpu/bin/python mindchange/plot_synth_trajectories.py
"""
from __future__ import annotations

import os
import sys
import csv
import math
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / "models", ROOT / "data", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cnmp import CNMP
from positional_encoders import generate_positional_encoding
from wmp_baseline import WaveletMovementPrimitive
from movement_primitives.promp import ProMP  # noqa: F401  (pickle needs the class)

# ============================================================================
# config
# ============================================================================
BASES = {
    "simple":  ROOT / "outputs/comparison/mind_change/freq/bare_pe_promp_gmm",
    "complex": ROOT / "outputs/comparison/mind_change/freq/combined/bare_pe_promp_gmm",
}
OUT_DIR = ROOT / "outputs/comparison/mind_change/freq/plots"
COND_FILE = "conditioning_v2.npz"
AGG_CSV = "comparison_synth_v2_aggregated_byapproach.csv"
G_VALUE = 0.4
N_CTX = 3
PROMP_OBS_NOISE = 1e-3   # y_cov for position conditioning (matches compare_synthetic.py)

# CNMP/PEMP architecture (matches notebook + compare_synthetic.py)
DX, DG, DY = 1, 1, 1
DPE = 27
PE_SCALER = 0.2
ENC_DIMS = [256, 256]
DEC_DIMS = [256, 256]

# ============================================================================
# Palette + line styles.  Okabe-Ito colorblind-safe colors; GT near-black.
# PEMP (our method) gets the warm vermillion so it reads as the hero curve.
# These are the canonical paper colors -- import them elsewhere for consistency.
# ============================================================================
COLORS = {
    "GT":        "#9aa0a6",   # soft grey -- reference, deliberately recessive
    "PEMP":      "#D55E00",   # vermillion  (our method -- the hero curve)
    "WMP":       "#009E73",   # bluish green
    "CNMP":      "#0072B2",   # blue
    "ProMP+GMM": "#CC79A7",   # reddish purple
}
# GT is thin + soft so the colored model curves carry the visual weight; PEMP is
# the thickest/most-saturated line so the eye lands on our method first.
STYLES = {
    "GT":        dict(linestyle="-",  linewidth=1.3),
    "PEMP":      dict(linestyle="-",  linewidth=2.4),
    "WMP":       dict(linestyle="--", linewidth=1.9),
    "CNMP":      dict(linestyle="-.", linewidth=1.9),
    "ProMP+GMM": dict(linestyle=(0, (1, 1)), linewidth=2.1),
}
GT_ALPHA = 0.85
BAND_ALPHA = 0.13
# faint ground-truth family (all g) behind the focus trajectory -- lighter than GT
# so it never competes with either the GT reference or the model curves.
FAMILY_COLOR = "0.82"
FAMILY_LW = 0.6
FAMILY_ALPHA = 0.35
GVALS = [0.2, 0.4, 0.6, 0.8, 1.0]

# ============================================================================
# matplotlib defaults -- clean, publication quality
# ============================================================================
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titlepad": 6,
    "axes.labelsize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
FIGSIZE = (3.5, 2.6)


# ============================================================================
# data + model loading (mirrors compare_synthetic.py)
# ============================================================================
def _first_ts_with_models(base):
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / COND_FILE).exists() and (d / "saved_models").exists():
            return d
    raise FileNotFoundError(f"no usable ts dir under {base}")


def _load_cnmp_like(state_path, input_dim, t_steps):
    m = CNMP(input_dim=input_dim, output_dim=DY, n_max=1, m_max=t_steps,
             encoder_hidden_dims=ENC_DIMS, decoder_hidden_dims=DEC_DIMS,
             batch_size=1, device="cpu")
    m.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def _load_promp(path):
    with open(path / "promp.pkl", "rb") as f: promp = pickle.load(f)
    with open(path / "gmm.pkl",   "rb") as f: gmm   = pickle.load(f)
    return promp, gmm


def _best_promp_label(base):
    """Lowest mean MSE (averaged across n_ctx) ProMP_<k> in the aggregated CSV."""
    csv_path = base / AGG_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- run the aggregation first")
    means = {}
    for r in csv.DictReader(csv_path.open()):
        if r["approach"].startswith("ProMP_") and r["mse_mean"] not in ("", "nan"):
            means.setdefault(r["approach"], []).append(float(r["mse_mean"]))
    best = min(means, key=lambda k: sum(means[k]) / len(means[k]))
    return f"promp_{best.split('_')[1]}"


def _pick_traj_for_g(g_arr, g_value, atol=1e-3):
    diffs = np.abs(g_arr.reshape(-1) - g_value)
    idx = int(np.argmin(diffs))
    if diffs[idx] > atol:
        raise ValueError(f"no test traj with g~={g_value} (closest diff={diffs[idx]:.3g})")
    return idx


def _one_traj_per_g(g_arr):
    """Return {g_value: traj_idx} picking the first test traj at each g in GVALS."""
    out = {}
    g_flat = g_arr.reshape(-1)
    for gv in GVALS:
        idxs = np.where(np.abs(g_flat - gv) < 1e-3)[0]
        if len(idxs):
            out[gv] = int(idxs[0])
    return out


# ============================================================================
# per-method predictions: return (mean (T,), std (T,))
# ============================================================================
def _torch_mean_std(model, obs, tar, mask):
    import torch.nn.functional as F
    with torch.no_grad():
        pred = model.val(obs, tar, mask)[0]
        mu = pred[:, :DY].cpu().numpy().reshape(-1)
        sd = (F.softplus(pred[:, DY:]) + 1e-4).cpu().numpy().reshape(-1)
    return mu, sd


def predict_cnmp(model, ctx_x, ctx_y, g_val, full_x):
    n = len(ctx_x); T = len(full_x)
    obs = torch.zeros(1, n, DX + DG + DY)
    obs[0, :, :DX]        = torch.as_tensor(ctx_x, dtype=torch.float32).reshape(n, DX)
    obs[0, :, DX:DX + DG] = float(g_val)
    obs[0, :, DX + DG:]   = torch.as_tensor(ctx_y, dtype=torch.float32).reshape(n, DY)
    tar = torch.zeros(1, T, DX + DG)
    tar[0, :, :DX] = torch.as_tensor(full_x, dtype=torch.float32).reshape(T, DX)
    tar[0, :, DX:] = float(g_val)
    return _torch_mean_std(model, obs, tar, torch.ones(1, n, dtype=torch.bool))


def predict_pemp(model, pe, ctx_idx, ctx_y, g_val, full_idx):
    n = len(ctx_idx); T = len(full_idx)
    obs = torch.zeros(1, n, DPE + DG + DY)
    obs[0, :, :DPE]         = pe[ctx_idx]
    obs[0, :, DPE:DPE + DG] = float(g_val)
    obs[0, :, DPE + DG:]    = torch.as_tensor(ctx_y, dtype=torch.float32).reshape(n, DY)
    tar = torch.zeros(1, T, DPE + DG)
    tar[0, :, :DPE] = pe[full_idx]
    tar[0, :, DPE:] = float(g_val)
    return _torch_mean_std(model, obs, tar, torch.ones(1, n, dtype=torch.bool))


def predict_wmp(wmp, ctx_x, ctx_y, g_val):
    mean, std = wmp.predict(
        context=np.atleast_2d(g_val).reshape(1, -1),
        t_cond=np.asarray(ctx_x).reshape(-1),
        y_cond=np.asarray(ctx_y).reshape(-1, DY),
        method="phase_adaptive", batched=True, return_std=True,
    )
    return mean.reshape(-1), std.reshape(-1)


def predict_promp(promp, gmm, ctx_x, ctx_y, g_val, full_x):
    """condition_position returns a NEW ProMP (no in-place mutation) -> rebind cp."""
    g_q = np.atleast_2d(float(g_val))
    cond = gmm.condition(np.arange(g_q.shape[1]), g_q).to_mvn()
    promp.from_weight_distribution(cond.mean, cond.covariance)
    cp = promp
    y_cov = np.array([[PROMP_OBS_NOISE]])
    for tc, yc in zip(np.asarray(ctx_x).reshape(-1), np.asarray(ctx_y).reshape(-1)):
        cp = cp.condition_position(np.array([float(yc)]), y_cov=y_cov,
                                   t=float(tc), t_max=1.0)
    full_x = np.asarray(full_x).reshape(-1)
    mean = cp.mean_trajectory(full_x).reshape(-1)
    var = cp.var_trajectory(full_x).reshape(-1)
    return mean, np.sqrt(np.maximum(var, 0.0))


# ============================================================================
# plotting
# ============================================================================
def _finish_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("phase  $x$")
    ax.set_ylabel("position  $y$")
    ax.set_xlim(0.0, 1.0)
    ax.margins(y=0.08)


def plot_examples(dataset, x_test, y_test, g_test, out_path):
    """Demonstration family: one trajectory per g, colored by a sequential map."""
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    per_g = _one_traj_per_g(g_test)
    norm = Normalize(vmin=min(GVALS), vmax=max(GVALS))
    cmap = mpl.colormaps["viridis"]
    for gv, ti in per_g.items():
        ax.plot(x_test[ti].reshape(-1), y_test[ti].reshape(-1),
                color=cmap(norm(gv)), linewidth=1.4, alpha=0.9,
                solid_capstyle="round")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("frequency  $g$", fontsize=8.5)
    cbar.set_ticks(GVALS)
    cbar.ax.tick_params(labelsize=7.5)
    label = "Simple" if dataset == "simple" else "Complex"
    _finish_axes(ax, f"{label} synthetic: demonstration family")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_comparison(dataset, x_full, y_focus, family, preds, ctx_x, ctx_y, out_path):
    """All-g GT faint behind; focus-g GT bold; 4 model means + sigma bands; context."""
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    # faint ground-truth family (all g), thin + transparent, behind everything
    for xx, yy in family:
        ax.plot(xx, yy, color=FAMILY_COLOR, linewidth=FAMILY_LW,
                alpha=FAMILY_ALPHA, zorder=1)

    # +-1 sigma bands (under the mean lines)
    for name in ("ProMP+GMM", "CNMP", "WMP", "PEMP"):
        mu, sd = preds[name]
        ax.fill_between(x_full, mu - sd, mu + sd, color=COLORS[name],
                        alpha=BAND_ALPHA, linewidth=0, zorder=2)

    # model means; PEMP last so the hero curve sits on top of the other models
    for name in ("ProMP+GMM", "CNMP", "WMP", "PEMP"):
        mu, _ = preds[name]
        ax.plot(x_full, mu, color=COLORS[name], label=name, zorder=3,
                solid_capstyle="round", **STYLES[name])

    # focus-g ground truth: thin + soft, sits just under the model lines so it
    # reads as the reference the colored curves are tracking, not the hero.
    ax.plot(x_full, y_focus, color=COLORS["GT"], label="ground truth",
            zorder=2.5, alpha=GT_ALPHA, solid_capstyle="round", **STYLES["GT"])

    # context points on top, dark neutral edge (visible against the soft-grey GT)
    ax.scatter(ctx_x, ctx_y, s=46, facecolor="white", edgecolor="#333333",
               linewidths=1.4, zorder=6,
               label=f"context ($n_{{\\mathrm{{ctx}}}}={N_CTX}$)")

    label = "Simple" if dataset == "simple" else "Complex"
    _finish_axes(ax, f"{label} synthetic ($g={G_VALUE}$)")

    # legend placed ABOVE the axes (outside) so it never occludes the curves;
    # compact 4-column horizontal block, faint-family entry appended last.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=FAMILY_COLOR, lw=FAMILY_LW,
                          alpha=FAMILY_ALPHA))
    labels.append("other $g$ (GT)")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=4, columnspacing=1.0, handletextpad=0.5, labelspacing=0.4,
              borderaxespad=0.0, fontsize=7)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# main
# ============================================================================
def make_for_base(dataset, base):
    ts = _first_ts_with_models(base)
    best_promp = _best_promp_label(base)
    print(f"[{dataset}] ts={ts.name}  best ProMP={best_promp}")
    proto = np.load(ts / COND_FILE, allow_pickle=False)
    g_test = proto["g"]
    x_all = proto["x_test_full"]                      # (n, T, 1)
    y_all = proto["y_test_full"]
    traj_idx = _pick_traj_for_g(g_test, G_VALUE)
    g_val = float(g_test[traj_idx].reshape(-1)[0])
    x_full = x_all[traj_idx].reshape(-1)
    y_focus = y_all[traj_idx].reshape(-1)
    ctx_idx = proto[f"context_indices_n{N_CTX}"][traj_idx]
    ctx_x = proto[f"context_x_n{N_CTX}"][traj_idx].reshape(-1)
    ctx_y = proto[f"context_y_n{N_CTX}"][traj_idx].reshape(-1)
    t_steps = len(x_full)
    full_idx = np.arange(t_steps, dtype=np.int64)
    pe = generate_positional_encoding(t_steps, DPE, PE_SCALER)

    # faint GT family: one traj per g (excluding the focus traj which is drawn bold)
    per_g = _one_traj_per_g(g_test)
    family = [(x_all[ti].reshape(-1), y_all[ti].reshape(-1))
              for gv, ti in per_g.items() if ti != traj_idx]

    # ----- models -----
    sm = ts / "saved_models"
    cnmp = _load_cnmp_like(sm / "bare.pt", DX + DG, t_steps)
    pemp = _load_cnmp_like(sm / "pe.pt",   DPE + DG, t_steps)
    y_train = torch.load(ts / "y.pt", weights_only=False).numpy()
    g_train = torch.load(ts / "g.pt", weights_only=False).numpy().reshape(-1, 1)
    wmp = WaveletMovementPrimitive(wavelet="db4", mode="periodization",
                                   obs_noise=1e-3).fit(y_train, contexts=g_train)
    promp, gmm = _load_promp(sm / best_promp)

    preds = {
        "CNMP":      predict_cnmp(cnmp, ctx_x, ctx_y, g_val, x_full),
        "PEMP":      predict_pemp(pemp, pe, ctx_idx, ctx_y, g_val, full_idx),
        "WMP":       predict_wmp(wmp,  ctx_x, ctx_y, g_val),
        "ProMP+GMM": predict_promp(promp, gmm, ctx_x, ctx_y, g_val, x_full),
    }
    for k, (mu, sd) in preds.items():
        mse = float(np.mean((mu - y_focus) ** 2))
        print(f"    {k:9s} full-traj MSE={mse:.4f}  std[mean]={sd.mean():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_examples(dataset, x_all, y_all, g_test, OUT_DIR / f"{dataset}_examples.pdf")
    plot_comparison(dataset, x_full, y_focus, family, preds, ctx_x, ctx_y,
                    OUT_DIR / f"{dataset}_comparison_g{G_VALUE}_n{N_CTX}.pdf")


def main():
    for dataset, base in BASES.items():
        make_for_base(dataset, base)
    print(f"\nall figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
