"""
Run with:
python mindchange/plot_trajectory_comparison_v2.py --n-ctx 10 --no-error-panel --no-title --combined
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import compare_synthetic as cs   # reuse the exact prediction code used by the tables


# ---------------------------------------------------------------------
# Journal-style plotting parameters
# ---------------------------------------------------------------------

plt.rcParams.update({
    # Keep text editable in SVG and use publication-friendly embedded fonts in PDF.
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Journal-like typography. If Times is unavailable, DejaVu Serif is used.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


COLORS = {
    "GT":    "#333333",   # dark gray, not pure black
    "PEMP":  "#D55E00",   # strong orange
    "CNMP":  "#4C78A8",   # muted blue
    "ProMP": "#560591",   # indigo
    "WMP":   "#006D77",   # dark teal, much less pale
}

LINESTYLES = {
    "GT":    "-",
    "PEMP":  "-",
    "CNMP":  "--",
    "ProMP": (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
    "WMP":   "-.",
}

LINEWIDTHS = {
    "GT": 1.3,
    "PEMP": 2.5,
    "CNMP": 1.75,
    "ProMP": 1.75,
    "WMP": 1.75,
}

ALPHAS = {
    "GT": 0.5,
    "PEMP": 1.00,
    "CNMP": 0.9,
    "ProMP": 0.9,
    "WMP": 0.90,
}

# Baselines first, then GT, then PEMP. This lets PEMP remain visible
# without artificially hiding the other methods.
TOP_DRAW_ORDER = ("ProMP", "WMP", "CNMP", "GT", "PEMP")
ERROR_DRAW_ORDER = ("ProMP", "WMP", "CNMP", "PEMP")


DEF_SIMPLE = "outputs/comparison/mind_change/sawtooth/1780303718"
DEF_COMPLEX = "outputs/comparison/mind_change/freq/combined/bare_pe_promp_gmm/1777148322"
AGG_CSV = "comparison_synth_v2_aggregated_byapproach.csv"


def best_promp_dirname(base: Path) -> str:
    """Lowest mean-MSE-across-n_ctx ProMP_<k> from the base's aggregated CSV
    identical selection to make_v2_tables.py. Returns 'promp_<k>'.
    """
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
    Returns a context dict reused across n_ctx values.
    """
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

    y_train = torch.load(
        run_dir / "y.pt",
        map_location="cpu",
        weights_only=False,
    ).float().numpy()

    g_train = torch.load(
        run_dir / "g.pt",
        map_location="cpu",
        weights_only=False,
    ).float().numpy().reshape(-1)

    wmp = cs._fit_wmp(y_train, g_train)

    promp_dir = best_promp_dirname(run_dir.parent)
    promp, gmm = cs._load_promp_gmm_dir(str(sm / promp_dir))

    return dict(
        proto=proto,
        traj=traj,
        g_val=g_val,
        x_full=x_full,
        y_full=y_full,
        T=T,
        full_idx=np.arange(T, dtype=np.int64),
        pe=pe,
        cnmp=cnmp,
        pemp=pemp,
        wmp=wmp,
        promp=promp,
        gmm=gmm,
        promp_dir=promp_dir,
    )


def panel_for_nctx(R, n_ctx: int):
    """Reconstruct the full trajectory with every approach for one n_ctx value."""
    proto, traj, g_val, T = R["proto"], R["traj"], R["g_val"], R["T"]
    x_full, y_full = R["x_full"], R["y_full"]

    ctx_x = proto[f"context_x_n{n_ctx}"][traj].reshape(-1, cs.DX)
    ctx_y = proto[f"context_y_n{n_ctx}"][traj].reshape(-1, cs.DY)
    ctx_idx = proto[f"context_indices_n{n_ctx}"][traj]

    preds = {
        "CNMP": cs.pred_cnmp(
            R["cnmp"],
            ctx_x,
            ctx_y,
            g_val,
            x_full.reshape(-1, cs.DX),
        ),
        "PEMP": cs.pred_pemp(
            R["pemp"],
            R["pe"],
            ctx_idx,
            ctx_y,
            g_val,
            R["full_idx"],
        ),
        "WMP": cs.pred_wmp(
            R["wmp"],
            ctx_x.reshape(-1),
            ctx_y,
            g_val,
            R["full_idx"],
        ),
        "ProMP": cs.pred_promp_gmm(
            R["promp"],
            R["gmm"],
            ctx_x.reshape(-1),
            ctx_y,
            g_val,
            x_full,
        ),
    }

    preds = {k: np.asarray(v).reshape(-1) for k, v in preds.items()}
    mses = {k: float(np.mean((v - y_full) ** 2)) for k, v in preds.items()}

    return dict(
        x=x_full,
        y=y_full,
        ctx_x=ctx_x.reshape(-1),
        ctx_y=ctx_y.reshape(-1),
        preds=preds,
        mses=mses,
        g=g_val,
        traj=traj,
        promp_dir=R["promp_dir"],
        T=T,
    )


def _pretty_label(name: str) -> str:
    return "ProMP+GMM" if name == "ProMP" else name


def _legend_handles(n_ctx: int, scale: float = 1.0):
    """Create legend handles.

    scale is useful for the combined 1x2 figure, where the default legend
    handles become visually too small after fitting the shared legend below
    both subfigures.
    """
    return [
        Line2D(
            [0], [0],
            color=COLORS["GT"],
            linestyle=LINESTYLES["GT"],
            linewidth=LINEWIDTHS["GT"] * scale,
            alpha=ALPHAS["GT"],
            label="Ground truth",
        ),
        Line2D(
            [0], [0],
            color=COLORS["PEMP"],
            linestyle=LINESTYLES["PEMP"],
            linewidth=LINEWIDTHS["PEMP"] * scale,
            alpha=ALPHAS["PEMP"],
            label="PEMP",
        ),
        Line2D(
            [0], [0],
            color=COLORS["CNMP"],
            linestyle=LINESTYLES["CNMP"],
            linewidth=LINEWIDTHS["CNMP"] * scale,
            alpha=ALPHAS["CNMP"],
            label="CNMP",
        ),
        Line2D(
            [0], [0],
            color=COLORS["ProMP"],
            linestyle=LINESTYLES["ProMP"],
            linewidth=LINEWIDTHS["ProMP"] * scale,
            alpha=ALPHAS["ProMP"],
            label="ProMP+GMM",
        ),
        Line2D(
            [0], [0],
            color=COLORS["WMP"],
            linestyle=LINESTYLES["WMP"],
            linewidth=LINEWIDTHS["WMP"] * scale,
            alpha=ALPHAS["WMP"],
            label="WMP",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0 * scale,
            markersize=5.5 * scale,
            label=f"Context points",
        ),
    ]


def _style_axis(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)



def _plot_panel_contents(
    panel,
    title,
    ax,
    ax_err=None,
    *,
    show_title: bool = True,
    show_xlabel: bool = True,
):
    """Plot one reconstruction panel into existing axes.

    This helper is used by both draw() and draw_combined() so the single and
    combined figures remain visually identical.
    """
    # -----------------------------
    # Reconstruction panel
    # -----------------------------
    for name in TOP_DRAW_ORDER:
        if name == "GT":
            ax.plot(
                panel["x"],
                panel["y"],
                color=COLORS["GT"],
                linestyle=LINESTYLES["GT"],
                linewidth=LINEWIDTHS["GT"],
                alpha=ALPHAS["GT"],
                zorder=3,
            )
        else:
            z = 4 if name == "PEMP" else 2
            ax.plot(
                panel["x"],
                panel["preds"][name],
                color=COLORS[name],
                linestyle=LINESTYLES[name],
                linewidth=LINEWIDTHS[name],
                alpha=ALPHAS[name],
                zorder=z,
            )

    # Hollow context points: visible but not visually dominant.
    ax.scatter(
        panel["ctx_x"],
        panel["ctx_y"],
        s=80,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
    )

    if show_title:
        ax.set_title(title, fontsize=16, fontweight="normal", pad=6)

    ax.set_ylabel("Position", fontsize=14)
    _style_axis(ax)

    # -----------------------------
    # Optional error panel
    # -----------------------------
    if ax_err is not None:
        for name in ERROR_DRAW_ORDER:
            err = np.abs(panel["preds"][name] - panel["y"])
            ax_err.plot(
                panel["x"],
                err,
                color=COLORS[name],
                linestyle=LINESTYLES[name],
                linewidth=max(1.1, LINEWIDTHS[name] - 0.2),
                alpha=ALPHAS[name],
                zorder=4 if name == "PEMP" else 2,
            )

        ax_err.set_ylabel("Abs. error", fontsize=14)
        if show_xlabel:
            ax_err.set_xlabel("Time", fontsize=14)
        ax_err.set_ylim(bottom=0.0)
        _style_axis(ax_err)
    elif show_xlabel:
        ax.set_xlabel("Time", fontsize=14)


def draw(panel, title, out_path, n_ctx, with_error_panel: bool = True, show_title: bool = True):
    """Draw one 7x5 reconstruction figure."""
    out_svg = Path(out_path).with_suffix(".svg")
    out_png = Path(out_path).with_suffix(".png")

    if with_error_panel:
        fig, (ax, ax_err) = plt.subplots(
            2,
            1,
            figsize=(7.0, 5.0),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax_err = None

    _plot_panel_contents(
        panel,
        title,
        ax,
        ax_err,
        show_title=show_title,
        show_xlabel=True,
    )

    handles = _legend_handles(n_ctx)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=True,
        framealpha=0.75,
        fontsize=12,
        handlelength=2.6,
        columnspacing=1.0,
    )

    # Leave room for the outside legend.
    fig.tight_layout(rect=[0.02, 0.12, 0.995, 0.98])

    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"[out] wrote {out_svg}")


def draw_combined(
    simple_panel,
    complex_panel,
    out_path,
    n_ctx,
    with_error_panel: bool = True,
    show_title: bool = True,
):
    """Draw Simple and Complex figures vertically with one shared legend."""
    out_svg = Path(out_path).with_suffix(".svg")
    out_png = Path(out_path).with_suffix(".png")

    if with_error_panel:
        fig = plt.figure(figsize=(7.0, 9.2))

        outer = fig.add_gridspec(
            2,
            1,
            height_ratios=[1, 1],
            hspace=0.06,   # decreased to bring the two main blocks closer
        )

        simple_gs = outer[0].subgridspec(
            2,
            1,
            height_ratios=[3.2, 1.0],
            hspace=0.05,
        )
        complex_gs = outer[1].subgridspec(
            2,
            1,
            height_ratios=[3.2, 1.0],
            hspace=0.05,
        )

        ax_simple = fig.add_subplot(simple_gs[0])
        axerr_simple = fig.add_subplot(simple_gs[1], sharex=ax_simple)

        ax_complex = fig.add_subplot(complex_gs[0])
        axerr_complex = fig.add_subplot(complex_gs[1], sharex=ax_complex)

        _plot_panel_contents(
            simple_panel,
            "Simple synthetic reconstruction",
            ax_simple,
            axerr_simple,
            show_title=show_title,
            show_xlabel=True,
        )
        _plot_panel_contents(
            complex_panel,
            "Complex synthetic reconstruction",
            ax_complex,
            axerr_complex,
            show_title=show_title,
            show_xlabel=True,
        )

        ax_simple.set_ylim(-0.9, 1.2)
        ax_complex.set_ylim(-1.05, 1.05)

    else:
        fig, (ax_simple, ax_complex) = plt.subplots(
            2,
            1,
            figsize=(6.5, 9.2),
            sharex=True,
            sharey=False,
            gridspec_kw={
                "height_ratios": [1, 1],
                "hspace": 0.02,   # decreased from previous value
            },
        )

        _plot_panel_contents(
            simple_panel,
            "Simple synthetic reconstruction",
            ax_simple,
            None,
            show_title=show_title,
            show_xlabel=False,
        )
        _plot_panel_contents(
            complex_panel,
            "Complex synthetic reconstruction",
            ax_complex,
            None,
            show_title=show_title,
            show_xlabel=True,
        )

        ax_simple.set_ylim(-0.9, 1.2)
        ax_complex.set_ylim(-1.05, 1.05)

    handles = _legend_handles(n_ctx, scale=1.15)
    legend = fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.999),
        ncol=2,
        frameon=True,
        framealpha=0.85,
        prop={"weight": "bold", "size": 12},
        handlelength=3.0,
        handletextpad=0.7,
        columnspacing=1.2,
        labelspacing=0.45,
        borderpad=0.45,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.07,
        top=0.96 if show_title else 0.985,
    )

    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"[out] wrote {out_svg}")


def _print_mses(label: str, n_ctx: int, panel):
    print(
        f"   [{label}] n_ctx={n_ctx:<3d} MSE: "
        + "  ".join(
            f"{k}={panel['mses'][k]:.4f}"
            for k in ("PEMP", "CNMP", "WMP", "ProMP")
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--simple-run", default=DEF_SIMPLE)
    ap.add_argument("--complex-run", default=DEF_COMPLEX)

    ap.add_argument("--simple-traj", type=int, default=None)
    ap.add_argument("--complex-traj", type=int, default=None)

    ap.add_argument(
        "--n-ctx",
        type=int,
        nargs="+",
        default=[3, 10, 20],
        help="one SVG per value; default: 3 10 20",
    )

    ap.add_argument(
        "--target-g",
        type=float,
        default=0.6,
        help="pick the test trajectory whose g is closest to this, unless --*-traj is set",
    )

    ap.add_argument(
        "--out-dir",
        default="outputs/comparison/mind_change/freq/plots",
    )

    ap.add_argument(
        "--no-error-panel",
        action="store_true",
        help="draw only the reconstruction panel",
    )

    ap.add_argument(
        "--no-title",
        action="store_true",
        help="omit axes title; useful when the paper caption already gives the title",
    )

    ap.add_argument(
        "--combined",
        action="store_true",
        help="combine Simple and Complex in a 2x1 vertical layout",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with_error_panel = not args.no_error_panel
    show_title = not args.no_title

    if args.combined:
        simple_run_dir = Path(args.simple_run)
        complex_run_dir = Path(args.complex_run)

        R_simple = load_run(simple_run_dir, args.target_g, args.simple_traj)
        R_complex = load_run(complex_run_dir, args.target_g, args.complex_traj)

        print(
            f"[Simple] run={simple_run_dir.name} traj={R_simple['traj']} "
            f"g={R_simple['g_val']:.3f} bestProMP={R_simple['promp_dir']}"
        )
        print(
            f"[Complex] run={complex_run_dir.name} traj={R_complex['traj']} "
            f"g={R_complex['g_val']:.3f} bestProMP={R_complex['promp_dir']}"
        )

        for N in args.n_ctx:
            simple_panel = panel_for_nctx(R_simple, N)
            complex_panel = panel_for_nctx(R_complex, N)

            _print_mses("Simple", N, simple_panel)
            _print_mses("Complex", N, complex_panel)

            suffix = "reconerr" if with_error_panel else "recon"
            out_path = out_dir / f"combined_comp_n{N}_{suffix}"

            draw_combined(
                simple_panel,
                complex_panel,
                out_path,
                N,
                with_error_panel=with_error_panel,
                show_title=show_title,
            )

        return

    for label, run, override in (
        ("Simple", args.simple_run, args.simple_traj),
        ("Complex", args.complex_run, args.complex_traj),
    ):
        run_dir = Path(run)
        R = load_run(run_dir, args.target_g, override)

        print(
            f"[{label}] run={run_dir.name} traj={R['traj']} "
            f"g={R['g_val']:.3f} bestProMP={R['promp_dir']}"
        )

        for N in args.n_ctx:
            panel = panel_for_nctx(R, N)
            _print_mses(label, N, panel)

            title = f"{label} synthetic reconstruction"

            suffix = "reconerr" if with_error_panel else "recon"
            out_path = out_dir / f"{label.lower()}_comp_n{N}_{suffix}"

            draw(
                panel,
                title,
                out_path,
                N,
                with_error_panel=with_error_panel,
                show_title=show_title,
            )


if __name__ == "__main__":
    main()