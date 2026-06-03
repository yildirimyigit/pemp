"""Quantitative comparison of FluidLab Mixing action trajectories from N approaches,
with mean +/- std over seeds.

Seeds vs conditions:
  * SEED   = a training run (different init).  Averaging over runs = real method variance.
  * g      = the conditioning frequency, an experimental CONDITION, not a seed.  Metrics
             depend on g, so we (a) average over runs *at each g* (rigorous per-condition
             error bars) and (b) also report an across-frequency aggregate over runs x g,
             explicitly as "expected performance over the operating frequency range".
For the frequency-fidelity column we use the loop ERROR |realized - commanded|, which is
comparable across g (raw loop count trivially grows with g and can't be pooled).

Core API:
    samples = collect_samples(env, runs, gs, approaches, n_draws=...)  # 1 row per (run,approach,g,draw)
    by_g    = aggregate(samples, ["approach", "g"])   # mean+/-std over runs x draws (per condition)
    overall = aggregate(samples, ["approach"])        # mean+/-std over runs x g x draws
`approaches` is {label: file_key}, e.g. {"PEMP": "pe", "CNMP": "bare"}.  Each run dir is a
SEED and must hold saved_models/<key>.pt (+ y.pt/g.pt/hyperparameters.yaml); the metrics
script loads the model and generates the trajectory itself, so it can vary BOTH the model
context and the start position per draw.  A draw (when --n-draws>1) = random context
(ctx_seed) + a right-biased random stirrer start offset (phase-independent spatial
randomness; PEMP and CNMP share the same per-(g,draw) offset, so the comparison is paired).
To add a new approach (e.g. ProMP) give it a generator branch in _generate().  You can also
call evaluate(env, actions, commanded_loops, start_offset) directly on your own actions.

Metrics (v lower better, ^ higher better):
  MIXING   CoV v, Seg(Danckwerts) v, Entropy ^, Disp(Rg) ^, Interf(kNN) ^
  FIDELITY LoopErr v  (|realized loops - commanded g*MAX_LOOPS|)
  CONTROL  Smooth(jerk) v, Path v, MixRate ^

Run in the `fluidlab` conda env:
  FLUIDLAB_TI_MEM_GB=4.0 PYTHONPATH=$HOME/projects/FluidLab \
    ~/sw/anaconda3/envs/fluidlab/bin/python sim/compare_mixing.py \
      --runs outputs/sim/mixing/<runA> outputs/sim/mixing/<runB> --g 0.333 0.5 0.667 0.833 1.0
"""
import os
import sys
import math
import csv
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from fluidlab_mixing_rollout import make_env, set_start_offset, ACT_LIMIT

MAX_LOOPS = 6
DEFAULT_APPROACHES = {"PEMP": "pe", "CNMP": "bare"}  # label -> file key

# poolable metrics for the mean+/-std tables: (key, header, fmt, higher_is_better)
AGG = [
    ("cov",      "CoV",      "{:.3f}", False),
    ("seg",      "Seg",      "{:.3f}", False),
    ("entropy",  "Entropy",  "{:.3f}", True),
    ("disp",     "Disp",     "{:.2f}", True),
    ("interf",   "Interf",   "{:.3f}", True),
    ("loop_err", "LoopErr",  "{:.2f}", False),
    ("smooth",   "Smooth",   "{:.3f}", False),
    ("path",     "Path",     "{:.2f}", False),
    ("mixrate",  "MixRate",  "{:.3f}", True),
]


# ----------------------------- metric helpers ------------------------------- #
def _rg(pts):
    c = pts.mean(0)
    return float(np.sqrt(((pts - c) ** 2).sum(1).mean()))


def _conc_cells(x, is_m, is_c, n_bins, min_count):
    fluid = is_m | is_c
    lo, hi = x[fluid].min(0), x[fluid].max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    idx = np.floor((x - lo) / span * n_bins).clip(0, n_bins - 1).astype(np.int64)
    flat = (idx[:, 0] * n_bins + idx[:, 1]) * n_bins + idx[:, 2]
    nb = n_bins ** 3
    milk = np.bincount(flat[is_m], minlength=nb).astype(float)
    coff = np.bincount(flat[is_c], minlength=nb).astype(float)
    tot = milk + coff
    keep = tot >= min_count
    return milk[keep] / tot[keep], milk[keep]


def _mixing_quality(x, is_m, is_c, n_bins, min_count):
    c, milk = _conc_cells(x, is_m, is_c, n_bins, min_count)
    if len(c) < 2:
        return dict(cov=float("nan"), seg=float("nan"), entropy=float("nan"))
    cbar = c.mean()
    cov = float(c.std() / (cbar + 1e-12))
    seg = float(c.var() / (cbar * (1 - cbar) + 1e-12))
    p = milk / (milk.sum() + 1e-12)
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    return dict(cov=cov, seg=seg, entropy=H / math.log(len(c)))


def _interface_fraction(x, is_m, is_c, k=12, sample=2000):
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return float("nan")
    idx_m = np.where(is_m)[0]
    if len(idx_m) > sample:
        idx_m = np.random.default_rng(0).choice(idx_m, sample, replace=False)
    tree = cKDTree(x)
    _, nn = tree.query(x[idx_m], k=k + 1)
    return float(is_c[nn[:, 1:]].any(axis=1).mean())


def _realized_loops(a):
    s = a[:, 0] - a[:, 0].mean()
    sp = np.abs(np.fft.rfft(s)); sp[0] = 0.0
    return int(np.argmax(sp))


def _smoothness(a):
    mj = np.linalg.norm(np.diff(a, n=3, axis=0), axis=1).mean()
    return float(mj / (np.linalg.norm(a, axis=1).mean() + 1e-12))


# ------------------------------- evaluation --------------------------------- #
def evaluate(env, actions, commanded_loops=None, start_offset=(0.0, 0.0),
             n_bins=6, min_count=20, eval_stride=25):
    from fluidlab.configs.macros import MILK_VIS, COFFEE_VIS

    env.reset()
    set_start_offset(env, *start_offset)  # phase-independent spatial randomness (no-op if (0,0))
    te = env.taichi_env
    mat = te.simulator.particles_i.mat.to_numpy()
    is_m, is_c = mat == MILK_VIS, mat == COFFEE_VIS

    x0 = te.get_state()["state"]["x"]
    rg0 = _rg(x0[is_m])
    seg_series = [_mixing_quality(x0, is_m, is_c, n_bins, min_count)["seg"]]
    for t in range(len(actions)):
        te.step(np.clip(actions[t], -ACT_LIMIT, ACT_LIMIT))
        if (t % eval_stride == 0) or (t == len(actions) - 1):
            xt = te.get_state()["state"]["x"]
            seg_series.append(_mixing_quality(xt, is_m, is_c, n_bins, min_count)["seg"])

    xf = te.get_state()["state"]["x"]
    q = _mixing_quality(xf, is_m, is_c, n_bins, min_count)
    seg = np.array(seg_series)
    a = np.asarray(actions, dtype=np.float64)
    loops = _realized_loops(a)
    cmd = round(commanded_loops) if commanded_loops is not None else float("nan")
    return dict(
        cov=q["cov"], seg=q["seg"], entropy=q["entropy"],
        disp=_rg(xf[is_m]) / (rg0 + 1e-12),
        interf=_interface_fraction(xf, is_m, is_c),
        loops=loops, loops_cmd=cmd,
        loop_err=abs(loops - cmd) if commanded_loops is not None else float("nan"),
        smooth=_smoothness(a),
        path=float(np.linalg.norm(a, axis=1).sum()),
        mixrate=float(np.clip((seg[0] - seg[len(seg) // 2]) / (seg[0] - seg[-1] + 1e-12), 0, 1)),
    )


# ----------------------------- collect + aggregate -------------------------- #
_WMP_CACHE = {}  # run_dir -> (wmp, y_train, g_train, denorm); fit once per run


def _generate(key, run, g, n_ctx, pe_scaler, ctx_seed, snap_g=True):
    """Generate one action trajectory for `key` by loading that run's model.
    Currently supports:
        'pe'   -> PEMP  (PE-CNMP, loads saved_models/pe.pt)
        'bare' -> CNMP  (loads saved_models/bare.pt)
        'wmp'  -> WMP   (fitted in-process from y.pt/g.pt; cached per run)
    snap_g=False conditions on the REQUESTED g (off-training-grid evaluation);
    default True keeps the prior behaviour.
    Imported lazily so compare_mixing has no hard import cycle.
    """
    if key == "pe":
        import fluidlab_mixing_pemp_test as M
        return M.generate_trajectory(run, g, n_ctx, None, pe_scaler,
                                     ctx_seed=ctx_seed, snap_g=snap_g)
    if key == "bare":
        import fluidlab_mixing_bare_test as M
        return M.generate_trajectory(run, g, n_ctx, None,
                                     ctx_seed=ctx_seed, snap_g=snap_g)
    if key == "wmp":
        import fluidlab_mixing_wmp_test as M
        if run not in _WMP_CACHE:
            _WMP_CACHE[run] = M._fit_wmp(run)
        wmp, y_train, g_train, denorm = _WMP_CACHE[run]
        return M.generate_trajectory(wmp, y_train, g_train, denorm, g,
                                     n_ctx=n_ctx, ctx_seed=ctx_seed, snap_g=snap_g)
    raise ValueError(f"no generator for approach key '{key}' (extend _generate)")


def collect_samples(env, runs, gs, approaches, n_draws=1, n_ctx=10, pe_scaler=0.2,
                    start_dx=(0.10, 0.30), start_dz=0.12, n_bins=6, snap_g=True):
    """One sample per (run, approach, g, draw).  A draw varies BOTH the model context
    (ctx_seed) and a right-biased start offset; draw 0 with n_draws==1 is the canonical
    centered, evenly-spaced eval (backward compatible).  PEMP and CNMP get the SAME
    per-(g,draw) start offset, so the comparison stays paired."""
    samples = []
    for run in runs:
        rname = os.path.basename(run.rstrip("/"))
        for g in gs:
            for j in range(n_draws):
                ctx_seed = None if n_draws == 1 else j
                if n_draws == 1:
                    off = (0.0, 0.0)
                else:
                    r = np.random.default_rng(int(round(g * 1000)) * 1000 + j)
                    off = (float(r.uniform(*start_dx)), float(r.uniform(-start_dz, start_dz)))
                for label, key in approaches.items():
                    print(f"[eval] {label} g={g} run={rname} draw={j} start=({off[0]:.2f},{off[1]:.2f})")
                    try:
                        actions, g_used = _generate(key, run, g, n_ctx, pe_scaler, ctx_seed,
                                                    snap_g=snap_g)
                    except FileNotFoundError:
                        print(f"[skip] {label} g={g} run={rname}: model not found"); continue
                    m = evaluate(env, actions, commanded_loops=g * MAX_LOOPS,
                                 start_offset=off, n_bins=n_bins)
                    m.update(approach=label, g=round(g, 3), run=rname, draw=j)
                    samples.append(m)
    return samples


def aggregate(samples, keys):
    groups = {}
    for s in samples:
        groups.setdefault(tuple(s[k] for k in keys), []).append(s)
    out = []
    for gk, ss in groups.items():
        row = dict(zip(keys, gk)); row["n"] = len(ss)
        for k, *_ in AGG:
            v = np.array([s[k] for s in ss
                          if not (isinstance(s[k], float) and math.isnan(s[k]))], float)
            row[k + "_mean"] = float(v.mean()) if len(v) else float("nan")
            row[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    # sort: by approach then g if present
    out.sort(key=lambda r: (str(r.get("approach", "")), r.get("g", 0)))
    return out


# ------------------------------- formatting --------------------------------- #
def _cell(row, k, fmt):
    m, s = row.get(k + "_mean"), row.get(k + "_std")
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "-"
    return f"{fmt.format(m)}+-{fmt.format(s)}" if row.get("n", 1) > 1 else fmt.format(m)


def _print_agg(rows, keys, title):
    arrow = {True: "^", False: "v"}
    cols = list(keys) + ["n"] + [k for k, *_ in AGG]
    head = {k: k for k in keys}; head["n"] = "n"
    head.update({k: f"{h}{arrow.get(b, '')}" for k, h, _f, b in AGG})
    def cellstr(r, c):
        if c in keys: return str(r[c])
        if c == "n": return str(r["n"])
        fmt = next(f for k, _h, f, _b in AGG if k == c)
        return _cell(r, c, fmt)
    w = {c: max(len(head[c]), max(len(cellstr(r, c)) for r in rows)) for c in cols}
    print(f"\n=== {title} ===")
    print("  ".join(f"{head[c]:>{w[c]}}" for c in cols))
    print("-" * (sum(w.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        print("  ".join(f"{cellstr(r, c):>{w[c]}}" for c in cols))


def _write_csv(rows, path, keys):
    cols = list(keys) + ["n"] + [f"{k}_{stat}" for k, *_ in AGG for stat in ("mean", "std")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"[out] wrote {path}")


def _write_samples_csv(samples, path):
    cols = ["approach", "g", "run", "loops", "loops_cmd"] + [k for k, *_ in AGG]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for s in samples:
            w.writerow([s.get(c, "") for c in cols])
    print(f"[out] wrote {path}")


def _write_latex(rows, path, keys):
    arrow = {True: "$\\uparrow$", False: "$\\downarrow$"}
    head = [k for k in keys] + [f"{h}{arrow.get(b, '')}" for _k, h, _f, b in AGG]
    lines = ["\\begin{tabular}{" + "l" * len(keys) + "r" * len(AGG) + "}",
             "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for r in rows:
        cells = [str(r[k]) for k in keys]
        for k, _h, fmt, _b in AGG:
            m, s = r.get(k + "_mean"), r.get(k + "_std")
            if m is None or (isinstance(m, float) and math.isnan(m)):
                cells.append("-")
            elif r.get("n", 1) > 1:
                cells.append(f"${fmt.format(m)}{{\\scriptstyle\\,\\pm {fmt.format(s)}}}$")
            else:
                cells.append(f"${fmt.format(m)}$")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[out] wrote {path}")


# ---------------------------------- CLI ------------------------------------- #
def _parse_approaches(items):
    if not items:
        return DEFAULT_APPROACHES
    return dict(it.split("=", 1) for it in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["/home/yigit/projects/pemp/outputs/sim/mixing/1779633078"],
                    help="one or more training-run dirs (each = a seed)")
    ap.add_argument("--g", type=float, nargs="+", default=[0.333, 0.5, 0.667, 0.833, 1.0])
    ap.add_argument("--approaches", nargs="+", default=None,
                    help="label=filekey pairs, e.g. PEMP=pe CNMP=bare ProMP=promp")
    ap.add_argument("--n-draws", type=int, default=1,
                    help="eval draws per (run,g): each varies the model context AND a "
                         "right-biased random start offset. >1 gives eval-side error bars.")
    ap.add_argument("--n-ctx", type=int, default=10)
    ap.add_argument("--pe-scaler", type=float, default=0.2, help="PEMP PE scaler (match training)")
    ap.add_argument("--start-dx", type=float, nargs=2, default=[0.10, 0.30],
                    help="rightward start-offset range (used when --n-draws>1)")
    ap.add_argument("--start-dz", type=float, default=0.12, help="lateral start-offset half-range")
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--off-grid", action="store_true",
                    help="condition models on the REQUESTED g instead of snapping to "
                         "the nearest training g (use for off-training-grid evaluation)")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    approaches = _parse_approaches(args.approaches)
    snap_g = not args.off_grid
    print(f"[cfg] snap_g={snap_g}  ({'off-grid' if not snap_g else 'on-grid'} evaluation)")

    env = make_env()
    samples = collect_samples(env, args.runs, args.g, approaches, n_draws=args.n_draws,
                              n_ctx=args.n_ctx, pe_scaler=args.pe_scaler,
                              start_dx=tuple(args.start_dx), start_dz=args.start_dz,
                              n_bins=args.n_bins, snap_g=snap_g)
    env.close()
    if not samples:
        raise SystemExit("no samples produced (check --runs / models)")

    by_g = aggregate(samples, ["approach", "g"])
    overall = aggregate(samples, ["approach"])
    n_runs = len({s["run"] for s in samples})
    n_per_cell = n_runs * args.n_draws  # samples averaged per (approach,g)

    _print_agg(by_g, ["approach", "g"],
               f"Per-frequency: mean+-std over {n_runs} run(s) x {args.n_draws} draw(s) = {n_per_cell}")
    _print_agg(overall, ["approach"],
               f"Across frequencies: mean+-std over {n_runs} run(s) x {len(args.g)} g x {args.n_draws} draw(s)")
    if n_per_cell < 2:
        print("\n[note] per-(approach,g) n=1 -> std is 0. Add seeds (--runs r1 r2 ...) for "
              "training-seed error bars and/or --n-draws>1 for eval-side (context + start "
              "position) error bars. Across-frequency std also includes the g-sweep.")

    prefix = args.out_prefix or os.path.join(args.runs[0], "rollout", "mixing_metrics")
    _write_samples_csv(samples, prefix + "_samples.csv")
    _write_csv(by_g, prefix + "_byg.csv", ["approach", "g"])
    _write_latex(by_g, prefix + "_byg.tex", ["approach", "g"])
    _write_csv(overall, prefix + "_overall.csv", ["approach"])
    _write_latex(overall, prefix + "_overall.tex", ["approach"])


if __name__ == "__main__":
    main()
