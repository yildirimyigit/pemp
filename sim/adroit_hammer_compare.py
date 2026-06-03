"""Quantitative comparison of Adroit-hammer action trajectories from N approaches,
with mean+-std over training-run seeds (and optional eval draws).

Mirror of sim/compare_mixing.py.  Each approach saves its generated action trajectory
per (run, commanded-strikes [, draw]) as an .npz file; this script replays each in the
env, computes precision-oriented metrics, and writes aggregate tables.

Expected file layout (per training run):
    <run_dir>/adroit_rollout/<filekey>_n<strikes>[_d<draw>].npz
where each .npz holds at minimum:
    actions     (T, 26) float32   joint actions in [-1, 1]
    num_strikes int               commanded loops (also recoverable from filename)
Optional env-config fields (default to the controller's values if missing):
    nail_start_qpos, nail_frictionloss, warmup_steps

Core API:
    samples = collect_samples(env, runs, strikes_list, approaches, n_draws=1)
    by_n    = aggregate(samples, ["approach", "strikes"])   # mean+-std per condition
    overall = aggregate(samples, ["approach"])              # mean+-std overall

The approach map is the analogue of the FluidLab mixing comparison:
    --approaches PEMP=pemp CNMP=cnmp ProMP=promp MyMethod=mm
PEMP and CNMP are paired by env reset seed per (strikes, draw) -- their evaluations
share the same initial conditions so the comparison is paired across approaches.

Metrics (v lower is better, ^ higher is better)
  OUTCOME:    NailDist^ (total nail qpos driven)   Success^ (goal_distance < 5mm)
  STRIKES:    HitRate^ (realized / commanded)      LoopErr v (|realized - commanded|)
              NailPer^ (mean nail-qpos delta between detected strikes)
  PRECISION:  TimeStd v (std of first-contact step within commanded cycle)
              ToolCyErr v (max cycle-to-cycle deviation of hammer-head position)
              RelCyErr  v (same, on hammer-relative-to-nail trajectory)
              StrikeXY  v (std of hammer xy at first-contact -- targeting precision)
              ImpactV ^ (mean ||d(tool)/dt|| at first contact -- impact speed)
  CONTROL:    Smooth v (dimensionless mean jerk)   Effort v (mean ||a||)

Run in the pemp-gpu conda env (mujoco + gymnasium_robotics):
    ~/sw/anaconda3/envs/pemp-gpu/bin/python sim/adroit_hammer_compare.py \
        --runs <run_dir1> <run_dir2> <run_dir3>  --strikes 3 4 5 6
"""
from __future__ import annotations

import os
import sys
import re
import math
import csv
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Registers AdroitHandHammer-vPEMPTap and exposes env helpers.  Reuses the
# controller's task_positions/hammer_nail_contact/cycle_max_error so cycle-error
# numbers stay directly comparable to the saved demo metadata.
import adroit_hand_hammer_tap  # noqa: F401
from adroit_hand_hammer_tap import ENV_ID, NAIL_QPOS_INDEX
from adroit_clean_tap_hammer_controller import (
    task_positions, hammer_nail_contact, goal_distance,
)


def _cycle_max_error(values, strikes):
    """Like the controller's cycle_max_error but tolerant of T not divisible by
    `strikes` -- truncates the leftover tail before reshape.  Needed because
    learned models output a fixed-length T trajectory regardless of `strikes`."""
    if strikes < 2 or len(values) == 0:
        return 0.0
    cycle_steps = len(values) // strikes
    if cycle_steps == 0:
        return 0.0
    n = strikes * cycle_steps
    cycles = np.asarray(values)[:n].reshape(strikes, cycle_steps, -1)
    return float(np.max(np.abs(cycles - cycles[0])))

DEFAULT_APPROACHES = {"PEMP": "pemp", "CNMP": "cnmp"}
DEFAULT_ENV_CONFIG = dict(nail_start_qpos=0.04, nail_frictionloss=15.0, warmup_steps=40)
GOAL_THRESH = 0.005  # m -- nail considered "driven" when goal_distance < this

# (key, header, fmt, higher_is_better)
AGG = [
    ("nail_dist",       "NailDist",  "{:.4f}", True),
    ("success",         "Success",   "{:.2f}", True),
    ("hit_rate",        "HitRate",   "{:.3f}", True),
    ("loop_err",        "LoopErr",   "{:.2f}", False),
    ("nail_per_strike", "NailPer",   "{:.4f}", True),
    ("timing_std",      "TimeStd",   "{:.2f}", False),
    ("tool_cycle_err",  "ToolCyErr", "{:.4f}", False),
    ("rel_cycle_err",   "RelCyErr",  "{:.4f}", False),
    ("strike_xy_std",   "StrikeXY",  "{:.4f}", False),
    ("impact_vel",      "ImpactV",   "{:.4f}", True),
    ("smooth",          "Smooth",    "{:.3f}", False),
    ("effort",          "Effort",    "{:.3f}", False),
]


# ----------------------------- env + replay --------------------------------- #
def make_env(seed=0, max_episode_steps=600, **env_cfg):
    """Headless Adroit hammer env (one per process; reset between samples)."""
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    cfg = {**DEFAULT_ENV_CONFIG, **env_cfg}
    env = gym.make(ENV_ID, render_mode=None, max_episode_steps=max_episode_steps, **cfg)
    env.reset(seed=seed)
    return env


def replay(env, actions, reset_seed):
    """Reset the env, step actions, record per-step tool/nail/contact/nail-qpos."""
    env.reset(seed=reset_seed)
    n = len(actions)
    tool = np.zeros((n, 3)); nail = np.zeros((n, 3))
    nailq = np.zeros(n); contacts = np.zeros(n, dtype=bool)
    for i, a in enumerate(actions):
        env.step(np.clip(a, -1.0, 1.0))
        t, m, _ = task_positions(env)
        tool[i] = t; nail[i] = m
        contacts[i] = hammer_nail_contact(env)
        nailq[i] = float(env.unwrapped.data.qpos[NAIL_QPOS_INDEX])
    return dict(tool=tool, nail=nail, nailq=nailq, contacts=contacts,
                goal_dist=goal_distance(env))


# ----------------------------- metric helpers ------------------------------- #
def _strike_starts(contacts):
    """Step indices of every contact rising edge (first contact of each strike)."""
    if len(contacts) < 2:
        return np.array([], dtype=int)
    return np.where(np.diff(contacts.astype(int)) == 1)[0] + 1


def _smoothness(a):
    mj = np.linalg.norm(np.diff(a, n=3, axis=0), axis=1).mean()
    return float(mj / (np.linalg.norm(a, axis=1).mean() + 1e-12))


def compute_metrics(actions, st, commanded):
    """Compute the publication metrics from the replayed state."""
    a = np.asarray(actions, dtype=np.float64)
    T = len(a)
    strikes = _strike_starts(st["contacts"])
    realized = int(len(strikes))
    cycle = max(T // max(commanded, 1), 1)

    nail_dist = float(st["nailq"][-1] - st["nailq"][0])
    success = float(st["goal_dist"] < GOAL_THRESH)
    hit_rate = float(realized / max(commanded, 1))
    loop_err = float(abs(realized - commanded))

    if realized > 0:
        cycle_pos = strikes % cycle
        timing_std = float(cycle_pos.std()) if realized > 1 else 0.0
        xy = st["tool"][strikes, :2]
        strike_xy_std = float(np.linalg.norm(xy - xy.mean(0), axis=1).std()) if realized > 1 else 0.0
        vels = np.linalg.norm(np.diff(st["tool"], axis=0), axis=1)
        impact_vel = float(vels[np.clip(strikes - 1, 0, len(vels) - 1)].mean())
        per_strike = (np.diff(st["nailq"][strikes]) if realized > 1
                      else np.array([st["nailq"][-1] - st["nailq"][strikes[0]]]))
        nail_per_strike = float(per_strike.mean())
    else:
        timing_std = float("nan"); strike_xy_std = float("nan")
        impact_vel = float("nan"); nail_per_strike = 0.0

    tool_cycle_err = float(_cycle_max_error(st["tool"], commanded)) if commanded >= 2 else 0.0
    rel_cycle_err = float(_cycle_max_error(st["tool"] - st["nail"], commanded)) if commanded >= 2 else 0.0

    return dict(
        nail_dist=nail_dist, success=success, hit_rate=hit_rate, loop_err=loop_err,
        nail_per_strike=nail_per_strike, timing_std=timing_std,
        tool_cycle_err=tool_cycle_err, rel_cycle_err=rel_cycle_err,
        strike_xy_std=strike_xy_std, impact_vel=impact_vel,
        smooth=_smoothness(a), effort=float(np.linalg.norm(a, axis=1).mean()),
        realized=realized, commanded=commanded,
    )


# ----------------------------- file loading --------------------------------- #
_RE_N = re.compile(r"_n(\d+)")


def _load_actions(path):
    """Return (actions, commanded_num_strikes, optional env_cfg)."""
    with np.load(path, allow_pickle=False) as data:
        actions = data["actions"].astype(np.float32)
        commanded = int(data["num_strikes"].item()) if "num_strikes" in data.files else None
        env_cfg = {k: float(data[k].item()) for k in
                   ("nail_start_qpos", "nail_frictionloss") if k in data.files}
        if "warmup_steps" in data.files:
            env_cfg["warmup_steps"] = int(data["warmup_steps"].item())
    if commanded is None:
        m = _RE_N.search(os.path.basename(path))
        commanded = int(m.group(1)) if m else 0
    return actions, commanded, env_cfg


def _file_for(run, key, strikes, draw, n_draws):
    name = f"{key}_n{strikes}.npz" if n_draws == 1 else f"{key}_n{strikes}_d{draw}.npz"
    return os.path.join(run, "adroit_rollout", name)


# ------------------------------- evaluation --------------------------------- #
def evaluate(env, action_file, reset_seed=0):
    actions, commanded, _env_cfg = _load_actions(action_file)
    st = replay(env, actions, reset_seed=reset_seed)
    m = compute_metrics(actions, st, commanded)
    m["action_file"] = os.path.basename(action_file)
    return m


def collect_samples(env, runs, strikes_list, approaches, n_draws=1):
    """One sample per (run, approach, strikes, draw).  Per (strikes, draw) all
    approaches use the SAME env reset seed (paired comparison)."""
    samples = []
    for run in runs:
        rname = os.path.basename(run.rstrip("/"))
        for strikes in strikes_list:
            for j in range(n_draws):
                reset_seed = int(strikes) * 1000 + j
                for label, key in approaches.items():
                    p = _file_for(run, key, strikes, j, n_draws)
                    if not os.path.exists(p):
                        print(f"[skip] {label} n={strikes} run={rname} draw={j}: missing {os.path.basename(p)}")
                        continue
                    print(f"[eval] {label} n={strikes} run={rname} draw={j}")
                    m = evaluate(env, p, reset_seed=reset_seed)
                    m.update(approach=label, strikes=strikes, run=rname, draw=j)
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
    out.sort(key=lambda r: (str(r.get("approach", "")), r.get("strikes", 0)))
    return out


# -------------------------------- formatting -------------------------------- #
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
    cols = ["approach", "strikes", "run", "draw", "realized", "commanded",
            "action_file"] + [k for k, *_ in AGG]
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
    ap.add_argument("--runs", nargs="+", required=True,
                    help="one or more training-run dirs (each = a training-seed)")
    ap.add_argument("--strikes", type=int, nargs="+", default=[3, 4, 5, 6],
                    help="commanded strike counts to evaluate")
    ap.add_argument("--approaches", nargs="+", default=None,
                    help="label=filekey pairs, e.g. PEMP=pemp CNMP=cnmp ProMP=promp")
    ap.add_argument("--n-draws", type=int, default=1,
                    help=">1 expects per-draw files <key>_n<strikes>_d<j>.npz")
    ap.add_argument("--max-steps", type=int, default=600,
                    help="env max_episode_steps; raise if action trajectories are longer")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    approaches = _parse_approaches(args.approaches)

    env = make_env(seed=0, max_episode_steps=args.max_steps)
    samples = collect_samples(env, args.runs, args.strikes, approaches, n_draws=args.n_draws)
    env.close()
    if not samples:
        raise SystemExit("no samples produced (check --runs / file layout)")

    by_n = aggregate(samples, ["approach", "strikes"])
    overall = aggregate(samples, ["approach"])
    n_runs = len({s["run"] for s in samples})
    n_per_cell = n_runs * args.n_draws

    _print_agg(by_n, ["approach", "strikes"],
               f"Per-strike-count: mean+-std over {n_runs} run(s) x {args.n_draws} draw(s) = {n_per_cell}")
    _print_agg(overall, ["approach"],
               f"Across strikes: mean+-std over {n_runs} run(s) x {len(args.strikes)} strikes x {args.n_draws} draw(s)")
    if n_per_cell < 2:
        print("\n[note] per-(approach,strikes) n=1 -> std is 0; add more runs or --n-draws "
              "for error bars. Across-strikes std also includes the strike-count sweep.")

    prefix = args.out_prefix or os.path.join(args.runs[0], "adroit_rollout", "adroit_metrics")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    _write_samples_csv(samples, prefix + "_samples.csv")
    _write_csv(by_n, prefix + "_byn.csv", ["approach", "strikes"])
    _write_latex(by_n, prefix + "_byn.tex", ["approach", "strikes"])
    _write_csv(overall, prefix + "_overall.csv", ["approach"])
    _write_latex(overall, prefix + "_overall.tex", ["approach"])


if __name__ == "__main__":
    main()
