"""Aggregate compare_synthetic results across every timestamp folder under a base
directory.  For each <base>/<ts>/, ensures <ts>/comparison_synth_samples.csv exists
(runs compare_synthetic.py on it if not), then concatenates all samples and
aggregates MSE by (approach, n_ctx) over the *timestamp x test_traj* product.

Usage:
  ~/sw/anaconda3/envs/pemp-gpu/bin/python mindchange/aggregate_synthetic.py \\
      --base-dir outputs/comparison/mind_change/freq/bare_pe_promp_gmm
  ~/sw/anaconda3/envs/pemp-gpu/bin/python mindchange/aggregate_synthetic.py \\
      --base-dir outputs/comparison/mind_change/freq/combined/bare_pe_promp_gmm
"""
from __future__ import annotations

import os
import re
import csv
import math
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_BASE = "/home/yigit/projects/pemp/outputs/comparison/mind_change/freq/bare_pe_promp_gmm"
PY = sys.executable  # same interpreter that ran us; expected to be pemp-gpu


def _sort_key(label):
    if label == "CNMP": return (0,)
    if label == "PEMP": return (1,)
    if label == "WMP":  return (2,)
    m = re.match(r"ProMP_(\d+)", label)
    return (3, int(m.group(1))) if m else (4, label)


def _tag_from_cond(cond_file):
    """`conditioning.npz` -> ``; `conditioning_v2.npz` -> `_v2`; arbitrary names
    fall back to `_<basename>`.  Mirrors compare_synthetic.tag_from_cond so the
    aggregator's cached samples/output files match the per-ts comparator's."""
    base = os.path.splitext(os.path.basename(cond_file))[0]
    if base == "conditioning":
        return ""
    if base.startswith("conditioning_"):
        return base[len("conditioning"):]
    return "_" + base


def _ensure_samples(ts_dir, force=False, cond_file="conditioning.npz"):
    tag = _tag_from_cond(cond_file)
    sp = os.path.join(ts_dir, f"comparison_synth{tag}_samples.csv")
    if os.path.exists(sp) and not force:
        return sp, False
    cmd = [PY, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "compare_synthetic.py"),
           "--ts", ts_dir, "--cond-file", cond_file]
    subprocess.run(cmd, check=True)
    return sp, True


def _read_csv_dicts(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _aggregate(rows):
    """Group MSE samples by (approach, n_ctx); return mean+-std rows."""
    groups = {}
    for r in rows:
        key = (r["approach"], int(r["n_ctx"]))
        groups.setdefault(key, []).append(float(r["mse"]))
    out = []
    for (a, N), vs in groups.items():
        v = np.array([x for x in vs if not math.isnan(x)], float)
        out.append({"approach": a, "n_ctx": N, "n": len(v),
                    "mse_mean": float(v.mean()) if len(v) else float("nan"),
                    "mse_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0})
    return sorted(out, key=lambda r: (_sort_key(r["approach"]), r["n_ctx"]))


def _print_table(agg, n_ctx_values, n_ts):
    by = {(r["approach"], r["n_ctx"]): r for r in agg}
    approaches = sorted({r["approach"] for r in agg}, key=_sort_key)
    head = ["approach", "n_obs"] + [f"n={N}" for N in n_ctx_values]
    cells = []
    for a in approaches:
        row = [a, str(by[(a, n_ctx_values[0])]["n"])]
        for N in n_ctx_values:
            r = by.get((a, N))
            if r is None or math.isnan(r["mse_mean"]):
                row.append("-")
            elif r["n"] > 1:
                row.append(f"{r['mse_mean']:.4f}+-{r['mse_std']:.4f}")
            else:
                row.append(f"{r['mse_mean']:.4f}")
        cells.append(row)
    widths = [max(len(head[i]), max(len(r[i]) for r in cells)) for i in range(len(head))]
    line = "  ".join(f"{head[i]:>{widths[i]}}" for i in range(len(head)))
    print(f"\n=== Aggregated test MSE over {n_ts} timestamp(s) x test_trajs ===")
    print(line); print("-" * len(line))
    for r in cells:
        print("  ".join(f"{r[i]:>{widths[i]}}" for i in range(len(head))))


def _write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[out] wrote {path}")


def _write_latex(agg, n_ctx_values, path):
    by = {(r["approach"], r["n_ctx"]): r for r in agg}
    approaches = sorted({r["approach"] for r in agg}, key=_sort_key)
    head = ["Approach"] + [f"$n_{{ctx}}={N}$" for N in n_ctx_values]
    lines = ["\\begin{tabular}{l" + "r" * len(n_ctx_values) + "}",
             "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for a in approaches:
        cs = [a]
        for N in n_ctx_values:
            r = by.get((a, N))
            if r is None or math.isnan(r["mse_mean"]):
                cs.append("-")
            elif r["n"] > 1:
                cs.append(f"${r['mse_mean']:.4f}{{\\scriptstyle\\,\\pm {r['mse_std']:.4f}}}$")
            else:
                cs.append(f"${r['mse_mean']:.4f}$")
        lines.append(" & ".join(cs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[out] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE)
    ap.add_argument("--cond-file", default="conditioning.npz",
                    help="protocol filename inside each ts dir (e.g. conditioning_v2.npz "
                         "for the {1,3,10,20} protocol).  Per-ts samples are cached as "
                         "comparison_synth{tag}_samples.csv where tag derives from this; "
                         "aggregated outputs use the same tag so v1 and v2 coexist.")
    ap.add_argument("--force", action="store_true",
                    help="re-run per-timestamp compare_synthetic even if samples CSV exists")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--timestamps", nargs="+", default=None,
                    help="restrict to these run-folder names (default: every dir under --base-dir). "
                         "Use to aggregate over a chosen subset of runs/seeds.")
    args = ap.parse_args()

    tag = _tag_from_cond(args.cond_file)
    base = Path(args.base_dir).resolve()
    ts_dirs = [d for d in sorted(base.iterdir()) if d.is_dir()]
    if args.timestamps is not None:
        want = set(args.timestamps)
        ts_dirs = [d for d in ts_dirs if d.name in want]
        missing = want - {d.name for d in ts_dirs}
        if missing:
            raise SystemExit(f"--timestamps not found under {base}: {sorted(missing)}")
    print(f"[base] {base}  ({len(ts_dirs)} candidate folders)  cond={args.cond_file} tag='{tag}'")

    all_rows = []
    ran_count = 0
    cached_count = 0
    skipped_count = 0
    for d in ts_dirs:
        if not (d / args.cond_file).exists() or not (d / "saved_models").exists():
            print(f"[skip] {d.name}: no {args.cond_file} or saved_models/")
            skipped_count += 1
            continue
        sp, ran = _ensure_samples(str(d), force=args.force, cond_file=args.cond_file)
        if ran:
            print(f"[ran ] {d.name}: produced {os.path.basename(sp)}")
            ran_count += 1
        else:
            print(f"[hit ] {d.name}: cached {os.path.basename(sp)}")
            cached_count += 1
        for r in _read_csv_dicts(sp):
            r["timestamp"] = d.name
            all_rows.append(r)
    n_ts = ran_count + cached_count
    print(f"[total] {len(all_rows)} samples from {n_ts} timestamp(s) "
          f"({ran_count} freshly run, {cached_count} cached, {skipped_count} skipped)")
    if not all_rows:
        raise SystemExit("no samples collected")

    agg = _aggregate(all_rows)
    n_ctx_values = sorted({r["n_ctx"] for r in agg})
    _print_table(agg, n_ctx_values, n_ts)

    prefix = args.out_prefix or str(base / f"comparison_synth{tag}_aggregated")
    _write_csv(all_rows, prefix + "_samples.csv",
               ["timestamp", "approach", "n_ctx", "traj", "mse"])
    _write_csv(agg, prefix + "_byapproach.csv",
               ["approach", "n_ctx", "n", "mse_mean", "mse_std"])
    _write_latex(agg, n_ctx_values, prefix + "_byapproach.tex")


if __name__ == "__main__":
    main()
