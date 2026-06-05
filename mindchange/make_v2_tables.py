"""Regenerate the v2 comparison LaTeX tables from the (corrected) aggregated CSVs.

Produces, after compare/aggregate have been re-run with the ProMP conditioning fix:
  <simple_base>/comparison_synth_v2_aggregated_byapproach.tex   (4-row, bold, 2dp)
  <complex_base>/comparison_synth_v2_aggregated_byapproach.tex  (4-row, bold, 2dp)
  <freq_root>/comparison_synth_v2_simple_vs_complex.tex         (merged, bold, 2dp)

Conventions requested:
  * rows in order ProMP+GMM, CNMP, WMP, PEMP
  * best (lowest mean MSE) per column in \textbf{}
  * values rounded to 2 decimals, "mean +- std"
  * ProMP+GMM = best variant per dataset (lowest mean MSE across n_ctx columns)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

FREQ_ROOT = Path("/home/yigit/projects/pemp/outputs/comparison/mind_change/freq")
BASES = {
    "Simple":  FREQ_ROOT / "bare_pe_promp_gmm",
    "Complex": FREQ_ROOT / "combined/bare_pe_promp_gmm",
}
AGG_CSV = "comparison_synth_v2_aggregated_byapproach.csv"
N_CTX = [1, 3, 10, 20]
ROW_ORDER = ["ProMP+GMM", "CNMP", "WMP", "PEMP"]   # display order


def _n_weights(idx):
    return idx + 2 if idx < 28 else (40 if idx == 28 else 50)


def _read(csv_path):
    """{approach: {n_ctx: (mean, std, n)}} from an aggregated byapproach CSV path."""
    out = {}
    for r in csv.DictReader(Path(csv_path).open()):
        if r["mse_mean"] in ("", "nan"):
            continue
        out.setdefault(r["approach"], {})[int(r["n_ctx"])] = (
            float(r["mse_mean"]), float(r["mse_std"]), int(r["n"]))
    return out


def _best_promp(data):
    promps = [(k, v) for k, v in data.items() if k.startswith("ProMP_")]
    def score(kv):
        ms = [m for (m, _, _) in kv[1].values() if not math.isnan(m)]
        return sum(ms) / len(ms) if ms else float("inf")
    return min(promps, key=score)


def _resolve_rows(data):
    """Map display labels -> source key in `data`, choosing the best ProMP variant."""
    best_label, _ = _best_promp(data)
    src = {"ProMP+GMM": best_label, "CNMP": "CNMP", "WMP": "WMP", "PEMP": "PEMP"}
    return src, best_label


def _fmt(mean, std, bold):
    """Plain 2-decimal 'mean +- std' (narrow, fits page width).  Safe here because
    the smallest cell is ~0.01 so nothing collapses to 0.00 (the sawtooth/complex
    floors are well above the sub-0.01 range that forced scientific notation on the
    old clean-sine data).  Bold wraps the mean for the per-column best."""
    m = f"\\mathbf{{{mean:.2f}}}" if bold else f"{mean:.2f}"
    return f"${m}{{\\scriptstyle\\,\\pm {std:.2f}}}$"


def _col_min_label(data, src, N):
    """Display label with the lowest mean MSE at this n_ctx (full precision)."""
    best, best_v = None, float("inf")
    for disp, key in src.items():
        v = data.get(key, {}).get(N)
        if v and not math.isnan(v[0]) and v[0] < best_v:
            best, best_v = disp, v[0]
    return best


def _sample_count(data):
    """Per-cell sample count n (runs x test_trajs); read from any present cell."""
    for v in data.values():
        for cell in v.values():
            return cell[2]
    return 0


def write_per_base(label, data, src, best_label, out_tex):
    nw = _n_weights(int(best_label.split("_")[1]))
    lines = [
        f"% v2 comparison ({label} synthetic), corrected ProMP+GMM conditioning.",
        f"% Rows: ProMP+GMM/CNMP/WMP/PEMP; best per column in bold; 2 decimals.",
        f"% ProMP+GMM = {best_label} (n_weights_per_dim={nw}), best mean MSE across n_ctx.",
        f"% per-cell n = {_sample_count(data)} (runs x test_trajs).",
        "\\begin{tabular}{l" + "r" * len(N_CTX) + "}",
        "\\toprule",
        "Approach & " + " & ".join(f"$n_{{\\mathrm{{ctx}}}}={N}$" for N in N_CTX) + " \\\\",
        "\\midrule",
    ]
    col_best = {N: _col_min_label(data, src, N) for N in N_CTX}
    for disp in ROW_ORDER:
        key = src[disp]
        cells = [disp]
        for N in N_CTX:
            v = data.get(key, {}).get(N)
            cells.append("-" if v is None else _fmt(v[0], v[1], col_best[N] == disp))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    Path(out_tex).write_text("\n".join(lines))
    print(f"  wrote {out_tex}  (ProMP+GMM={best_label}, n_w={nw})")


def write_merged(loaded, out_path):
    ncol = len(N_CTX)
    spec = "l" + (" " + "r" * ncol) * 2
    head_groups = ("Approach & \\multicolumn{%d}{c}{Simple} & \\multicolumn{%d}{c}{Complex} \\\\"
                   % (ncol, ncol))
    cmid = ("\\cmidrule(lr){2-%d} \\cmidrule(lr){%d-%d}"
            % (1 + ncol, 2 + ncol, 1 + 2 * ncol))
    subhead = (" & " + " & ".join(f"$n_{{\\mathrm{{ctx}}}}={N}$" for N in N_CTX)
               + " & " + " & ".join(f"$n_{{\\mathrm{{ctx}}}}={N}$" for N in N_CTX) + " \\\\")
    lines = [
        "% Merged v2 comparison: Simple and Complex synthetic.",
        "% Corrected ProMP+GMM conditioning; rows ProMP+GMM/CNMP/WMP/PEMP;",
        "% best per column in bold; 2 decimals; mean +- std over (runs x test_trajs).",
    ]
    for label in ("Simple", "Complex"):
        data, src, best_label = loaded[label]
        nw = _n_weights(int(best_label.split("_")[1]))
        lines.append(f"%   {label}: ProMP+GMM={best_label} (n_weights_per_dim={nw}); "
                     f"per-cell n={_sample_count(data)}")
    lines += ["\\begin{tabular}{" + spec + "}", "\\toprule", head_groups, cmid, subhead,
              "\\midrule"]
    # precompute per-(dataset, N) best label
    col_best = {}
    for label in ("Simple", "Complex"):
        data, src, _ = loaded[label]
        for N in N_CTX:
            col_best[(label, N)] = _col_min_label(data, src, N)
    for disp in ROW_ORDER:
        cells = [disp]
        for label in ("Simple", "Complex"):
            data, src, _ = loaded[label]
            key = src[disp]
            for N in N_CTX:
                v = data.get(key, {}).get(N)
                cells.append("-" if v is None
                             else _fmt(v[0], v[1], col_best[(label, N)] == disp))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    Path(out_path).write_text("\n".join(lines))
    print(f"  wrote {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simple-csv", default=str(BASES["Simple"] / AGG_CSV),
                    help="aggregated byapproach CSV for the Simple columns")
    ap.add_argument("--complex-csv", default=str(BASES["Complex"] / AGG_CSV),
                    help="aggregated byapproach CSV for the Complex columns")
    ap.add_argument("--out", default=str(FREQ_ROOT / "comparison_synth_v2_simple_vs_complex.tex"),
                    help="output path for the merged tex table")
    ap.add_argument("--per-base", action="store_true",
                    help="also write a per-base byapproach.tex next to each input CSV")
    args = ap.parse_args()

    csvs = {"Simple": Path(args.simple_csv), "Complex": Path(args.complex_csv)}
    loaded = {}
    for label in ("Simple", "Complex"):
        data = _read(csvs[label])
        src, best_label = _resolve_rows(data)
        if args.per_base:
            write_per_base(label, data, src, best_label,
                           csvs[label].parent / "comparison_synth_v2_aggregated_byapproach.tex")
        loaded[label] = (data, src, best_label)
    write_merged(loaded, Path(args.out))
    # echo the final merged numbers for a quick sanity read
    print("\n=== final merged values (mean MSE) ===")
    for disp in ROW_ORDER:
        row = [f"{disp:10s}"]
        for label in ("Simple", "Complex"):
            data, src, _ = loaded[label]
            key = src[disp]
            row.append("| " + label[:4] + ": " +
                       " ".join(f"{data[key][N][0]:.4f}" for N in N_CTX))
        print("  ".join(row))


if __name__ == "__main__":
    main()
