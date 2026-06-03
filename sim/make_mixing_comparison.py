"""Tile per-frequency Mixing rollout videos into a comparison grid:
rows = approaches (top->bottom in the order given), columns = stir frequency.

Default = the original PEMP-on-top / CNMP-on-bottom grid.  Pass --approaches to
build any N-row grid from the rollouts present in a run's rollout/ dir.
Pure ffmpeg (hstack/vstack) over the mp4s -- no re-simulation.

Examples:
  python sim/make_mixing_comparison.py --run outputs/sim/mixing/<ts>          # PEMP/CNMP
  python sim/make_mixing_comparison.py --run <ts> \\
      --approaches CNMP=bare WMP=wmp PEMP=pe                                  # 3 rows
  python sim/make_mixing_comparison.py --run <ts> \\
      --approaches WMP=wmp PEMP=pe --g 0.4 0.9                                # subset of g
"""
import os
import re
import glob
import argparse
import subprocess

DEFAULT_RUN = "/home/yigit/projects/pemp/outputs/sim/mixing/1779529990"
DEFAULT_APPROACHES = ["PEMP=pe", "CNMP=bare"]


def _parse_approaches(items):
    pairs = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected LABEL=filekey, got {it!r}")
        label, key = it.split("=", 1)
        pairs.append((label, key))
    return pairs


def find_videos(rollout_dir, filekeys, suffix="", g_filter=None):
    """Return [(g_tag, [path_for_filekey0, path_for_filekey1, ...])] for freqs
    present under EVERY filekey.  If g_filter is given (set of g_tags like {'0p400'})
    restrict to those tags."""
    per_key = {k: {} for k in filekeys}
    for f in glob.glob(os.path.join(rollout_dir, f"mixing_*_g*{suffix}.mp4")):
        bn = os.path.basename(f)
        for k in filekeys:
            m = re.match(rf"^mixing_{re.escape(k)}_g(\d+p\d+){re.escape(suffix)}\.mp4$", bn)
            if m:
                per_key[k][m.group(1)] = f
                break
    common = set.intersection(*[set(per_key[k]) for k in filekeys]) if per_key else set()
    if g_filter is not None:
        common &= set(g_filter)
    rows = sorted(common, key=lambda s: float(s.replace("p", ".")))
    return [(g, [per_key[k][g] for k in filekeys]) for g in rows]


def _g_to_tag(g):
    return f"{g:.3f}".replace(".", "p")


def _tag_to_g(tag):
    return float(tag.replace("p", "."))


def _safe_label_for_path(s):
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower() or "x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--variant", choices=["matplotlib", "ggui"], default="matplotlib")
    ap.add_argument("--approaches", nargs="+", default=DEFAULT_APPROACHES,
                    help="LABEL=filekey, top-to-bottom, e.g. CNMP=bare WMP=wmp PEMP=pe")
    ap.add_argument("--g", type=float, nargs="+", default=None,
                    help="restrict to these frequencies (default: all present)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suffix = "_ggui" if args.variant == "ggui" else ""
    pairs = _parse_approaches(args.approaches)
    labels = [lbl for lbl, _ in pairs]
    keys = [k for _, k in pairs]
    g_filter = {_g_to_tag(g) for g in args.g} if args.g else None

    rollout_dir = os.path.join(args.run, "rollout")
    rows = find_videos(rollout_dir, keys, suffix=suffix, g_filter=g_filter)
    if not rows:
        raise SystemExit(f"no rollouts found in {rollout_dir} covering all of "
                         f"filekeys={keys}{' for the requested g' if g_filter else ''}")
    n_cols = len(rows)
    n_rows = len(keys)
    print(f"[cmp] {n_rows} rows ({' x '.join(labels)})  x  {n_cols} cols "
          f"(g={[_tag_to_g(g) for g, _ in rows]})")

    # Build inputs in row-major order: row0_col0, row0_col1, ..., row1_col0, ...
    inputs = []
    for r in range(n_rows):
        for _g, paths in rows:
            inputs.append(paths[r])

    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    row_labels = [f"r{r}" for r in range(n_rows)]
    parts = []
    for r in range(n_rows):
        in_refs = "".join(f"[{r * n_cols + c}:v]" for c in range(n_cols))
        parts.append(f"{in_refs}hstack=inputs={n_cols}[{row_labels[r]}]")
    if n_rows == 1:
        fc = parts[0].rsplit("[", 1)[0] + "[v]"  # no vstack needed
    else:
        vstack_in = "".join(f"[{x}]" for x in row_labels)
        parts.append(f"{vstack_in}vstack=inputs={n_rows}[v]")
        fc = ";".join(parts)
    label_slug = "_".join(_safe_label_for_path(lbl) for lbl in labels)
    g_slug = ("_g" + "_".join(g for g, _ in rows)) if g_filter else ""
    out = args.out or os.path.join(rollout_dir, f"comparison_{label_slug}{g_slug}{suffix}.mp4")
    cmd += ["-filter_complex", fc, "-map", "[v]", out]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[cmp] wrote {out}")


if __name__ == "__main__":
    main()
