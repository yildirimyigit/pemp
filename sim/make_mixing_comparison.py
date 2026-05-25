"""Tile the per-frequency Mixing rollout videos into one PE-vs-bare comparison grid
(top row = PEMP, bottom row = bare CNMP; columns = stir frequency, left->right).

Pure ffmpeg (hstack/vstack) over the existing mp4s in a run's rollout/ dir -- no
re-simulation.  Reusable: rerun it after training new models / new rollouts.

  python sim/make_mixing_comparison.py --run outputs/sim/mixing/<ts>
"""
import os
import re
import glob
import argparse
import subprocess

DEFAULT_RUN = "/home/yigit/projects/pemp/outputs/sim/mixing/1779529990"


def find_videos(rollout_dir, suffix=""):
    """Return sorted list of (g_tag, pe_path, bare_path) for freqs present in both.
    suffix '' matches the matplotlib renders, '_ggui' matches --render ggui outputs."""
    pat = re.compile(r"mixing_(pe|bare)_g(\d+p\d+)" + re.escape(suffix) + r"\.mp4$")
    pe, bare = {}, {}
    for f in glob.glob(os.path.join(rollout_dir, f"mixing_*_g*{suffix}.mp4")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        (pe if m.group(1) == "pe" else bare)[m.group(2)] = f
    common = sorted(set(pe) & set(bare), key=lambda s: float(s.replace("p", ".")))
    return [(g, pe[g], bare[g]) for g in common]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--variant", choices=["matplotlib", "ggui"], default="matplotlib")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = "_ggui" if args.variant == "ggui" else ""

    rollout_dir = os.path.join(args.run, "rollout")
    rows = find_videos(rollout_dir, suffix)
    if not rows:
        raise SystemExit(f"no paired pe/bare {args.variant} videos found in {rollout_dir}")
    n = len(rows)
    print(f"[cmp] {n} frequencies: {[g for g, _, _ in rows]}")

    inputs = [pe for _, pe, _ in rows] + [bare for _, _, bare in rows]  # PE row then bare row
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    top = "".join(f"[{i}:v]" for i in range(n)) + f"hstack=inputs={n}[top]"
    bot = "".join(f"[{i}:v]" for i in range(n, 2 * n)) + f"hstack=inputs={n}[bot]"
    fc = f"{top};{bot};[top][bot]vstack=inputs=2[v]"
    out = args.out or os.path.join(rollout_dir, f"comparison_pe_vs_bare{suffix}.mp4")
    cmd += ["-filter_complex", fc, "-map", "[v]", out]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[cmp] wrote {out}")


if __name__ == "__main__":
    main()
