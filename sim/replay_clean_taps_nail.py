"""Replay (visualize) a saved clean-taps-nail demonstration by seed.

For a given seed it reads the dataset's root metadata.json -- the per-demo record
(board_pos, strikes, npz file) plus the dataset-level env params (nail_start_qpos,
nail_frictionloss, env_id) -- rebuilds the Adroit hammer env with that board position,
and steps the corresponding raw/<file>.npz action sequence with rendering.

board_pos and the nail params come from metadata.json (as requested); warmup_steps is
not stored in metadata, so it is taken from the demo's npz (authoritative) or the
controller default.

Run (needs a display for the live MuJoCo viewer; mujoco is in the pemp-gpu env):
  ~/sw/anaconda3/envs/pemp-gpu/bin/python sim/replay_clean_taps_nail.py --seed 1000
  # options:
  #   --dataset sim/data/clean_taps_nail_s1   (default)
  #   --sleep 0.03   --no-render   --show-info-pane
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# make_env(render, seed, max_episode_steps, nail_start_qpos, nail_frictionloss,
#          warmup_steps, hide_info_pane, board_pos=None) -- reuse the controller's setup
import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from adroit_clean_tap_hammer_controller import make_env


def find_record(meta: dict, seed: int) -> dict:
    recs = [r for r in meta.get("per_demo", []) if int(r["seed"]) == int(seed)]
    if not recs:
        avail = sorted(int(r["seed"]) for r in meta.get("per_demo", []))
        raise SystemExit(f"seed {seed} not in metadata; available seeds: {avail}")
    if len(recs) > 1:
        print(f"[warn] {len(recs)} records with seed {seed}; using the first")
    return recs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sim/data/clean_taps_nail_s1",
                    help="dataset root holding metadata.json and raw/")
    ap.add_argument("--seed", type=int, required=True, help="demo seed to replay (e.g. 1000)")
    ap.add_argument("--sleep", type=float, default=0.03, help="per-step delay for the viewer")
    ap.add_argument("--no-render", action="store_true", help="step without opening the viewer")
    ap.add_argument("--show-info-pane", action="store_true", help="keep the MuJoCo info overlay")
    args = ap.parse_args()

    ds = Path(args.dataset)
    meta = json.loads((ds / "metadata.json").read_text())
    rec = find_record(meta, args.seed)

    # --- env setup data from metadata (board_pos + nail params), as requested ---
    board_pos = np.asarray(rec["board_pos"], dtype=np.float64)
    strikes = int(rec.get("strikes", -1))
    nail_start_qpos = float(meta["nail_start_qpos"])
    nail_frictionloss = float(meta["nail_frictionloss"])

    # --- actions from the demo's npz (in raw/) ---
    npz_path = ds / "raw" / rec["file"]
    z = np.load(npz_path, allow_pickle=True)
    actions = z["actions"].astype(np.float32)            # (T, 26)
    # warmup_steps isn't in metadata -> take it from the npz (or controller default 40)
    warmup_steps = int(z["warmup_steps"]) if "warmup_steps" in z.files else 40
    T = actions.shape[0]

    render = not args.no_render
    print(f"[replay] {ds.name}  seed={args.seed}  file={rec['file']}  strikes={strikes}")
    print(f"  board_pos={board_pos.round(5).tolist()}  (default "
          f"{np.asarray(meta['board_pos_default']).round(5).tolist()})")
    print(f"  nail_start_qpos={nail_start_qpos}  nail_frictionloss={nail_frictionloss}  "
          f"warmup_steps={warmup_steps}  T={T}")
    if "strike_nail_delta" in z.files:
        print(f"  strike_nail_delta (from npz) = {np.asarray(z['strike_nail_delta']).round(4).tolist()}")

    env = make_env(
        render=render,
        seed=args.seed,
        max_episode_steps=warmup_steps + T + 50,
        nail_start_qpos=nail_start_qpos,
        nail_frictionloss=nail_frictionloss,
        warmup_steps=warmup_steps,
        hide_info_pane=not args.show_info_pane,
        board_pos=board_pos,
    )

    try:
        for t in range(T):
            env.step(np.clip(actions[t], -1.0, 1.0))
            if render:
                env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)
        print(f"[replay] done ({T} steps).")
    finally:
        env.close()


if __name__ == "__main__":
    main()
