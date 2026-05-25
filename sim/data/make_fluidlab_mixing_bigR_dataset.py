"""Larger-radius variant of make_fluidlab_mixing_dataset.py.

The original stir traced small circles: the action is the stirrer's xz *velocity*
(capped at +-0.007), and an integrated circle has radius ~= speed*T/(2*pi*n_loops),
so with T=200 the loops were only ~0.04-0.11 wide and the milk barely swept.

Here we instead trace a circle of an explicit, larger STIR_RADIUS for *every*
frequency: velocity(t) = R*omega*[-sin th, 0, cos th], omega = 2*pi*n/T.  The peak
speed is R*omega = R*2*pi*n/T, so to keep it within the action range we pick the
horizon T just large enough that the fastest frequency (n=MAX_LOOPS) hits the cap.
Result: all stirs sweep the same big circle (radius R), differing only in how many
loops they complete -> much more visible mixing, and the longer horizon gives more
mixing time too.  (Low frequencies therefore use smaller action magnitudes -- same
radius, slower stir -- which is physically correct.)

ALIGN_PHASE / theta / phase semantics are unchanged (see the original generator).

Output: sim/data/fluidlab_mixing_bigR/  (raw/<i>.npz + x/y/g/theta/phase .npy [+_test])
"""
from __future__ import annotations

import math
import argparse
from pathlib import Path
import numpy as np

ENV_ID = "Mixing-v0"
ACTION_DIM = 3
ACT_LIMIT = 0.007          # MixingEnv.action_range
STIR_RADIUS = 0.15         # target circular-stir path radius (was effectively ~0.04-0.11)
RADIUS_JITTER = (0.85, 1.0)
FREQS = [2, 3, 4, 5, 6]    # stir loops over the horizon
MAX_LOOPS = 6              # g = n_loops / MAX_LOOPS
DEMOS_PER_FREQ = 4         # last one per frequency is held out for test
ALIGN_PHASE = True         # match the working dataset (phi=0 for every demo)

# horizon: smallest multiple of 50 s.t. the fastest stir's peak speed <= ACT_LIMIT
T = int(math.ceil(STIR_RADIUS * 2 * math.pi * MAX_LOOPS / ACT_LIMIT / 50.0) * 50)

BASE = Path(__file__).resolve().parent

# Each --seed produces a DISTINCT same-characteristics draw (different amplitude jitter,
# and phases too if ALIGN_PHASE=False), written to its own folder so you can train one
# run per seed and use them as error-bar seeds.  seed 0 keeps the canonical folder name.
def out_dir(seed):
    return BASE / ("fluidlab_mixing_bigR" if seed == 0 else f"fluidlab_mixing_bigR_s{seed}")


def circular_stir(n_loops: int, radius: float, phase: float):
    """Velocity actions that trace a circle of `radius` over n_loops in T steps.
    Returns (actions (T,3), theta (T,)).  Peak |v| = radius * 2*pi*n_loops / T."""
    t = np.arange(T)
    omega = 2 * np.pi * n_loops / T
    th = omega * t + phase
    a = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], axis=1) * (radius * omega)
    return np.clip(a, -ACT_LIMIT, ACT_LIMIT).astype(np.float32), th.astype(np.float32)


def main(seed: int = 0) -> None:
    OUT = out_dir(seed)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    x_time = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)
    train = ([], [], [], [], [])  # x, y, g, theta, phase
    test = ([], [], [], [], [])
    idx = 0
    print(f"seed={seed} -> {OUT.name}; horizon T={T}  (radius {STIR_RADIUS}, peak speed at "
          f"n={MAX_LOOPS}: {STIR_RADIUS * 2 * np.pi * MAX_LOOPS / T:.5f} <= {ACT_LIMIT})")
    for n_loops in FREQS:
        g = np.float32(n_loops / MAX_LOOPS)
        for k in range(DEMOS_PER_FREQ):
            rng = np.random.default_rng(1_000_000 * seed + 1000 * n_loops + k)
            radius = STIR_RADIUS * rng.uniform(*RADIUS_JITTER)
            phase = np.float32(0.0 if ALIGN_PHASE else rng.uniform(0, 2 * np.pi))
            y, theta = circular_stir(n_loops, radius, phase)
            theta = theta.reshape(-1, 1)
            np.savez(OUT / "raw" / f"{idx}.npz", actions=y, x=x_time, theta=theta,
                     g=g, phase=phase, n_loops=np.int64(n_loops),
                     radius=np.float32(radius), env_id=ENV_ID)
            bucket = test if k == DEMOS_PER_FREQ - 1 else train
            bucket[0].append(x_time); bucket[1].append(y); bucket[2].append(g)
            bucket[3].append(theta); bucket[4].append(phase)
            idx += 1

    for name, (xs, ys, gs, ths, phs) in (("", train), ("_test", test)):
        np.save(OUT / f"x{name}.npy", np.asarray(xs, np.float32))
        np.save(OUT / f"y{name}.npy", np.asarray(ys, np.float32))
        np.save(OUT / f"g{name}.npy", np.asarray(gs, np.float32))
        np.save(OUT / f"theta{name}.npy", np.asarray(ths, np.float32))
        np.save(OUT / f"phase{name}.npy", np.asarray(phs, np.float32))

    print(f"wrote {OUT}  (ALIGN_PHASE={ALIGN_PHASE}, T={T})")
    print(f"  train: y{np.asarray(train[1]).shape}  test: y{np.asarray(test[1]).shape}")
    print(f"  g train: {sorted(set(float(v) for v in train[2]))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", default=[0],
                    help="dataset seed(s); each writes fluidlab_mixing_bigR[_s<seed>]/")
    args = ap.parse_args()
    for s in args.seed:
        main(s)
