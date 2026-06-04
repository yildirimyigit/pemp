"""Mixing demonstration dataset: each demo is the SAME motion in the same order --

    1. a linear reach (constant velocity, fixed distance, random xz direction),
    2. a short pause (zero action),
    3. n full circles of radius STIR_RADIUS, all in the SAME direction, filling the
       rest of the horizon (so g = n / MAX_LOOPS indexes the stir frequency).

n in {3, 4, 5, 6}.  Same horizon T=850 and Mixing-v0 actuator (3-D xz velocity,
range +-0.007), same per-seed folder layout (raw/<i>.npz + x/y/g/theta/phase .npy
[+_test]) as make_fluidlab_mixing_bigR_dataset.py, so existing notebooks read it
unmodified.  There are no reversals and no inter-loop pauses; within-g variability
comes from the random reach direction and the small reach-duration / pause / radius
jitter (the stir velocity pattern itself is shared across demos at a given g).

phase.npy is a zeros placeholder for schema parity; theta.npy carries the per-step
stir angle (0 during reach + pause).

Output: sim/data/fluidlab_mixing_hostile[_s<seed>]/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ENV_ID = "Mixing-v0"
ACTION_DIM = 3
ACT_LIMIT = 0.007          # MixingEnv.action_range

T = 850                    # horizon
STIR_RADIUS = 0.10         # circular stir radius
RADIUS_JITTER = (0.85, 1.0)
REACH_MIN, REACH_MAX = 100, 160   # reach duration (steps), jittered per demo
REACH_DIST = 0.20          # straight-line reach distance -> speed = REACH_DIST / reach_steps
PAUSE_MIN, PAUSE_MAX = 10, 30     # short pause after the reach (steps)

FREQS = [3, 4, 5, 6]       # n = number of full circles
MAX_LOOPS = 6              # g = n / MAX_LOOPS  -> {0.5, 0.667, 0.833, 1.0}
DEMOS_PER_FREQ = 12        # last one per freq held out for test (4 test total)

# Budget: the shortest loop (fastest freq, longest reach+pause) must stay >= the length
# at which peak speed R*2pi/L reaches ACT_LIMIT, so the stir is never clipped.
_MIN_LOOP = (T - REACH_MAX - PAUSE_MAX) // MAX_LOOPS
_NEED_LOOP = int(np.ceil(STIR_RADIUS * 2 * np.pi / ACT_LIMIT))
assert _MIN_LOOP >= _NEED_LOOP, (
    f"fastest loop {_MIN_LOOP} steps < {_NEED_LOOP} needed to keep peak <= ACT_LIMIT")

BASE = Path(__file__).resolve().parent


def out_dir(seed: int) -> Path:
    return BASE / ("fluidlab_mixing_hostile" if seed == 0 else
                   f"fluidlab_mixing_hostile_s{seed}")


def make_demo(n_loops: int, radius: float, rng: np.random.Generator):
    """reach -> short pause -> n same-direction full circles, filling T exactly."""
    reach_steps = int(rng.integers(REACH_MIN, REACH_MAX + 1))
    pause_steps = int(rng.integers(PAUSE_MIN, PAUSE_MAX + 1))
    remaining = T - reach_steps - pause_steps                 # circles fill this
    base, rem = divmod(remaining, n_loops)
    loop_lengths = np.array([base + 1 if i < rem else base for i in range(n_loops)],
                            dtype=np.int64)                    # sum == remaining

    actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
    theta = np.zeros((T, 1), dtype=np.float32)                 # stir angle (0 in reach + pause)
    active = np.zeros(T, dtype=np.int8)                        # 1 while moving, 0 during the pause

    # 1. reach: constant velocity, fixed distance, random direction
    ang = float(rng.uniform(0, 2 * np.pi))
    v_reach = REACH_DIST / reach_steps
    actions[:reach_steps, 0] = np.cos(ang) * v_reach
    actions[:reach_steps, 2] = np.sin(ang) * v_reach
    active[:reach_steps] = 1

    # 2. short pause: zero action (already), theta stays 0

    # 3. n full circles, SAME direction (+1)
    t = reach_steps + pause_steps
    cur_phase = 0.0
    for i in range(n_loops):
        L = int(loop_lengths[i])
        omega = 2 * np.pi / L
        peak = radius * omega                                  # <= ACT_LIMIT by the budget assert
        s = np.arange(L)
        th = cur_phase + omega * s
        actions[t:t + L, 0] = -np.sin(th) * peak
        actions[t:t + L, 2] =  np.cos(th) * peak
        theta[t:t + L, 0] = th
        active[t:t + L] = 1
        cur_phase = cur_phase + omega * L                      # advances by +2*pi per loop
        t += L

    theta[t:, 0] = cur_phase                                   # t should equal T
    actions = np.clip(actions, -ACT_LIMIT, ACT_LIMIT).astype(np.float32)

    meta = dict(
        reach_steps=np.int64(reach_steps),
        reach_angle=np.float32(ang),
        reach_dist=np.float32(REACH_DIST),
        reach_speed=np.float32(v_reach),
        pause_steps=np.int64(pause_steps),
        loop_lengths=loop_lengths.astype(np.int64),
        n_loops_done=np.int64(n_loops),
        direction=np.int64(1),
        used_steps=np.int64(t),
        active_mask=active,
    )
    return actions, theta, meta


def main(seed: int = 0) -> None:
    OUT = out_dir(seed)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    x_time = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)
    train = ([], [], [], [], [])  # x, y, g, theta, phase(placeholder)
    test = ([], [], [], [], [])
    idx = 0
    print(f"seed={seed} -> {OUT.name}; T={T}  reach in [{REACH_MIN},{REACH_MAX}] (dist {REACH_DIST}) "
          f"+ pause in [{PAUSE_MIN},{PAUSE_MAX}] + n circles (same direction)")
    print(f"  g = n/{MAX_LOOPS} for n in {FREQS}; min loop {_MIN_LOOP} >= {_NEED_LOOP} needed")
    peak_max = 0.0
    for n_loops in FREQS:
        g = np.float32(n_loops / MAX_LOOPS)
        for k in range(DEMOS_PER_FREQ):
            rng = np.random.default_rng(1_000_000 * seed + 1000 * n_loops + k)
            radius = STIR_RADIUS * rng.uniform(*RADIUS_JITTER)
            y, theta, meta = make_demo(n_loops, radius, rng)
            peak_max = max(peak_max, float(np.abs(y).max()))
            phase = np.float32(0.0)               # placeholder for schema parity with bigR
            np.savez(OUT / "raw" / f"{idx}.npz",
                     actions=y, x=x_time, theta=theta,
                     g=g, phase=phase, n_loops=np.int64(n_loops),
                     radius=np.float32(radius), env_id=ENV_ID, **meta)
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

    print(f"wrote {OUT}")
    print(f"  train: y{np.asarray(train[1]).shape}  test: y{np.asarray(test[1]).shape}")
    print(f"  g train: {sorted(set(float(v) for v in train[2]))}")
    print(f"  max|action| across demos: {peak_max:.5f} <= {ACT_LIMIT}  (reach+pause+circles fill T, no dead time)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", default=[0],
                    help="dataset seed(s); each writes fluidlab_mixing_hostile[_s<seed>]/")
    args = ap.parse_args()
    for s in args.seed:
        main(s)
