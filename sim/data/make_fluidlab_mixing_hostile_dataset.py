"""WMP-hostile Mixing dataset: a repeated HALF-DISK "D" stir centred on the fluid.

Each demo is ONLY the stir (no reach, no pause): n laps of a "D" path -- a 180-degree
arc plus its straight diameter -- traced over the whole horizon T.  The D is sized and
offset so the FLUID CENTRE sits INSIDE the half-disk (the diameter is half a radius
below the centre).  Per demo the whole D is rotated by a random angle about the fluid
centre (the within-g latent): the centre stays inside for every orientation, and the
model must infer the orientation from the context points.

Why this breaks WMP but not PEMP:
  * the arc<->diameter junctions are sharp velocity-direction corners -> a smooth db4
    wavelet basis rings (the Gibbs floor we measured on square/sawtooth); PEMP's PE+MLP
    reproduces the corners.
  * random per-demo orientation -> demos at the same g differ, so phase/orientation must
    be inferred from context (PEMP's encoder strength; WMP's periodization + phase-
    adaptive blending degrade).  This restores the within-g variability that a single
    shared circular stir lacked (which made all methods tie).

g = n / MAX_LOOPS indexes the stir frequency (n in {3,4,5,6}).  Same horizon T=850,
Mixing-v0 actuator (3-D xz velocity, +-0.007), per-seed folder layout as bigR.  Each
raw npz also stores `start_offset` = (dx,dz) to place the stirrer so the rotated D is
centred on the cup centre at rollout time.  phase.npy is a zeros placeholder; theta.npy
carries the per-step stir angle.

Output: sim/data/fluidlab_mixing_hostile[_s<seed>]/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ENV_ID = "Mixing-v0"
ACTION_DIM = 3
ACT_LIMIT = 0.007          # MixingEnv.action_range

T = 850                    # horizon (the D stir fills it; no reach/pause)
STIR_R = 0.18              # arc radius; diameter sits STIR_R/2 below the centre so the
                           # fluid centre is inside the half-disk.  Peak step speed is
                           # ~ perimeter/lap = R*(2+pi)*n/T <= ACT_LIMIT for n<=6 (checked).
CENTER_H = STIR_R / 2.0    # diameter offset below the fluid centre

FREQS = [3, 4, 5, 6]       # n = number of D laps
MAX_LOOPS = 6              # g = n / MAX_LOOPS -> {0.5, 0.667, 0.833, 1.0}
DEMOS_PER_FREQ = 12        # last one per freq held out for test (4 test total)

# peak step speed at the fastest frequency must stay under the actuator cap
_PEAK = STIR_R * (2 + np.pi) * max(FREQS) / T
assert _PEAK <= ACT_LIMIT, f"peak step speed {_PEAK:.5f} > ACT_LIMIT {ACT_LIMIT}"

BASE = Path(__file__).resolve().parent


def out_dir(seed: int) -> Path:
    return BASE / ("fluidlab_mixing_hostile" if seed == 0 else
                   f"fluidlab_mixing_hostile_s{seed}")


def _d_positions(n_loops: int) -> np.ndarray:
    """Un-rotated D path (T,2) in xz, origin = fluid centre: n laps of
    [straight diameter A->B] + [semicircle arc B->A, bulging +z].  A=(-R,-h), B=(R,-h)."""
    h = CENTER_H
    A = np.array([-STIR_R, -h]); B = np.array([STIR_R, -h])
    base, rem = divmod(T, n_loops)
    laps = [base + 1 if i < rem else base for i in range(n_loops)]
    segs = []
    for L in laps:
        n_diam = max(2, round(L * 2.0 / (2.0 + np.pi)))   # split by arc-length 2R : piR
        n_arc = L - n_diam
        diam = np.linspace(A, B, n_diam, endpoint=False)
        phi = np.linspace(0.0, np.pi, n_arc, endpoint=False)
        arc = np.stack([STIR_R * np.cos(phi), -h + STIR_R * np.sin(phi)], axis=1)
        segs.append(diam); segs.append(arc)
    P = np.concatenate(segs, axis=0)                      # (T, 2), closed laps
    assert P.shape[0] == T, (P.shape, T)
    return P


def make_demo(n_loops: int, rng: np.random.Generator):
    """n D-laps, the whole D rotated by a random angle about the fluid centre."""
    P = _d_positions(n_loops)                              # (T,2) un-rotated, origin=centre
    # per-step velocity = displacement to the next point (last step closes the loop)
    vel = np.empty_like(P)
    vel[:-1] = P[1:] - P[:-1]
    vel[-1] = P[0] - P[-1]

    th = float(rng.uniform(0, 2 * np.pi))                  # random orientation (within-g latent)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    vel_rot = vel @ R.T                                    # rotate every velocity vector

    actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = vel_rot[:, 0]
    actions[:, 2] = vel_rot[:, 1]
    actions = np.clip(actions, -ACT_LIMIT, ACT_LIMIT).astype(np.float32)

    # per-step stir angle along the (rotated) path, for reference / theta.npy
    theta = np.arctan2(P[:, 1] + CENTER_H, P[:, 0]).astype(np.float32).reshape(-1, 1)

    # start offset (dx,dz) that places the stirrer so the rotated D is centred on the cup:
    #   integrated path = start + R(th)*(P - A); fluid centre maps to start + R(th)*(-A),
    #   so start = cup_centre - R(th)*(-A) = cup_centre + R(th)*A.
    A = np.array([-STIR_R, -CENTER_H])
    off = R @ A
    active = np.ones(T, dtype=np.int8)                     # all moving; no dead time

    meta = dict(
        stir_r=np.float32(STIR_R),
        center_h=np.float32(CENTER_H),
        theta_rot=np.float32(th),
        start_offset=np.array([off[0], off[1]], dtype=np.float32),  # (dx, dz) cup coords
        n_loops_done=np.int64(n_loops),
        used_steps=np.int64(T),
        active_mask=active,
    )
    return actions, theta, meta


def main(seed: int = 0) -> None:
    OUT = out_dir(seed)
    raw_dir = OUT / "raw"
    if raw_dir.exists():                                   # clear stale demos (avoid orphans)
        for f in raw_dir.glob("*.npz"):
            f.unlink()
    raw_dir.mkdir(parents=True, exist_ok=True)
    x_time = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)
    train = ([], [], [], [], [])  # x, y, g, theta, phase(placeholder)
    test = ([], [], [], [], [])
    idx = 0
    print(f"seed={seed} -> {OUT.name}; T={T}  D stir (arc+diameter), R={STIR_R}, "
          f"centre inside; random orientation per demo; no reach/pause")
    print(f"  g = n/{MAX_LOOPS} for n in {FREQS}; peak step speed {_PEAK:.5f} <= {ACT_LIMIT}")
    peak_max = 0.0
    for n_loops in FREQS:
        g = np.float32(n_loops / MAX_LOOPS)
        for k in range(DEMOS_PER_FREQ):
            rng = np.random.default_rng(1_000_000 * seed + 1000 * n_loops + k)
            y, theta, meta = make_demo(n_loops, rng)
            peak_max = max(peak_max, float(np.abs(y).max()))
            phase = np.float32(0.0)               # placeholder for schema parity with bigR
            np.savez(raw_dir / f"{idx}.npz",
                     actions=y, x=x_time, theta=theta,
                     g=g, phase=phase, n_loops=np.int64(n_loops),
                     env_id=ENV_ID, **meta)
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
    print(f"  max|action|={peak_max:.5f} <= {ACT_LIMIT}  (D fills T, no dead time)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", default=[0],
                    help="dataset seed(s); each writes fluidlab_mixing_hostile[_s<seed>]/")
    args = ap.parse_args()
    for s in args.seed:
        main(s)
