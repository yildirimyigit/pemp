"""Generate a multi-frequency periodic-stir dataset for the FluidLab Mixing task,
for PEMP training.  The stir frequency (number of loops over the horizon) is the
PEMP conditioning g = n_loops / max_loops.

Actions are analytic (a constant-magnitude velocity rotating in the xz-plane;
the y/vertical dim is fixed at 0, matching MixingEnv's fix_dim=[1]), so this needs
no simulator/GPU.  Trajectories have length T=200 and live in the Mixing action
range [-0.007, 0.007].

PHASE.  Each demo gets its own random starting phase phi (we deliberately do NOT
force every demo to start at the same point).  A plain PE-of-time model collapses
on this -- with a random global phase the same (t, g) maps to uncorrelated y, so
the L2/NLL-optimal answer is the mean (a flat line) and bare ~= PE.  The systematic
fix is to index the trajectory by its *angular phase* theta(t)=2*pi*n*t/T + phi
instead of absolute time, so every demo becomes the same canonical function of
theta and the phase offset is absorbed.  We therefore store, per demo:
  theta : (T,1) the unwrapped angular phase (radians) of each step
  phase : ()    the demo's starting phase phi
so the processing step can build PE(theta) (recommended) or condition on phi.

Output (mirrors the Adroit PEMP dataset layout) in sim/data/fluidlab_mixing/:
  raw/<i>.npz    per-demo: actions(T,3) x(T,1) theta(T,1) g() phase() n_loops() env_id
  x.npy (n,T,1)  normalized time      x_test.npy
  y.npy (n,T,3)  stirrer xyz actions  y_test.npy
  g.npy (n,)     normalized frequency g_test.npy
  theta.npy (n,T,1) angular phase     theta_test.npy
  phase.npy (n,)    starting phase    phase_test.npy
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

ENV_ID = "Mixing-v0"
ACTION_DIM = 3
ACT_LIMIT = 0.007          # MixingEnv.action_range
T = 200                    # trajectory length
FREQS = [2, 3, 4, 5, 6]    # stir loops over the horizon
MAX_LOOPS = 6              # g = n_loops / MAX_LOOPS
DEMOS_PER_FREQ = 4         # last one per frequency is held out for test
ALIGN_PHASE = True        # True -> phi=0 for every demo (simple aligned baseline)

OUT = Path(__file__).resolve().parent / "fluidlab_mixing"


def circular_stir(n_loops: int, speed: float, phase: float):
    """Analytic periodic stir.  Returns (actions (T,3), theta (T,)).

    theta(t) = 2*pi*n_loops*t/T + phase is the angular phase of the stir; the
    actions are a pure function of it: [-sin(theta), 0, cos(theta)] * speed.
    """
    t = np.arange(T)
    th = 2 * np.pi * n_loops * t / T + phase
    a = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], axis=1) * speed
    return np.clip(a, -ACT_LIMIT, ACT_LIMIT).astype(np.float32), th.astype(np.float32)


def main() -> None:
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    x_time = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)
    train = ([], [], [], [], [])  # x, y, g, theta, phase
    test = ([], [], [], [], [])
    idx = 0
    for n_loops in FREQS:
        g = np.float32(n_loops / MAX_LOOPS)
        for k in range(DEMOS_PER_FREQ):
            rng = np.random.default_rng(1000 * n_loops + k)
            speed = ACT_LIMIT * rng.uniform(0.85, 1.0)
            phase = np.float32(0.0 if ALIGN_PHASE else rng.uniform(0, 2 * np.pi))
            y, theta = circular_stir(n_loops, speed, phase)
            theta = theta.reshape(-1, 1)
            np.savez(OUT / "raw" / f"{idx}.npz", actions=y, x=x_time, theta=theta,
                     g=g, phase=phase, n_loops=np.int64(n_loops), env_id=ENV_ID)
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

    xtr = np.asarray(train[0]); xte = np.asarray(test[0])
    print(f"wrote {OUT}  (ALIGN_PHASE={ALIGN_PHASE})")
    print(f"  train: x{xtr.shape} y{np.asarray(train[1]).shape} g{np.asarray(train[2]).shape} "
          f"theta{np.asarray(train[3]).shape}")
    print(f"  test:  x{xte.shape} y{np.asarray(test[1]).shape} g{np.asarray(test[2]).shape} "
          f"theta{np.asarray(test[3]).shape}")
    print(f"  g train: {sorted(set(float(v) for v in train[2]))}")
    print(f"  g test:  {sorted(float(v) for v in test[2])}")
    print(f"  phase train (rad): {[round(float(v), 2) for v in train[4]]}")


if __name__ == "__main__":
    main()
