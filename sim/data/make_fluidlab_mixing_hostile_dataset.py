"""WMP-hostile variant of the Mixing dataset: same horizon T=850, same Mixing-v0
actuator (3-D xz velocity), same conditioning semantics (g = n_loops / MAX_LOOPS),
same per-seed folder layout (raw/<i>.npz + x/y/g/theta/phase .npy[+_test]) as
make_fluidlab_mixing_bigR_dataset.py -- so existing notebooks (bare_pe_on_*,
compare_mixing, processed/) read it without modification.

Three deliberate violations of perfect periodicity, all *uncorrelated with g*:

  1. Initial offset (k0 in U[0, OFFSET_MAX]): a zero-action prefix before the
     first stir loop.  Demos no longer share their t=0 phase.  Wavelet's
     phase-locked basis and RBF-demo-blending conditioning lose alignment.

  2. Inter-loop pauses (p_i in U[PAUSE_MIN, PAUSE_MAX]): zero-action segments
     between successive loops.  The cadence is non-stationary -- a stationary
     wavelet basis amortises poorly across demos with different pause patterns.

  3. Per-loop direction reversals (Bernoulli(P_REVERSE)): each loop independently
     flips rotation sign.  This introduces velocity-sign discontinuities at
     non-periodic locations.  Wavelets handle a single local discontinuity, but
     the periodization mode adds boundary error when start- and end-state of the
     trajectory disagree.

The hostile knobs are NOT exposed via g -- they are latent nuisance variables that
both methods must infer from context points at test time.  This is the regime that
exposes WMP's wavelet+GMM(weights|g_scalar) factorization weakness vs PEMP's
per-timestep (t, g)-conditioned MLP decoder.

Design choices to keep schema parity with bigR while creating a clean knob:
 * Loop period L is FIXED across all demos (not scaled by n_loops as in bigR).
   This decouples the hostile structure from g, so the gap between PEMP and WMP
   cannot be explained by frequency-scaling effects.
 * STIR_RADIUS=0.10 (smaller than bigR's 0.15) so peak speed R*2pi/L stays under
   ACT_LIMIT.  Mixing still happens; the absolute MSE numbers won't be directly
   comparable to bigR but the relative ranking (PEMP vs WMP) is the metric.
 * phase.npy is written as zeros (placeholder) to keep schema parity; theta.npy
   carries the per-timestep angular phase including reversals/pauses.

Output: sim/data/fluidlab_mixing_hostile[_s<seed>]/
        same .npy layout as bigR; raw/<i>.npz adds offset/pauses/reversals fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ENV_ID = "Mixing-v0"
ACTION_DIM = 3
ACT_LIMIT = 0.007          # MixingEnv.action_range

T = 850                    # horizon -- matches bigR for like-for-like comparison
STIR_RADIUS = 0.10         # circular stir radius (peak speed R*2pi/L <= ACT_LIMIT)
RADIUS_JITTER = (0.85, 1.0)
LOOP_STEPS = 90            # steps for one full 2*pi loop; peak |v| = 0.10*2pi/90 = 0.00698
OFFSET_MAX = 60            # initial zero-action prefix k0 in U[0, OFFSET_MAX]
PAUSE_MIN, PAUSE_MAX = 6, 30   # inter-loop pause length
P_REVERSE = 0.4            # per-loop probability of rotation-direction flip

FREQS = [2, 3, 4, 5, 6]    # n_loops requested per demo
MAX_LOOPS = 6
DEMOS_PER_FREQ = 12        # last one per freq held out for test (5 test total, matches bigR);
                           # train count is DEMOS_PER_FREQ-1 = 11 per freq -- enough examples
                           # of the latent offset/pause/reversal variability for the encoder
                           # to disambiguate context points at test time.  Earlier we used 4
                           # (3 train), which under-sampled the within-g latent space.

# Worst-case budget check (offset=max, all pauses=max, no early break):
#   60 + 6*90 + 5*30 + tail = 60 + 540 + 150 + tail = 750 + tail; T=850 leaves >=100 tail. OK.
_WORST = OFFSET_MAX + MAX_LOOPS * LOOP_STEPS + (MAX_LOOPS - 1) * PAUSE_MAX
assert _WORST < T, f"hostile budget overflows horizon: worst={_WORST} >= T={T}"

BASE = Path(__file__).resolve().parent


def out_dir(seed: int) -> Path:
    return BASE / ("fluidlab_mixing_hostile" if seed == 0 else
                   f"fluidlab_mixing_hostile_s{seed}")


def make_demo(n_loops: int, radius: float, rng: np.random.Generator):
    """Build one hostile demo: offset prefix, n_loops circular stirs separated by
    variable pauses, with per-loop random direction reversals.  Pads to length T."""
    omega = 2 * np.pi / LOOP_STEPS              # rad per step (constant)
    peak = radius * omega
    assert peak <= ACT_LIMIT, f"peak {peak} > ACT_LIMIT {ACT_LIMIT}"

    actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
    theta = np.zeros((T, 1), dtype=np.float32)   # per-step cumulative angle (incl reversals)
    active = np.zeros(T, dtype=np.int8)           # 1 during a stir loop, 0 during pause/offset/tail

    k0 = int(rng.integers(0, OFFSET_MAX + 1))     # initial zero-action prefix
    cur_phase = 0.0
    t = k0
    reversals = np.zeros(n_loops, dtype=np.int8)
    pauses = np.full(max(n_loops - 1, 0), -1, dtype=np.int64)
    loops_done = 0
    for i in range(n_loops):
        if t + LOOP_STEPS > T:
            break                                  # ran out of horizon (shouldn't with budget)
        sign = -1 if rng.random() < P_REVERSE else 1
        reversals[i] = sign
        # one full loop
        s = np.arange(LOOP_STEPS)
        th = cur_phase + sign * omega * s
        actions[t:t + LOOP_STEPS, 0] = -np.sin(th) * peak
        actions[t:t + LOOP_STEPS, 2] =  np.cos(th) * peak
        theta[t:t + LOOP_STEPS, 0] = th
        active[t:t + LOOP_STEPS] = 1
        cur_phase = cur_phase + sign * omega * LOOP_STEPS
        t += LOOP_STEPS
        loops_done += 1
        # pause (only between successive loops, not after the last)
        if i < n_loops - 1:
            p = int(rng.integers(PAUSE_MIN, PAUSE_MAX + 1))
            if t + p > T:                          # don't overflow into tail; just stop here
                break
            theta[t:t + p, 0] = cur_phase           # hold phase constant during pause
            pauses[i] = p
            t += p

    # Trailing pad: actions/active already zero; hold theta at cur_phase for plot continuity.
    theta[t:, 0] = cur_phase

    # Clamp for safety (peak is already <= ACT_LIMIT but float arithmetic can graze it).
    actions = np.clip(actions, -ACT_LIMIT, ACT_LIMIT).astype(np.float32)

    meta = dict(
        offset_steps=np.int64(k0),
        reversals=reversals[:loops_done].astype(np.int8),
        pauses=pauses[:max(loops_done - 1, 0)].astype(np.int64),
        n_loops_done=np.int64(loops_done),
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
    print(f"seed={seed} -> {OUT.name}; T={T}  loop_steps={LOOP_STEPS}  R={STIR_RADIUS} "
          f"peak={STIR_RADIUS * 2 * np.pi / LOOP_STEPS:.5f} <= {ACT_LIMIT}")
    print(f"  hostile knobs: offset<={OFFSET_MAX}  pause in [{PAUSE_MIN},{PAUSE_MAX}]  "
          f"P_reverse={P_REVERSE}")
    n_dropped_loops = 0
    for n_loops in FREQS:
        g = np.float32(n_loops / MAX_LOOPS)
        for k in range(DEMOS_PER_FREQ):
            rng = np.random.default_rng(1_000_000 * seed + 1000 * n_loops + k)
            radius = STIR_RADIUS * rng.uniform(*RADIUS_JITTER)
            y, theta, meta = make_demo(n_loops, radius, rng)
            n_dropped_loops += (n_loops - int(meta["n_loops_done"]))
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
    if n_dropped_loops:
        print(f"  WARN: budget truncated {n_dropped_loops} loop(s) across demos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", default=[0],
                    help="dataset seed(s); each writes fluidlab_mixing_hostile[_s<seed>]/")
    args = ap.parse_args()
    for s in args.seed:
        main(s)
