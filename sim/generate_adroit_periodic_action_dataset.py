"""Generate periodic open-loop Adroit action trajectories for CNMP experiments.

The trajectories here are not controller rollouts.  They are built by taking a
single recorded Adroit hammering primitive and repeating it open loop.  This is
the right dataset when the experiment is about modeling periodicity in action
space rather than solving the contact-rich manipulation problem itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adroit_pemp_controller import RECORDED_ACTIONS, stretch_trajectory


SWING_STEPS = 16
RETRACT_STEPS = 18
SETTLE_STEPS = 10
CYCLE_STEPS = SWING_STEPS + RETRACT_STEPS + SETTLE_STEPS


def build_base_cycle() -> tuple[np.ndarray, list[str]]:
    """Build one fixed open-loop strike/retract/settle action cycle."""
    swing_actions = stretch_trajectory(
        RECORDED_ACTIONS[60:70].copy(),
        SWING_STEPS,
    )
    retract_actions = stretch_trajectory(
        RECORDED_ACTIONS[50:60].copy(),
        RETRACT_STEPS,
    )
    settle_actions = np.tile(retract_actions[-1], (SETTLE_STEPS, 1))
    actions = np.vstack([swing_actions, retract_actions, settle_actions]).astype(
        np.float32
    )
    phases = (
        ["swing"] * SWING_STEPS
        + ["retract"] * RETRACT_STEPS
        + ["settle"] * SETTLE_STEPS
    )
    return actions, phases


def vary_cycle(
    base_cycle: np.ndarray,
    rng: np.random.Generator,
    variation: float,
) -> tuple[np.ndarray, float]:
    """Apply one smooth, cycle-preserving style variation."""
    if variation <= 0:
        return base_cycle.copy(), 1.0

    amplitude_scale = float(rng.uniform(1.0 - variation, 1.0 + variation))
    cycle_mean = base_cycle.mean(axis=0, keepdims=True)
    varied_cycle = cycle_mean + amplitude_scale * (base_cycle - cycle_mean)
    return np.clip(varied_cycle, -1.0, 1.0).astype(np.float32), amplitude_scale


def build_trajectory(
    strike_count: int,
    rng: np.random.Generator,
    variation: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    base_cycle, base_phases = build_base_cycle()
    cycle, amplitude_scale = vary_cycle(base_cycle, rng, variation)
    actions = np.tile(cycle, (strike_count, 1)).astype(np.float32)
    phases = np.array(base_phases * strike_count, dtype="U16")
    return actions, phases, amplitude_scale


def save_trajectory(
    path: Path,
    actions: np.ndarray,
    phases: np.ndarray,
    strike_count: int,
    seed: int,
    amplitude_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        actions=actions,
        phases=phases,
        seed=np.array(seed, dtype=np.int64),
        controller=np.array("recorded_adroit_periodic_open_loop"),
        env_id=np.array("AdroitHandHammer-vPEMP"),
        pure_action_trajectory=np.array(True, dtype=bool),
        periodic_open_loop=np.array(True, dtype=bool),
        num_strikes=np.array(strike_count, dtype=np.int64),
        cycle_steps=np.array(CYCLE_STEPS, dtype=np.int64),
        swing_steps=np.array(SWING_STEPS, dtype=np.int64),
        retract_steps=np.array(RETRACT_STEPS, dtype=np.int64),
        settle_steps=np.array(SETTLE_STEPS, dtype=np.int64),
        amplitude_scale=np.array(amplitude_scale, dtype=np.float32),
    )


def generate_dataset(
    output_dir: Path,
    min_strikes: int,
    max_strikes: int,
    demos_per_frequency: int,
    base_seed: int,
    variation: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_index = 0
    for strike_count in range(min_strikes, max_strikes + 1):
        for demo_index in range(demos_per_frequency):
            seed = base_seed + file_index
            rng = np.random.default_rng(seed)
            actions, phases, amplitude_scale = build_trajectory(
                strike_count=strike_count,
                rng=rng,
                variation=variation,
            )
            path = output_dir / f"{file_index}.npz"
            save_trajectory(
                path=path,
                actions=actions,
                phases=phases,
                strike_count=strike_count,
                seed=seed,
                amplitude_scale=amplitude_scale,
            )
            print(
                f"saved {path} shape={actions.shape} "
                f"strikes={strike_count} scale={amplitude_scale:.4f}"
            )
            file_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sim/data/raw"))
    parser.add_argument("--min-strikes", type=int, default=3)
    parser.add_argument("--max-strikes", type=int, default=6)
    parser.add_argument("--demos-per-frequency", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--variation",
        type=float,
        default=0.03,
        help="Cycle-wide amplitude variation; the cycle remains exactly periodic.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.min_strikes < 1:
        raise ValueError("--min-strikes must be >= 1")
    if args.max_strikes < args.min_strikes:
        raise ValueError("--max-strikes must be >= --min-strikes")
    if args.demos_per_frequency < 1:
        raise ValueError("--demos-per-frequency must be >= 1")
    if args.variation < 0:
        raise ValueError("--variation must be >= 0")

    generate_dataset(
        output_dir=args.output_dir,
        min_strikes=args.min_strikes,
        max_strikes=args.max_strikes,
        demos_per_frequency=args.demos_per_frequency,
        base_seed=args.base_seed,
        variation=args.variation,
    )
