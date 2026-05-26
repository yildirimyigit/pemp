"""Impact-and-rebound Adroit hammer demos without the sticky push phase.

This is a companion to ``adroit_cartesian_periodic_hammer_controller.py``.  The
older script keeps the hammer at the struck pose for many steps, which can look
like the hammer sticks to the nail and pushes it in.  This variant removes that
hold: each cycle swings into the nail and immediately retracts.

The result is less nail insertion per strike, but the visual behavior is closer
to hammering: contact, rebound, reset, repeat.  The saved ``actions`` are still
pure action trajectories; setup/warm-up actions are stored separately as
``warmup_actions`` and are replayed automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim._adroit_cartesian_periodic_hammer_controller import (  # not sure about "sim." since this script is only for archival
    DEFAULT_HANDLE_SLIDING_FRICTION,
    DEFAULT_HEAD_NAIL_SLIDING_FRICTION,
    ENV_ID,
    Trajectory,
    parse_board_pos,
    record_trajectory,
    replay_trajectory,
)


DEFAULT_IMPACT_NAIL_FRICTIONLOSS = 11.5
DEFAULT_RAMP_STEPS = 10
DEFAULT_HOLD_STEPS = 0
DEFAULT_RETRACT_STEPS = 10
DEFAULT_SETTLE_STEPS = 4
DEFAULT_WARMUP_STEPS = 40


def per_strike_stats(trajectory: Trajectory) -> tuple[list[int], list[float]]:
    stats_counts: list[int] = []
    stats_nail_delta: list[float] = []
    phases = trajectory.phases.astype(str)
    strike_ids = sorted(
        {
            int(phase.split("_")[1])
            for phase in phases
            if phase.startswith("strike_")
        }
    )
    for strike_id in strike_ids:
        mask = np.array(
            [phase.startswith(f"strike_{strike_id}_") for phase in phases],
            dtype=bool,
        )
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        stats_counts.append(int(trajectory.contacts[indices].sum()))
        start_index = max(indices[0] - 1, 0)
        stats_nail_delta.append(
            float(trajectory.nail_qpos[indices[-1]] - trajectory.nail_qpos[start_index])
        )
    return stats_counts, stats_nail_delta


def save_impact_trajectory(
    path: Path,
    trajectory: Trajectory,
    seed: int,
    strikes: int,
    nail_frictionloss: float,
    handle_sliding_friction: float,
    head_nail_sliding_friction: float,
    board_pos: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contact_counts, nail_deltas = per_strike_stats(trajectory)
    np.savez_compressed(
        path,
        actions=trajectory.actions,
        phases=trajectory.phases,
        warmup_actions=trajectory.warmup_actions,
        tool_positions=trajectory.tool_positions,
        nail_positions=trajectory.nail_positions,
        contacts=trajectory.contacts,
        nail_qpos=trajectory.nail_qpos,
        strike_hits=trajectory.strike_hits,
        strike_first_hit_steps=trajectory.strike_first_hit_steps,
        strike_contact_counts=np.asarray(contact_counts, dtype=np.int64),
        strike_nail_delta=np.asarray(nail_deltas, dtype=np.float32),
        seed=np.array(seed, dtype=np.int64),
        env_id=np.array(ENV_ID),
        controller=np.array("adroit_cartesian_impact_hammer_controller"),
        pure_action_trajectory=np.array(True, dtype=bool),
        impact_rebound=np.array(True, dtype=bool),
        num_strikes=np.array(strikes, dtype=np.int64),
        nail_frictionloss=np.array(nail_frictionloss, dtype=np.float32),
        handle_sliding_friction=np.array(handle_sliding_friction, dtype=np.float32),
        head_nail_sliding_friction=np.array(head_nail_sliding_friction, dtype=np.float32),
        board_pos=(
            np.asarray(board_pos, dtype=np.float64)
            if board_pos is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        goal_distance=np.array(trajectory.goal_distance, dtype=np.float32),
        action_cycle_max_error=np.array(trajectory.action_cycle_max_error, dtype=np.float32),
        tool_cycle_max_error=np.array(trajectory.tool_cycle_max_error, dtype=np.float32),
        relative_tool_nail_cycle_max_error=np.array(
            trajectory.relative_tool_nail_cycle_max_error,
            dtype=np.float32,
        ),
    )


def print_summary(prefix: str, trajectory: Trajectory) -> None:
    contact_counts, nail_deltas = per_strike_stats(trajectory)
    print(
        f"{prefix} steps={len(trajectory.actions)} "
        f"strike_hits={trajectory.strike_hits.tolist()} "
        f"first_hit_steps={trajectory.strike_first_hit_steps.tolist()} "
        f"contact_counts={contact_counts} "
        f"nail_deltas={[round(delta, 4) for delta in nail_deltas]} "
        f"final_nail_qpos={trajectory.nail_qpos[-1]:.4f} "
        f"goal_distance={trajectory.goal_distance:.4f} "
        f"action_cycle_max_error={trajectory.action_cycle_max_error:.6f} "
        f"tool_cycle_max_error={trajectory.tool_cycle_max_error:.4f}"
    )


def output_path(base_dir: Path, index: int) -> Path:
    return base_dir / f"{index}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-actions", type=Path)
    parser.add_argument("--save-actions", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sim/data/raw_cartesian_impact_periodic"),
    )
    parser.add_argument("--strikes", type=int, default=6)
    parser.add_argument("--min-strikes", type=int, default=3)
    parser.add_argument("--max-strikes", type=int, default=6)
    parser.add_argument("--demos-per-frequency", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--show-info-pane", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--ramp-steps", type=int, default=DEFAULT_RAMP_STEPS)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument("--retract-steps", type=int, default=DEFAULT_RETRACT_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Action-space style jitter. Keep at 0.0 for the most reliable contacts.",
    )
    parser.add_argument(
        "--nail-frictionloss",
        type=float,
        default=DEFAULT_IMPACT_NAIL_FRICTIONLOSS,
        help="Higher values reduce pushing and keep later strikes visible.",
    )
    parser.add_argument(
        "--handle-sliding-friction",
        type=float,
        default=DEFAULT_HANDLE_SLIDING_FRICTION,
    )
    parser.add_argument(
        "--head-nail-sliding-friction",
        type=float,
        default=DEFAULT_HEAD_NAIL_SLIDING_FRICTION,
    )
    parser.add_argument("--board-pos", type=float, nargs=3)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="When replaying, execute only saved actions and skip warmup_actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render = not args.no_render
    board_pos = parse_board_pos(args.board_pos)

    if render:
        time.sleep(2)

    if args.load_actions is not None:
        trajectory = replay_trajectory(
            path=args.load_actions,
            render=render,
            sleep_s=args.sleep,
            hide_info_pane=not args.show_info_pane,
            no_warmup=args.no_warmup,
        )
        print_summary("replayed", trajectory)
        return

    if args.save_actions is not None:
        trajectory = record_trajectory(
            strikes=args.strikes,
            seed=args.seed,
            render=render,
            sleep_s=args.sleep,
            warmup_steps=args.warmup_steps,
            ramp_steps=args.ramp_steps,
            hold_steps=args.hold_steps,
            retract_steps=args.retract_steps,
            settle_steps=args.settle_steps,
            jitter=args.jitter,
            nail_frictionloss=args.nail_frictionloss,
            handle_sliding_friction=args.handle_sliding_friction,
            head_nail_sliding_friction=args.head_nail_sliding_friction,
            board_pos=board_pos,
            hide_info_pane=not args.show_info_pane,
        )
        save_impact_trajectory(
            path=args.save_actions,
            trajectory=trajectory,
            seed=args.seed,
            strikes=args.strikes,
            nail_frictionloss=args.nail_frictionloss,
            handle_sliding_friction=args.handle_sliding_friction,
            head_nail_sliding_friction=args.head_nail_sliding_friction,
            board_pos=board_pos,
        )
        print_summary(f"saved {args.save_actions}", trajectory)
        return

    file_index = 0
    for strikes in range(args.min_strikes, args.max_strikes + 1):
        for _ in range(args.demos_per_frequency):
            seed = args.base_seed + file_index
            trajectory = record_trajectory(
                strikes=strikes,
                seed=seed,
                render=render,
                sleep_s=args.sleep,
                warmup_steps=args.warmup_steps,
                ramp_steps=args.ramp_steps,
                hold_steps=args.hold_steps,
                retract_steps=args.retract_steps,
                settle_steps=args.settle_steps,
                jitter=args.jitter,
                nail_frictionloss=args.nail_frictionloss,
                handle_sliding_friction=args.handle_sliding_friction,
                head_nail_sliding_friction=args.head_nail_sliding_friction,
                board_pos=board_pos,
                hide_info_pane=not args.show_info_pane,
            )
            path = output_path(args.output_dir, file_index)
            save_impact_trajectory(
                path=path,
                trajectory=trajectory,
                seed=seed,
                strikes=strikes,
                nail_frictionloss=args.nail_frictionloss,
                handle_sliding_friction=args.handle_sliding_friction,
                head_nail_sliding_friction=args.head_nail_sliding_friction,
                board_pos=board_pos,
            )
            print_summary(f"saved {path}", trajectory)
            file_index += 1


if __name__ == "__main__":
    main()
