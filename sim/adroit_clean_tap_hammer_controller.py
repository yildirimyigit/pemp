"""Clean late-contact tap controller for ``AdroitHandHammer-vPEMPTap``.

The environment wrapper starts at a raised pre-strike pose with the nail set
slightly into the board.  This controller then saves and replays pure action
cycles that contact the nail late in each swing and rebound immediately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import adroit_hand_hammer_tap  # noqa: F401  Registers AdroitHandHammer-vPEMPTap.
from adroit_hand_hammer_tap import (  # noqa: E402
    ENV_ID,
    NAIL_QPOS_INDEX,
    RAISED_ARM_QPOS,
    arm_action_for_qpos,
    full_action,
)


gym.register_envs(gymnasium_robotics)


@dataclass
class RunResult:
    actions: np.ndarray
    phases: np.ndarray
    tool_positions: np.ndarray
    nail_positions: np.ndarray
    contacts: np.ndarray
    nail_qpos: np.ndarray
    strike_hits: np.ndarray
    strike_first_hit_steps: np.ndarray
    strike_first_hit_cycle_steps: np.ndarray
    strike_contact_counts: np.ndarray
    strike_nail_delta: np.ndarray
    goal_distance: float
    action_cycle_max_error: float
    tool_cycle_max_error: float
    relative_cycle_max_error: float


def hide_mujoco_info_pane(env: gym.Env) -> None:
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    if renderer is None:
        return
    viewer = renderer._get_viewer(render_mode="human")
    if hasattr(viewer, "_hide_menu"):
        viewer._hide_menu = True


def find_body_id(base_env, name: str) -> int | None:
    for body_index in range(base_env.model.nbody):
        if base_env.model.body(body_index).name == name:
            return body_index
    return None


def hammer_nail_contact(env: gym.Env) -> bool:
    base_env = env.unwrapped
    model = base_env.model
    data = base_env.data
    object_body_id = find_body_id(base_env, "Object")
    nail_body_id = find_body_id(base_env, "nail")
    object_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == object_body_id
    }
    nail_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == nail_body_id
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if (geom1 in object_geoms and geom2 in nail_geoms) or (
            geom2 in object_geoms and geom1 in nail_geoms
        ):
            return True
    return False


def task_positions(env: gym.Env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_env = env.unwrapped
    tool = base_env.data.site_xpos[base_env.tool_site_id].copy()
    nail = base_env.data.site_xpos[base_env.target_obj_site_id].copy()
    goal = base_env.data.site_xpos[base_env.goal_site_id].copy()
    return tool, nail, goal


def goal_distance(env: gym.Env) -> float:
    _, nail, goal = task_positions(env)
    return float(np.linalg.norm(nail - goal))


def make_env(
    render: bool,
    seed: int,
    max_episode_steps: int,
    nail_start_qpos: float,
    nail_frictionloss: float,
    warmup_steps: int,
    hide_info_pane: bool,
    board_pos: np.ndarray | None = None,
) -> gym.Env:
    kwargs = dict(
        render_mode="human" if render else None,
        max_episode_steps=max_episode_steps,
        nail_start_qpos=nail_start_qpos,
        nail_frictionloss=nail_frictionloss,
        warmup_steps=warmup_steps,
    )
    if board_pos is not None:
        kwargs["board_pos"] = np.asarray(board_pos, dtype=np.float64)
    env = gym.make(ENV_ID, **kwargs)
    env.reset(seed=seed)
    if render and hide_info_pane:
        hide_mujoco_info_pane(env)
    return env


def default_board_pos() -> np.ndarray:
    """Read the MJCF default nail-board position by spawning a one-shot env."""
    env = gym.make(ENV_ID, render_mode=None)
    env.reset()
    base = env.unwrapped
    board_id = find_body_id(base, "nail_board")
    pos = base.model.body_pos[board_id].copy() if board_id is not None else None
    env.close()
    if pos is None:
        raise RuntimeError("nail_board body not found in env model.")
    return pos


def sample_board_pos(rng: np.random.Generator, ref: np.ndarray,
                     x_range: float, y_range: float, z_range: float) -> np.ndarray:
    """Per-demo nail-board position: ref + uniform(-range, range) on each axis."""
    return np.asarray([
        ref[0] + rng.uniform(-x_range, x_range),
        ref[1] + rng.uniform(-y_range, y_range),
        ref[2] + rng.uniform(-z_range, z_range),
    ], dtype=np.float64)


def build_cycle_actions(
    base_env,
    strikes: int,
    ramp_steps: int,
    retract_steps: int,
    settle_steps: int,
    jitter: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raised4 = arm_action_for_qpos(base_env, RAISED_ARM_QPOS)
    if jitter > 0.0:
        raised4[1] += rng.uniform(-jitter, jitter)
        raised4[3] += rng.uniform(-jitter, jitter)
    struck4 = np.array([0.2, raised4[1], -1.0, raised4[3]], dtype=np.float32)

    cycle_actions: list[np.ndarray] = []
    cycle_phases: list[str] = []
    for step in range(ramp_steps):
        alpha = (step + 1) / ramp_steps
        cycle_actions.append(full_action(raised4 + alpha * (struck4 - raised4)))
        cycle_phases.append("swing")
    for step in range(retract_steps):
        alpha = (step + 1) / retract_steps
        cycle_actions.append(full_action(struck4 + alpha * (raised4 - struck4)))
        cycle_phases.append("retract")
    for _ in range(settle_steps):
        cycle_actions.append(full_action(raised4))
        cycle_phases.append("settle")

    actions: list[np.ndarray] = []
    phases: list[str] = []
    for strike in range(1, strikes + 1):
        for action, phase in zip(cycle_actions, cycle_phases, strict=True):
            actions.append(action)
            phases.append(f"strike_{strike}_{phase}")
    return np.asarray(actions, dtype=np.float32), np.asarray(phases, dtype="U32")


def cycle_max_error(values: np.ndarray, strikes: int) -> float:
    if strikes < 2 or len(values) == 0:
        return 0.0
    cycle_steps = len(values) // strikes
    cycles = values.reshape(strikes, cycle_steps, -1)
    return float(np.max(np.abs(cycles - cycles[0])))


def execute(
    env: gym.Env,
    actions: np.ndarray,
    phases: np.ndarray,
    strikes: int,
    render: bool,
    sleep_s: float,
) -> RunResult:
    tool_positions: list[np.ndarray] = []
    nail_positions: list[np.ndarray] = []
    contacts: list[bool] = []
    nail_qpos: list[float] = []
    strike_hits = np.zeros(strikes, dtype=bool)
    strike_first_hit_steps = np.full(strikes, -1, dtype=np.int64)
    strike_first_hit_cycle_steps = np.full(strikes, -1, dtype=np.int64)
    strike_contact_counts = np.zeros(strikes, dtype=np.int64)
    strike_start_nail_qpos = np.zeros(strikes, dtype=np.float32)
    strike_end_nail_qpos = np.zeros(strikes, dtype=np.float32)
    cycle_steps = len(actions) // strikes

    for action_index, (action, phase) in enumerate(zip(actions, phases, strict=True)):
        strike_index = int(phase.split("_")[1]) - 1
        if action_index % cycle_steps == 0:
            strike_start_nail_qpos[strike_index] = float(env.unwrapped.data.qpos[NAIL_QPOS_INDEX])

        env.step(np.clip(action, -1.0, 1.0))
        contact = hammer_nail_contact(env)
        tool, nail, _ = task_positions(env)
        tool_positions.append(tool)
        nail_positions.append(nail)
        contacts.append(contact)
        nail_qpos.append(float(env.unwrapped.data.qpos[NAIL_QPOS_INDEX]))

        if contact:
            strike_contact_counts[strike_index] += 1
            if "_swing" in phase and not strike_hits[strike_index]:
                strike_hits[strike_index] = True
                strike_first_hit_steps[strike_index] = action_index
                strike_first_hit_cycle_steps[strike_index] = action_index % cycle_steps

        if action_index % cycle_steps == cycle_steps - 1:
            strike_end_nail_qpos[strike_index] = float(env.unwrapped.data.qpos[NAIL_QPOS_INDEX])

        if render:
            env.render()
            if sleep_s > 0:
                time.sleep(sleep_s)

    tool_array = np.asarray(tool_positions, dtype=np.float64)
    nail_array = np.asarray(nail_positions, dtype=np.float64)
    relative_array = tool_array - nail_array
    return RunResult(
        actions=actions,
        phases=phases,
        tool_positions=tool_array,
        nail_positions=nail_array,
        contacts=np.asarray(contacts, dtype=bool),
        nail_qpos=np.asarray(nail_qpos, dtype=np.float32),
        strike_hits=strike_hits,
        strike_first_hit_steps=strike_first_hit_steps,
        strike_first_hit_cycle_steps=strike_first_hit_cycle_steps,
        strike_contact_counts=strike_contact_counts,
        strike_nail_delta=strike_end_nail_qpos - strike_start_nail_qpos,
        goal_distance=goal_distance(env),
        action_cycle_max_error=cycle_max_error(actions, strikes),
        tool_cycle_max_error=cycle_max_error(tool_array, strikes),
        relative_cycle_max_error=cycle_max_error(relative_array, strikes),
    )


def record_run(args: argparse.Namespace, strikes: int, seed: int, render: bool,
               board_pos: np.ndarray | None = None) -> RunResult:
    total_steps = (
        strikes * (args.ramp_steps + args.retract_steps + args.settle_steps)
        + args.warmup_steps
        + 20
    )
    env = make_env(
        render=render,
        seed=seed,
        max_episode_steps=total_steps,
        nail_start_qpos=args.nail_start_qpos,
        nail_frictionloss=args.nail_frictionloss,
        warmup_steps=args.warmup_steps,
        hide_info_pane=not args.show_info_pane,
        board_pos=board_pos,
    )
    actions, phases = build_cycle_actions(
        env.unwrapped,
        strikes=strikes,
        ramp_steps=args.ramp_steps,
        retract_steps=args.retract_steps,
        settle_steps=args.settle_steps,
        jitter=args.jitter,
        seed=seed,
    )
    result = execute(env, actions, phases, strikes, render, args.sleep)
    env.close()
    return result


def save_run(path: Path, result: RunResult, args: argparse.Namespace, strikes: int, seed: int,
             board_pos: np.ndarray | None = None,
             board_pos_default: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra: dict = {}
    if board_pos is not None:
        # Save the actual nail-board position AND a g_vector convenience:
        # g_vector = [strikes / MAX_LOOPS, *board_pos] -- frequency conditioning
        # plus the nail position so downstream processing can build a vector g.
        # Also save the offset from the MJCF default for clearer interpretation.
        bp = np.asarray(board_pos, dtype=np.float32)
        ref = (np.asarray(board_pos_default, dtype=np.float32)
               if board_pos_default is not None else np.zeros(3, dtype=np.float32))
        MAX_LOOPS = 6
        g_freq = np.float32(strikes / MAX_LOOPS)
        g_vector = np.concatenate([[g_freq], bp]).astype(np.float32)  # (4,)
        extra.update(
            board_pos=bp,
            board_pos_default=ref,
            board_pos_offset=(bp - ref).astype(np.float32),
            g_freq=g_freq,
            g_vector=g_vector,
        )
    np.savez_compressed(
        path,
        actions=result.actions,
        phases=result.phases,
        tool_positions=result.tool_positions,
        nail_positions=result.nail_positions,
        contacts=result.contacts,
        nail_qpos=result.nail_qpos,
        strike_hits=result.strike_hits,
        strike_first_hit_steps=result.strike_first_hit_steps,
        strike_first_hit_cycle_steps=result.strike_first_hit_cycle_steps,
        strike_contact_counts=result.strike_contact_counts,
        strike_nail_delta=result.strike_nail_delta,
        seed=np.array(seed, dtype=np.int64),
        env_id=np.array(ENV_ID),
        controller=np.array("adroit_clean_tap_hammer_controller"),
        pure_action_trajectory=np.array(True, dtype=bool),
        num_strikes=np.array(strikes, dtype=np.int64),
        nail_start_qpos=np.array(args.nail_start_qpos, dtype=np.float32),
        nail_frictionloss=np.array(args.nail_frictionloss, dtype=np.float32),
        warmup_steps=np.array(args.warmup_steps, dtype=np.int64),
        ramp_steps=np.array(args.ramp_steps, dtype=np.int64),
        retract_steps=np.array(args.retract_steps, dtype=np.int64),
        settle_steps=np.array(args.settle_steps, dtype=np.int64),
        goal_distance=np.array(result.goal_distance, dtype=np.float32),
        action_cycle_max_error=np.array(result.action_cycle_max_error, dtype=np.float32),
        tool_cycle_max_error=np.array(result.tool_cycle_max_error, dtype=np.float32),
        relative_cycle_max_error=np.array(result.relative_cycle_max_error, dtype=np.float32),
        **extra,
    )


def replay(path: Path, render: bool, sleep_s: float, show_info_pane: bool) -> RunResult:
    with np.load(path, allow_pickle=False) as data:
        actions = data["actions"].astype(np.float32)
        phases = data["phases"].astype("U32")
        strikes = int(data["num_strikes"].item())
        seed = int(data["seed"].item()) if "seed" in data else 0
        nail_start_qpos = float(data["nail_start_qpos"].item())
        nail_frictionloss = float(data["nail_frictionloss"].item())
        warmup_steps = int(data["warmup_steps"].item())
        # Newer demos save the per-demo nail-board position; honor it so the
        # replay reproduces the exact same nail location (else MJCF default).
        board_pos = (data["board_pos"].astype(np.float64)
                     if "board_pos" in data.files else None)

    env = make_env(
        render=render,
        seed=seed,
        max_episode_steps=len(actions) + warmup_steps + 20,
        nail_start_qpos=nail_start_qpos,
        nail_frictionloss=nail_frictionloss,
        warmup_steps=warmup_steps,
        hide_info_pane=not show_info_pane,
        board_pos=board_pos,
    )
    result = execute(env, actions, phases, strikes, render, sleep_s)
    env.close()
    return result


def print_summary(prefix: str, result: RunResult) -> None:
    print(
        f"{prefix} steps={len(result.actions)} "
        f"strike_hits={result.strike_hits.tolist()} "
        f"first_hit_cycle_steps={result.strike_first_hit_cycle_steps.tolist()} "
        f"contact_counts={result.strike_contact_counts.tolist()} "
        f"nail_deltas={[round(float(x), 4) for x in result.strike_nail_delta]} "
        f"final_nail_qpos={result.nail_qpos[-1]:.4f} "
        f"goal_distance={result.goal_distance:.4f} "
        f"action_cycle_max_error={result.action_cycle_max_error:.6f} "
        f"tool_cycle_max_error={result.tool_cycle_max_error:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-actions", type=Path)
    parser.add_argument("--save-actions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("sim/data/raw_clean_taps"))
    parser.add_argument("--strikes", type=int, default=6)
    parser.add_argument("--min-strikes", type=int, default=3)
    parser.add_argument("--max-strikes", type=int, default=6)
    parser.add_argument("--demos-per-frequency", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--show-info-pane", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--ramp-steps", type=int, default=16)
    parser.add_argument("--retract-steps", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--nail-start-qpos", type=float, default=0.04)
    parser.add_argument("--nail-frictionloss", type=float, default=15.0)
    parser.add_argument("--nail-x-range", type=float, default=0.0,
                        help="half-range of per-demo nail-board x jitter (m); 0 = no jitter")
    parser.add_argument("--nail-y-range", type=float, default=0.0,
                        help="half-range of per-demo nail-board y jitter (m); 0 = no jitter")
    parser.add_argument("--nail-z-range", type=float, default=0.0,
                        help="half-range of per-demo nail-board z jitter (m); 0 = no jitter")
    parser.add_argument("--max-resample", type=int, default=30,
                        help="when nail-position jitter is enabled, retry a demo with a new "
                             "rng draw if its strikes don't all land cleanly; 0 disables retries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render = not args.no_render
    if render:
        time.sleep(0.2)

    if args.load_actions is not None:
        result = replay(args.load_actions, render, args.sleep, args.show_info_pane)
        print_summary("replayed", result)
        return

    if args.save_actions is not None:
        result = record_run(args, args.strikes, args.seed, render)
        save_run(args.save_actions, result, args, args.strikes, args.seed)
        print_summary(f"saved {args.save_actions}", result)
        return

    jitter_any = any(r > 0.0 for r in
                     (args.nail_x_range, args.nail_y_range, args.nail_z_range))
    ref_pos = default_board_pos() if jitter_any else None
    if jitter_any:
        print(f"nail-board default pos: {ref_pos.tolist()};  per-demo jitter "
              f"x={args.nail_x_range} y={args.nail_y_range} z={args.nail_z_range}")

    def _is_clean(result: RunResult) -> bool:
        """Every commanded strike contacts AND drives the nail at least a touch."""
        return bool(result.strike_hits.all()) and float(result.strike_nail_delta.sum()) > 1e-4

    file_index = 0
    per_demo: list[dict] = []
    for strikes in range(args.min_strikes, args.max_strikes + 1):
        for _ in range(args.demos_per_frequency):
            base = args.base_seed + file_index
            # Retry up to --max-resample times with a fresh rng draw (only the
            # nail-board position re-samples; action sequence is determined by
            # --jitter using the same per-demo seed for action variability).
            attempt = 0
            while True:
                # Shift seed for resampling rounds so the nail position changes.
                seed = base + attempt * 100_000
                board_pos = None
                if jitter_any:
                    rng = np.random.default_rng(seed)
                    board_pos = sample_board_pos(
                        rng, ref_pos, args.nail_x_range,
                        args.nail_y_range, args.nail_z_range,
                    )
                result = record_run(args, strikes, base, render, board_pos=board_pos)
                if not jitter_any or args.max_resample <= 0 or _is_clean(result):
                    break
                attempt += 1
                if attempt > args.max_resample:
                    print(f"  WARNING: file_index={file_index} couldn't find a clean draw "
                          f"after {args.max_resample} retries; keeping the last attempt.")
                    break
            path = args.output_dir / f"{file_index}.npz"
            save_run(path, result, args, strikes, seed,
                     board_pos=board_pos, board_pos_default=ref_pos)
            print_summary(f"saved {path} (attempt={attempt})", result)
            per_demo.append({
                "file": f"{file_index}.npz",
                "seed": int(seed),
                "base_seed": int(base),
                "resample_attempt": int(attempt),
                "strikes": int(strikes),
                "board_pos": board_pos.tolist() if board_pos is not None else None,
            })
            file_index += 1

    # Dataset-level metadata.json -- saved at the PARENT of --output-dir (e.g. next
    # to raw/), so it travels with the processed/ folder built later from it.
    import json
    meta_path = args.output_dir.parent / "metadata.json"
    meta = {
        "controller": "adroit_clean_tap_hammer_controller",
        "env_id": ENV_ID,
        "base_seed": int(args.base_seed),
        "jitter": float(args.jitter),
        "nail_x_range": float(args.nail_x_range),
        "nail_y_range": float(args.nail_y_range),
        "nail_z_range": float(args.nail_z_range),
        "nail_start_qpos": float(args.nail_start_qpos),
        "nail_frictionloss": float(args.nail_frictionloss),
        "board_pos_default": ref_pos.tolist() if ref_pos is not None else None,
        "min_strikes": int(args.min_strikes),
        "max_strikes": int(args.max_strikes),
        "demos_per_frequency": int(args.demos_per_frequency),
        "n_demos": file_index,
        "per_demo": per_demo,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote dataset metadata: {meta_path}")


if __name__ == "__main__":
    main()
