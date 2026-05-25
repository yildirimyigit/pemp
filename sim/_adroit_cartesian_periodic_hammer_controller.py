"""Cartesian-periodic pure-action hammering demos for AdroitHandHammer-vPEMP.

This script creates action trajectories for studying rhythmic imitation models
such as CNMP and ProMP.  It deliberately targets a narrower goal than solving
the full Adroit hammer task: produce repeated hammer-head swings that contact
the nail head in a visually periodic way.

No MuJoCo state is edited during the saved rhythmic trajectory.  The script can
modify model parameters once after reset, before warm-up and recording, to make
multi-strike contact observable:

* nail slide frictionloss is raised so the nail does not disappear after one hit
* hammer handle friction is raised to keep the grasp stable
* hammer head / nail sliding friction stays low so retracts do not pull the nail

The saved ``actions`` array contains only the rhythmic swing/retract/settle
cycles.  ``warmup_actions`` are saved separately and are executed automatically
when replaying a trajectory from a fresh environment.
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

import adroit_hand_hammer_updated  # noqa: F401  Registers AdroitHandHammer-vPEMP.


gym.register_envs(gymnasium_robotics)

ENV_ID = "AdroitHandHammer-vPEMP"
NAIL_QPOS_INDEX = 26
NAIL_DOF_INDEX = 26
DEFAULT_NAIL_FRICTIONLOSS = 10.0
DEFAULT_HANDLE_SLIDING_FRICTION = 2.5
DEFAULT_HEAD_NAIL_SLIDING_FRICTION = 0.05

# Holdable pre-strike pose measured in the PEMP reset state.
RAISED_ARM_QPOS = np.array([-0.081, -0.209, 0.030, -0.712], dtype=np.float32)

# A firm constant grasp.  The first four action dimensions are arm/wrist; the
# remaining 22 dimensions are finger/thumb targets.
GRASP_ACTION = np.array(
    [
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -0.302,
        -0.848,
        1.0,
        1.0,
        1.0,
        0.694,
        -0.079,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)


@dataclass
class Trajectory:
    actions: np.ndarray
    phases: np.ndarray
    warmup_actions: np.ndarray
    tool_positions: np.ndarray
    nail_positions: np.ndarray
    contacts: np.ndarray
    nail_qpos: np.ndarray
    strike_hits: np.ndarray
    strike_first_hit_steps: np.ndarray
    goal_distance: float
    action_cycle_max_error: float
    tool_cycle_max_error: float
    relative_tool_nail_cycle_max_error: float


def arm_action_for_qpos(base_env, qpos4: np.ndarray) -> np.ndarray:
    return np.asarray((qpos4 - base_env.act_mean[:4]) / base_env.act_rng[:4], dtype=np.float32)


def full_action(arm4: np.ndarray) -> np.ndarray:
    action = np.empty(26, dtype=np.float32)
    action[:4] = arm4
    action[4:] = GRASP_ACTION
    return np.clip(action, -1.0, 1.0)


def hide_mujoco_info_pane(env: gym.Env) -> None:
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    if renderer is None:
        return
    viewer = renderer._get_viewer(render_mode="human")
    if hasattr(viewer, "_hide_menu"):
        viewer._hide_menu = True


def body_id(base_env, name: str) -> int | None:
    for index in range(base_env.model.nbody):
        if base_env.model.body(index).name == name:
            return index
    return None


def configure_model(
    env: gym.Env,
    nail_frictionloss: float,
    handle_sliding_friction: float,
    head_nail_sliding_friction: float,
    board_pos: np.ndarray | None,
) -> None:
    base_env = env.unwrapped
    model = base_env.model

    model.dof_frictionloss[NAIL_DOF_INDEX] = float(nail_frictionloss)

    object_body_id = body_id(base_env, "Object")
    nail_body_id = body_id(base_env, "nail")
    board_body_id = body_id(base_env, "nail_board")

    for geom_id in range(model.ngeom):
        geom_body_id = int(model.geom_bodyid[geom_id])
        geom_name = model.geom(geom_id).name
        if geom_body_id == object_body_id:
            if geom_name == "handle":
                model.geom_friction[geom_id, 0] = float(handle_sliding_friction)
            else:
                model.geom_friction[geom_id, 0] = float(head_nail_sliding_friction)
        elif geom_body_id == nail_body_id:
            model.geom_friction[geom_id, 0] = float(head_nail_sliding_friction)
        elif geom_body_id == board_body_id:
            model.geom_friction[geom_id, 0] = max(
                model.geom_friction[geom_id, 0],
                float(head_nail_sliding_friction),
            )

    if board_pos is not None and hasattr(base_env, "set_env_state"):
        state = (
            base_env.get_env_state()
            if hasattr(base_env, "get_env_state")
            else {"qpos": base_env.data.qpos.copy(), "qvel": base_env.data.qvel.copy()}
        )
        state["board_pos"] = np.asarray(board_pos, dtype=np.float64)
        base_env.set_env_state(state)


def make_env(
    render: bool,
    max_episode_steps: int,
    seed: int,
    nail_frictionloss: float,
    handle_sliding_friction: float,
    head_nail_sliding_friction: float,
    board_pos: np.ndarray | None,
    hide_info_pane: bool,
) -> gym.Env:
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    configure_model(
        env,
        nail_frictionloss=nail_frictionloss,
        handle_sliding_friction=handle_sliding_friction,
        head_nail_sliding_friction=head_nail_sliding_friction,
        board_pos=board_pos,
    )
    if render and hide_info_pane:
        hide_mujoco_info_pane(env)
    return env


def task_positions(env: gym.Env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_env = env.unwrapped
    tool = base_env.data.site_xpos[base_env.tool_site_id].copy()
    nail = base_env.data.site_xpos[base_env.target_obj_site_id].copy()
    goal = base_env.data.site_xpos[base_env.goal_site_id].copy()
    return tool, nail, goal


def goal_distance(env: gym.Env) -> float:
    _, nail, goal = task_positions(env)
    return float(np.linalg.norm(nail - goal))


def hammer_nail_contact(env: gym.Env) -> bool:
    base_env = env.unwrapped
    model = base_env.model
    data = base_env.data
    object_body_id = body_id(base_env, "Object")
    nail_body_id = body_id(base_env, "nail")
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


def build_action_templates(
    base_env,
    strikes: int,
    ramp_steps: int,
    hold_steps: int,
    retract_steps: int,
    settle_steps: int,
    warmup_steps: int,
    jitter: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raised4 = arm_action_for_qpos(base_env, RAISED_ARM_QPOS)
    struck4 = np.array([0.2, raised4[1], -1.0, raised4[3]], dtype=np.float32)
    if jitter > 0.0:
        # These two dimensions alter the style without destroying contact.
        raised4[1] += rng.uniform(-jitter, jitter)
        raised4[3] += rng.uniform(-jitter, jitter)
        struck4[1] = raised4[1]
        struck4[3] = raised4[3]

    reset4 = arm_action_for_qpos(base_env, base_env.data.qpos[:4].copy())
    warmup_actions = []
    for step in range(warmup_steps):
        alpha = (step + 1) / warmup_steps
        warmup_actions.append(full_action(reset4 + alpha * (raised4 - reset4)))

    cycle_actions: list[np.ndarray] = []
    cycle_phases: list[str] = []
    for step in range(ramp_steps):
        alpha = (step + 1) / ramp_steps
        cycle_actions.append(full_action(raised4 + alpha * (struck4 - raised4)))
        cycle_phases.append("swing")
    for _ in range(hold_steps):
        cycle_actions.append(full_action(struck4))
        cycle_phases.append("swing")
    for step in range(retract_steps):
        alpha = (step + 1) / retract_steps
        cycle_actions.append(full_action(struck4 + alpha * (raised4 - struck4)))
        cycle_phases.append("retract")
    for _ in range(settle_steps):
        cycle_actions.append(full_action(raised4))
        cycle_phases.append("settle")

    actions = []
    phases = []
    for strike in range(1, strikes + 1):
        for action, phase in zip(cycle_actions, cycle_phases, strict=True):
            actions.append(action)
            phases.append(f"strike_{strike}_{phase}")

    return (
        np.asarray(warmup_actions, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(phases, dtype="U32"),
    )


def cycle_max_error(values: np.ndarray, strikes: int) -> float:
    if strikes < 2 or len(values) == 0:
        return 0.0
    cycle_steps = len(values) // strikes
    if cycle_steps * strikes != len(values):
        return float("nan")
    cycles = values.reshape(strikes, cycle_steps, -1)
    return float(np.max(np.abs(cycles - cycles[0])))


def execute_actions(
    env: gym.Env,
    actions: np.ndarray,
    phases: np.ndarray,
    strikes: int,
    render: bool,
    sleep_s: float,
) -> Trajectory:
    tool_positions: list[np.ndarray] = []
    nail_positions: list[np.ndarray] = []
    contacts: list[bool] = []
    nail_qpos: list[float] = []
    strike_hits = np.zeros(strikes, dtype=bool)
    strike_first_hit_steps = np.full(strikes, -1, dtype=np.int64)

    for action_index, (action, phase) in enumerate(zip(actions, phases, strict=True)):
        env.step(np.clip(action, -1.0, 1.0))
        contact = hammer_nail_contact(env)
        tool, nail, _ = task_positions(env)
        tool_positions.append(tool)
        nail_positions.append(nail)
        contacts.append(contact)
        nail_qpos.append(float(env.unwrapped.data.qpos[NAIL_QPOS_INDEX]))

        if contact and "_swing" in phase:
            strike_index = int(phase.split("_")[1]) - 1
            if 0 <= strike_index < strikes and not strike_hits[strike_index]:
                strike_hits[strike_index] = True
                strike_first_hit_steps[strike_index] = action_index

        if render:
            env.render()
            if sleep_s > 0:
                time.sleep(sleep_s)

    tool_array = np.asarray(tool_positions, dtype=np.float64)
    nail_array = np.asarray(nail_positions, dtype=np.float64)
    relative = tool_array - nail_array
    return Trajectory(
        actions=np.asarray(actions, dtype=np.float32),
        phases=np.asarray(phases, dtype="U32"),
        warmup_actions=np.empty((0, 26), dtype=np.float32),
        tool_positions=tool_array,
        nail_positions=nail_array,
        contacts=np.asarray(contacts, dtype=bool),
        nail_qpos=np.asarray(nail_qpos, dtype=np.float32),
        strike_hits=strike_hits,
        strike_first_hit_steps=strike_first_hit_steps,
        goal_distance=goal_distance(env),
        action_cycle_max_error=cycle_max_error(actions, strikes),
        tool_cycle_max_error=cycle_max_error(tool_array, strikes),
        relative_tool_nail_cycle_max_error=cycle_max_error(relative, strikes),
    )


def record_trajectory(
    strikes: int,
    seed: int,
    render: bool,
    sleep_s: float,
    warmup_steps: int,
    ramp_steps: int,
    hold_steps: int,
    retract_steps: int,
    settle_steps: int,
    jitter: float,
    nail_frictionloss: float,
    handle_sliding_friction: float,
    head_nail_sliding_friction: float,
    board_pos: np.ndarray | None,
    hide_info_pane: bool,
) -> Trajectory:
    total_steps = warmup_steps + strikes * (ramp_steps + hold_steps + retract_steps + settle_steps) + 50
    env = make_env(
        render=render,
        max_episode_steps=total_steps,
        seed=seed,
        nail_frictionloss=nail_frictionloss,
        handle_sliding_friction=handle_sliding_friction,
        head_nail_sliding_friction=head_nail_sliding_friction,
        board_pos=board_pos,
        hide_info_pane=hide_info_pane,
    )

    warmup_actions, actions, phases = build_action_templates(
        env.unwrapped,
        strikes=strikes,
        ramp_steps=ramp_steps,
        hold_steps=hold_steps,
        retract_steps=retract_steps,
        settle_steps=settle_steps,
        warmup_steps=warmup_steps,
        jitter=jitter,
        rng=np.random.default_rng(seed),
    )

    for action in warmup_actions:
        env.step(action)
        if render:
            env.render()
            if sleep_s > 0:
                time.sleep(sleep_s)

    trajectory = execute_actions(
        env,
        actions=actions,
        phases=phases,
        strikes=strikes,
        render=render,
        sleep_s=sleep_s,
    )
    env.close()
    trajectory.warmup_actions = warmup_actions
    return trajectory


def save_trajectory(
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
        seed=np.array(seed, dtype=np.int64),
        env_id=np.array(ENV_ID),
        controller=np.array("adroit_cartesian_periodic_hammer_controller"),
        pure_action_trajectory=np.array(True, dtype=bool),
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def replay_trajectory(
    path: Path,
    render: bool,
    sleep_s: float,
    hide_info_pane: bool,
    no_warmup: bool,
) -> Trajectory:
    data = load_npz(path)
    actions = data["actions"].astype(np.float32)
    phases = data["phases"].astype("U32")
    warmup_actions = data.get("warmup_actions", np.empty((0, 26), dtype=np.float32)).astype(
        np.float32
    )
    strikes = int(data["num_strikes"].item()) if "num_strikes" in data else 1
    seed = int(data["seed"].item()) if "seed" in data else 0
    board_pos = data.get("board_pos")
    if board_pos is not None and np.isnan(board_pos).any():
        board_pos = None

    env = make_env(
        render=render,
        max_episode_steps=len(warmup_actions) + len(actions) + 50,
        seed=seed,
        nail_frictionloss=float(data.get("nail_frictionloss", DEFAULT_NAIL_FRICTIONLOSS)),
        handle_sliding_friction=float(
            data.get("handle_sliding_friction", DEFAULT_HANDLE_SLIDING_FRICTION)
        ),
        head_nail_sliding_friction=float(
            data.get("head_nail_sliding_friction", DEFAULT_HEAD_NAIL_SLIDING_FRICTION)
        ),
        board_pos=board_pos,
        hide_info_pane=hide_info_pane,
    )

    if not no_warmup:
        for action in warmup_actions:
            env.step(action)
            if render:
                env.render()
                if sleep_s > 0:
                    time.sleep(sleep_s)

    trajectory = execute_actions(
        env,
        actions=actions,
        phases=phases,
        strikes=strikes,
        render=render,
        sleep_s=sleep_s,
    )
    env.close()
    trajectory.warmup_actions = warmup_actions if not no_warmup else np.empty((0, 26), dtype=np.float32)
    return trajectory


def print_summary(prefix: str, trajectory: Trajectory) -> None:
    print(
        f"{prefix} steps={len(trajectory.actions)} "
        f"strike_hits={trajectory.strike_hits.tolist()} "
        f"first_hit_steps={trajectory.strike_first_hit_steps.tolist()} "
        f"final_nail_qpos={trajectory.nail_qpos[-1]:.4f} "
        f"goal_distance={trajectory.goal_distance:.4f} "
        f"action_cycle_max_error={trajectory.action_cycle_max_error:.6f} "
        f"tool_cycle_max_error={trajectory.tool_cycle_max_error:.4f} "
        f"relative_tool_nail_cycle_max_error={trajectory.relative_tool_nail_cycle_max_error:.4f}"
    )


def output_path(base_dir: Path, index: int) -> Path:
    return base_dir / f"{index}.npz"


def parse_board_pos(values: list[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("--board-pos expects exactly three values: x y z")
    return np.asarray(values, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-actions", type=Path)
    parser.add_argument("--save-actions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("sim/data/raw_cartesian_periodic"))
    parser.add_argument("--strikes", type=int, default=3)
    parser.add_argument("--min-strikes", type=int, default=3)
    parser.add_argument("--max-strikes", type=int, default=6)
    parser.add_argument("--demos-per-frequency", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--show-info-pane", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--ramp-steps", type=int, default=10)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--retract-steps", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--jitter", type=float, default=0.006)
    parser.add_argument("--nail-frictionloss", type=float, default=DEFAULT_NAIL_FRICTIONLOSS)
    parser.add_argument("--handle-sliding-friction", type=float, default=DEFAULT_HANDLE_SLIDING_FRICTION)
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
        save_trajectory(
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
        for demo_index in range(args.demos_per_frequency):
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
            save_trajectory(
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
