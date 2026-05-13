"""Three-strike controller for ``AdroitHandHammer-vPEMP``.

The PEMP wrapper starts with the hammer already held.  This controller moves
to a pre-strike pose X, then repeats a deterministic swing/retract/settle
cycle three times.  By default the retract and settle phases are state-assisted:
the nail progress is preserved, while the hand and hammer are returned to the
same measured home pose X before every strike.  That gives a clean repeated
motion for visual inspection and trajectory replay, instead of relying on a
post-impact open-loop action sequence whose hammer pose drifts after contacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import adroit_hand_hammer_updated  # noqa: F401  Registers AdroitHandHammer-vPEMP.
from adroit_pemp_controller import ENV_ID, IMPACT_ACTIONS, RETRACT_ACTIONS


gym.register_envs(gymnasium_robotics)

NUM_STRIKES = 3
INITIAL_GO_TO_X_STEPS = len(RETRACT_ACTIONS)
RETRACT_TO_X_STEPS = 9
SETTLE_STEPS = 10
SETTLE_ACTION_INDEX = 6
NAIL_IMPACT_THRESHOLD = 0.5
POST_CONTACT_FOLLOW_THROUGH_STEPS = 2
KINEMATIC_RETRACT_STEPS = 18
CONTROLLER_NAME = "pemp_three_strike_fixed_home_state_assisted"


@dataclass
class ReplayTrajectory:
    actions: np.ndarray
    phases: list[str] | None
    seed: int | None
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    ctrl: np.ndarray | None = None
    sim_time: np.ndarray | None = None
    state_driven: np.ndarray | None = None


def normalize_save_path(path: Path) -> Path:
    if path.suffix:
        return path
    return path.with_suffix(".npz")


def resolve_load_path(path: Path) -> Path:
    if path.exists() or path.suffix:
        return path
    npz_path = path.with_suffix(".npz")
    return npz_path if npz_path.exists() else path


def episode_save_path(base_path: Path, seed: int, episode_count: int) -> Path:
    if episode_count == 1:
        return base_path
    suffix = base_path.suffix or ".npz"
    return base_path.with_name(f"{base_path.stem}_seed{seed}{suffix}")


def hide_mujoco_info_pane(env: gym.Env) -> None:
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    if renderer is None:
        return
    viewer = renderer._get_viewer(render_mode="human")
    if hasattr(viewer, "_hide_menu"):
        viewer._hide_menu = True


def object_nail_contact_names(env: gym.Env) -> list[str]:
    base_env = env.unwrapped
    model = base_env.model
    data = base_env.data

    object_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == int(base_env.obj_body_id)
    }
    nail_body_id = next(
        body_id for body_id in range(model.nbody) if model.body(body_id).name == "nail"
    )
    nail_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == nail_body_id
    }

    contacts: list[str] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        object_hits_nail = geom1 in object_geom_ids and geom2 in nail_geom_ids
        nail_hits_object = geom2 in object_geom_ids and geom1 in nail_geom_ids
        if object_hits_nail or nail_hits_object:
            name1 = model.geom(geom1).name or f"geom_{geom1}"
            name2 = model.geom(geom2).name or f"geom_{geom2}"
            contacts.append(f"{name1}:{name2}")
    return contacts


def task_positions(env: gym.Env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_env = env.unwrapped
    tool = base_env.data.site_xpos[base_env.tool_site_id].copy()
    nail = base_env.data.site_xpos[base_env.target_obj_site_id].copy()
    goal = base_env.data.site_xpos[base_env.goal_site_id].copy()
    return tool, nail, goal


def goal_distance(env: gym.Env) -> float:
    _, nail, goal = task_positions(env)
    return float(np.linalg.norm(nail - goal))


def body_state_indices(
    base_env, body_name: str
) -> tuple[np.ndarray, np.ndarray]:
    model = base_env.model
    body_id = next(
        body_index
        for body_index in range(model.nbody)
        if model.body(body_index).name == body_name
    )

    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    qpos_width_by_joint_type = {0: 7, 1: 4, 2: 1, 3: 1}
    qvel_width_by_joint_type = {0: 6, 1: 3, 2: 1, 3: 1}
    joint_start = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    for joint_id in range(joint_start, joint_start + joint_count):
        joint_type = int(model.jnt_type[joint_id])
        qpos_start = int(model.jnt_qposadr[joint_id])
        qvel_start = int(model.jnt_dofadr[joint_id])
        qpos_width = qpos_width_by_joint_type[joint_type]
        qvel_width = qvel_width_by_joint_type[joint_type]
        qpos_indices.extend(range(qpos_start, qpos_start + qpos_width))
        qvel_indices.extend(range(qvel_start, qvel_start + qvel_width))

    return np.array(qpos_indices, dtype=np.int64), np.array(qvel_indices, dtype=np.int64)


def set_sim_state(
    base_env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray | None = None,
    sim_time: float | None = None,
) -> None:
    base_env.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
    if ctrl is not None:
        base_env.data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
    if sim_time is not None:
        base_env.data.time = float(sim_time)


def home_state_preserving_nail(
    base_env,
    home_qpos: np.ndarray,
    home_qvel: np.ndarray,
    nail_qpos_indices: np.ndarray,
    nail_qvel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = home_qpos.copy()
    qvel = home_qvel.copy()
    qpos[nail_qpos_indices] = base_env.data.qpos[nail_qpos_indices]
    qvel[nail_qvel_indices] = base_env.data.qvel[nail_qvel_indices]
    return qpos, qvel


def load_action_trajectory(path: Path) -> ReplayTrajectory:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if "actions" not in loaded:
                raise ValueError(f"{path} does not contain an 'actions' array")
            actions = loaded["actions"].astype(np.float32)
            phases = loaded["phases"].astype(str).tolist() if "phases" in loaded else None
            seed = int(loaded["seed"].item()) if "seed" in loaded else None
            qpos = loaded["qpos"].astype(np.float64) if "qpos" in loaded else None
            qvel = loaded["qvel"].astype(np.float64) if "qvel" in loaded else None
            ctrl = loaded["ctrl"].astype(np.float64) if "ctrl" in loaded else None
            sim_time = (
                loaded["sim_time"].astype(np.float64) if "sim_time" in loaded else None
            )
            state_driven = (
                loaded["state_driven"].astype(bool) if "state_driven" in loaded else None
            )
    else:
        actions = loaded.astype(np.float32)
        phases = None
        seed = None
        qpos = None
        qvel = None
        ctrl = None
        sim_time = None
        state_driven = None

    if actions.ndim != 2 or actions.shape[1] != 26:
        raise ValueError(f"expected actions with shape (N, 26), got {actions.shape}")

    for name, values in (
        ("qpos", qpos),
        ("qvel", qvel),
        ("ctrl", ctrl),
        ("state_driven", state_driven),
    ):
        if values is not None and len(values) != len(actions):
            raise ValueError(
                f"{path} has {name} length {len(values)}, expected {len(actions)}"
            )

    return ReplayTrajectory(
        actions=np.clip(actions, -1.0, 1.0),
        phases=phases,
        seed=seed,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        sim_time=sim_time,
        state_driven=state_driven,
    )


def save_action_trajectory(
    path: Path,
    actions: np.ndarray,
    phases: list[str],
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    sim_time: np.ndarray,
    state_driven: np.ndarray,
    seed: int,
    final_goal_distance: float,
    success: bool,
    strike_hits: list[bool],
    strike_contact_steps: list[int],
    home_tool: np.ndarray | None,
    action_only: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        actions=actions.astype(np.float32),
        phases=np.array(phases, dtype="U48"),
        qpos=qpos.astype(np.float64),
        qvel=qvel.astype(np.float64),
        ctrl=ctrl.astype(np.float64),
        sim_time=sim_time.astype(np.float64),
        state_driven=state_driven.astype(bool),
        seed=np.array(seed, dtype=np.int64),
        env_id=np.array(ENV_ID),
        controller=np.array(CONTROLLER_NAME),
        num_strikes=np.array(NUM_STRIKES, dtype=np.int64),
        initial_go_to_x_steps=np.array(INITIAL_GO_TO_X_STEPS, dtype=np.int64),
        retract_to_x_steps=np.array(RETRACT_TO_X_STEPS, dtype=np.int64),
        settle_steps=np.array(SETTLE_STEPS, dtype=np.int64),
        settle_action_index=np.array(SETTLE_ACTION_INDEX, dtype=np.int64),
        post_contact_follow_through_steps=np.array(
            POST_CONTACT_FOLLOW_THROUGH_STEPS, dtype=np.int64
        ),
        kinematic_retract_steps=np.array(KINEMATIC_RETRACT_STEPS, dtype=np.int64),
        nail_impact_threshold=np.array(NAIL_IMPACT_THRESHOLD, dtype=np.float32),
        final_goal_distance=np.array(final_goal_distance, dtype=np.float32),
        success=np.array(success, dtype=bool),
        strike_hits=np.array(strike_hits, dtype=bool),
        strike_contact_steps=np.array(strike_contact_steps, dtype=np.int64),
        home_tool=(
            np.asarray(home_tool, dtype=np.float64)
            if home_tool is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        state_assisted_homing=np.array(not action_only, dtype=bool),
    )
    state_count = int(np.asarray(state_driven, dtype=bool).sum())
    print(f"saved {len(actions)} trajectory steps to {path} ({state_count} state-driven)")


def strike_index_from_phase(phase: str) -> int | None:
    if not phase.startswith("strike_"):
        return None
    parts = phase.split("_")
    if len(parts) < 3:
        return None
    try:
        index = int(parts[1]) - 1
    except ValueError:
        return None
    return index if 0 <= index < NUM_STRIKES else None


def run_episode(
    seed: int,
    render: bool,
    sleep_s: float,
    verbose: bool,
    hide_info_pane: bool,
    replay_trajectory: ReplayTrajectory | None,
    save_actions_path: Path | None,
    action_only: bool,
) -> bool:
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=250)
    env.reset(seed=seed)
    if render and hide_info_pane:
        hide_mujoco_info_pane(env)

    base_env = env.unwrapped
    nail_qpos_indices, nail_qvel_indices = body_state_indices(base_env, "nail")
    executed_actions: list[np.ndarray] = []
    phases: list[str] = []
    executed_qpos: list[np.ndarray] = []
    executed_qvel: list[np.ndarray] = []
    executed_ctrl: list[np.ndarray] = []
    executed_sim_time: list[float] = []
    state_driven_steps: list[bool] = []
    strike_hits = [False for _ in range(NUM_STRIKES)]
    strike_contact_steps = [-1 for _ in range(NUM_STRIKES)]
    success = False
    first_success_step: int | None = None
    home_tool: np.ndarray | None = None

    def record_current_step(
        action: np.ndarray,
        phase: str,
        state_driven: bool,
        obs: np.ndarray | None,
        reward: float | None,
        info: dict | None,
        terminated: bool,
        truncated: bool,
    ) -> tuple[bool, bool, float]:
        nonlocal success, first_success_step

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        executed_actions.append(action.copy())
        phases.append(phase)
        executed_qpos.append(base_env.data.qpos.copy())
        executed_qvel.append(base_env.data.qvel.copy())
        executed_ctrl.append(base_env.data.ctrl.copy())
        executed_sim_time.append(float(base_env.data.time))
        # Downstream LfD code can filter these if it needs pure env.step samples.
        state_driven_steps.append(state_driven)

        action_step = len(executed_actions) - 1
        if obs is None:
            obs = base_env._get_obs()
        info = info or {}
        step_success = bool(info.get("success", False)) or goal_distance(env) <= 0.01
        if step_success and first_success_step is None:
            first_success_step = action_step
        success = success or step_success

        contact_names = object_nail_contact_names(env)
        nail_impact = float(obs[-1])
        hit = bool(contact_names) or nail_impact >= NAIL_IMPACT_THRESHOLD
        strike_index = strike_index_from_phase(phase)
        if hit and strike_index is not None and not strike_hits[strike_index]:
            strike_hits[strike_index] = True
            strike_contact_steps[strike_index] = action_step

        should_log = (
            verbose
            or action_step % 10 == 0
            or hit
            or step_success
            or terminated
            or truncated
        )
        if should_log:
            tool, nail, goal = task_positions(env)
            reward_text = "state" if reward is None else f"{reward:.3f}"
            mode = "state" if state_driven else "action"
            print(
                f"{action_step:03d} {phase:24s} {mode:6s} "
                f"goal_dist={np.linalg.norm(nail - goal):.4f} "
                f"tool={np.round(tool, 3)} "
                f"nail={np.round(nail, 3)} "
                f"impact={nail_impact:.2f} "
                f"contact={contact_names} "
                f"reward={reward_text} success={step_success}"
            )

        if render and state_driven:
            env.render()
        if render and sleep_s > 0:
            time.sleep(sleep_s)
        return terminated or truncated, hit, nail_impact

    def step_action(action: np.ndarray, phase: str) -> tuple[bool, bool, float]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        return record_current_step(
            action=action,
            phase=phase,
            state_driven=False,
            obs=obs,
            reward=float(reward),
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

    def record_state_step(action: np.ndarray, phase: str) -> tuple[bool, bool, float]:
        return record_current_step(
            action=action,
            phase=phase,
            state_driven=True,
            obs=base_env._get_obs(),
            reward=None,
            info=None,
            terminated=False,
            truncated=False,
        )

    def set_home_pose(home_qpos: np.ndarray, home_qvel: np.ndarray, home_ctrl: np.ndarray) -> None:
        qpos, qvel = home_state_preserving_nail(
            base_env, home_qpos, home_qvel, nail_qpos_indices, nail_qvel_indices
        )
        set_sim_state(base_env, qpos, qvel, home_ctrl)

    if replay_trajectory is not None:
        replay = replay_trajectory
        for action_index, action in enumerate(replay.actions):
            phase = (
                replay.phases[action_index]
                if replay.phases is not None and action_index < len(replay.phases)
                else "replay"
            )
            has_state_replay = (
                replay.state_driven is not None
                and replay.qpos is not None
                and replay.qvel is not None
                and bool(replay.state_driven[action_index])
            )
            if has_state_replay:
                ctrl = replay.ctrl[action_index] if replay.ctrl is not None else None
                sim_time = (
                    float(replay.sim_time[action_index])
                    if replay.sim_time is not None
                    else None
                )
                set_sim_state(
                    base_env,
                    replay.qpos[action_index],
                    replay.qvel[action_index],
                    ctrl,
                    sim_time,
                )
                done, _, _ = record_state_step(action, phase)
            else:
                done, _, _ = step_action(action, phase)
            if done:
                break
    else:
        for action in RETRACT_ACTIONS:
            done, _, _ = step_action(action, "go_to_x")
            if done:
                break

        if action_only:
            settle_action = RETRACT_ACTIONS[SETTLE_ACTION_INDEX]
            for strike_index in range(NUM_STRIKES):
                for action in IMPACT_ACTIONS:
                    done, hit, _ = step_action(
                        action,
                        f"strike_{strike_index + 1}_swing",
                    )
                    if done or hit:
                        break

                for action in RETRACT_ACTIONS[:RETRACT_TO_X_STEPS]:
                    done, _, _ = step_action(
                        action,
                        f"strike_{strike_index + 1}_retract_to_x",
                    )
                    if done:
                        break

                for _ in range(SETTLE_STEPS):
                    done, _, _ = step_action(
                        settle_action,
                        f"strike_{strike_index + 1}_settle_at_x",
                    )
                    if done:
                        break
        else:
            home_action = RETRACT_ACTIONS[-1]
            home_qpos = base_env.data.qpos.copy()
            home_qvel = base_env.data.qvel.copy()
            home_ctrl = base_env.data.ctrl.copy()
            home_tool = task_positions(env)[0]
            print(f"fixed home tool position X={np.round(home_tool, 4)}")

            for _ in range(SETTLE_STEPS):
                set_home_pose(home_qpos, home_qvel, home_ctrl)
                record_state_step(home_action, "initial_settle_at_x")

            episode_done = False
            for strike_index in range(NUM_STRIKES):
                strike_number = strike_index + 1
                set_home_pose(home_qpos, home_qvel, home_ctrl)
                record_state_step(home_action, f"strike_{strike_number}_home_x")

                first_hit_swing_step: int | None = None
                for swing_step, action in enumerate(IMPACT_ACTIONS):
                    done, hit, _ = step_action(action, f"strike_{strike_number}_swing")
                    if hit and first_hit_swing_step is None:
                        first_hit_swing_step = swing_step
                    if done:
                        episode_done = True
                        break
                    if (
                        first_hit_swing_step is not None
                        and swing_step - first_hit_swing_step
                        >= POST_CONTACT_FOLLOW_THROUGH_STEPS
                    ):
                        break
                if episode_done:
                    break

                start_qpos = base_env.data.qpos.copy()
                start_qvel = base_env.data.qvel.copy()
                nail_qpos = start_qpos[nail_qpos_indices].copy()
                nail_qvel = start_qvel[nail_qvel_indices].copy()
                for retract_step in range(KINEMATIC_RETRACT_STEPS):
                    alpha = (retract_step + 1) / KINEMATIC_RETRACT_STEPS
                    qpos = (1.0 - alpha) * start_qpos + alpha * home_qpos
                    qvel = (1.0 - alpha) * start_qvel + alpha * home_qvel
                    qpos[nail_qpos_indices] = nail_qpos
                    qvel[nail_qvel_indices] = nail_qvel
                    set_sim_state(base_env, qpos, qvel, home_ctrl)
                    record_state_step(home_action, f"strike_{strike_number}_retract_to_x")

                for _ in range(SETTLE_STEPS):
                    set_home_pose(home_qpos, home_qvel, home_ctrl)
                    record_state_step(home_action, f"strike_{strike_number}_settle_at_x")

    final_goal_distance = goal_distance(env)
    env.close()

    print(
        f"finished seed={seed} success={success} "
        f"first_success_step={first_success_step} "
        f"final_goal_distance={final_goal_distance:.4f} "
        f"strike_hits={strike_hits} "
        f"strike_contact_steps={strike_contact_steps} "
        f"home_tool={np.round(home_tool, 4) if home_tool is not None else None}"
    )

    if save_actions_path is not None:
        save_action_trajectory(
            save_actions_path,
            np.asarray(executed_actions, dtype=np.float32),
            phases,
            np.asarray(executed_qpos, dtype=np.float64),
            np.asarray(executed_qvel, dtype=np.float64),
            np.asarray(executed_ctrl, dtype=np.float64),
            np.asarray(executed_sim_time, dtype=np.float64),
            np.asarray(state_driven_steps, dtype=bool),
            seed=seed,
            final_goal_distance=final_goal_distance,
            success=success,
            strike_hits=strike_hits,
            strike_contact_steps=strike_contact_steps,
            home_tool=home_tool,
            action_only=action_only,
        )
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--show-info-pane", action="store_true")
    parser.add_argument("--save-actions", type=Path)
    parser.add_argument("--load-actions", type=Path)
    parser.add_argument(
        "--action-only",
        action="store_true",
        help="Use the older open-loop action retract instead of fixed-home state homing.",
    )
    parser.add_argument("--sleep", type=float, default=0.06)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    replay_trajectory = None
    if args.load_actions is not None:
        load_path = resolve_load_path(args.load_actions)
        replay_trajectory = load_action_trajectory(load_path)
        print(f"loaded {len(replay_trajectory.actions)} trajectory steps from {load_path}")
        if replay_trajectory.seed is not None and replay_trajectory.seed != args.seed:
            print(
                f"warning: trajectory was saved with seed={replay_trajectory.seed}, "
                f"but this run uses seed={args.seed}"
            )

    save_actions_path = (
        normalize_save_path(args.save_actions) if args.save_actions is not None else None
    )

    if not args.no_render:
        time.sleep(3)
    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        run_episode(
            seed=episode_seed,
            render=not args.no_render,
            sleep_s=args.sleep,
            verbose=args.verbose,
            hide_info_pane=not args.show_info_pane,
            replay_trajectory=replay_trajectory,
            save_actions_path=(
                episode_save_path(save_actions_path, episode_seed, args.episodes)
                if save_actions_path is not None
                else None
            ),
            action_only=args.action_only,
        )
