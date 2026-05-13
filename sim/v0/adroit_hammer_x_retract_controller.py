"""Adroit hammer controller with an explicit retract point X.

The controller performs:
1. pickup,
2. move to a pre-impact point X,
3. swing the hammer,
4. retract to X,
5. repeat the swing/retract sequence three times.

It can save the exact normalized actions that were executed and replay them
later with the same seed. The default keeps ``--settle-steps`` at 0 because
stepping the simulator while holding at X makes the hammer sag enough to miss
later nail contacts. Use ``--settle-steps`` for experiments, but verify the
reported hammer-head/nail contacts before collecting demonstrations.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

try:
    from sim.adroit_scripted_hammer_controller import (
        CONTACT_DISTANCE,
        ENV_ID,
        IMPACT_PHASE_STEP,
        NUM_STRIKES,
        PICKUP_ACTIONS,
        STRIKE_CYCLE_STEPS,
        STRIKE_TEMPLATE_ACTIONS,
        get_hammer_nail_geom_ids,
        get_task_sites,
        hammer_nail_contact,
        hide_mujoco_info_pane,
        resolve_load_path,
        select_closed_loop_strike_action,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from sim.adroit_scripted_hammer_controller import (
        CONTACT_DISTANCE,
        ENV_ID,
        IMPACT_PHASE_STEP,
        NUM_STRIKES,
        PICKUP_ACTIONS,
        STRIKE_CYCLE_STEPS,
        STRIKE_TEMPLATE_ACTIONS,
        get_hammer_nail_geom_ids,
        get_task_sites,
        hammer_nail_contact,
        hide_mujoco_info_pane,
        resolve_load_path,
        select_closed_loop_strike_action,
    )


gym.register_envs(gymnasium_robotics)


GO_TO_X_ACTIONS = STRIKE_TEMPLATE_ACTIONS[:IMPACT_PHASE_STEP]
SWING_ACTIONS = STRIKE_TEMPLATE_ACTIONS[IMPACT_PHASE_STEP:]
X_ACTION = GO_TO_X_ACTIONS[-1]


def normalize_save_path(path: Path) -> Path:
    if path.suffix:
        return path
    return path.with_suffix(".npz")


def episode_save_path(base_path: Path, seed: int, episode_count: int) -> Path:
    if episode_count == 1:
        return base_path
    suffix = base_path.suffix or ".npz"
    return base_path.with_name(f"{base_path.stem}_seed{seed}{suffix}")


def phase_name(step: int, phases: list[str]) -> str:
    if step < len(phases):
        return phases[step]
    return "replay"


def save_action_trajectory(
    path: Path,
    actions: np.ndarray,
    phases: list[str],
    seed: int,
    settle_steps: int,
    final_nail_displacement: float,
    strike_min_distances: list[float],
    strike_head_nail_contacts: list[bool],
    strike_contact_steps: list[list[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_contact_count = max(1, max(len(steps) for steps in strike_contact_steps))
    contact_steps = np.full((NUM_STRIKES, max_contact_count), -1, dtype=np.int64)
    for strike_index, steps in enumerate(strike_contact_steps):
        contact_steps[strike_index, : len(steps)] = steps

    np.savez_compressed(
        path,
        actions=actions.astype(np.float32),
        phases=np.array(phases),
        seed=np.array(seed, dtype=np.int64),
        env_id=np.array(ENV_ID),
        controller=np.array("x_retract"),
        num_strikes=np.array(NUM_STRIKES, dtype=np.int64),
        x_step=np.array(IMPACT_PHASE_STEP - 1, dtype=np.int64),
        settle_steps=np.array(settle_steps, dtype=np.int64),
        strike_start_step=np.array(len(PICKUP_ACTIONS), dtype=np.int64),
        strike_cycle_steps=np.array(STRIKE_CYCLE_STEPS, dtype=np.int64),
        final_nail_displacement=np.array(final_nail_displacement, dtype=np.float32),
        strike_min_tool_nail=np.array(strike_min_distances, dtype=np.float32),
        strike_head_nail_contacts=np.array(strike_head_nail_contacts, dtype=bool),
        strike_contact_steps=contact_steps,
    )
    print(f"saved {len(actions)} actions to {path}")


def load_replay_trajectory(path: Path) -> tuple[np.ndarray, list[str] | None, int | None]:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if "actions" not in loaded:
                raise ValueError(f"{path} does not contain an 'actions' array")
            actions = loaded["actions"].astype(np.float32)
            phases = loaded["phases"].astype(str).tolist() if "phases" in loaded else None
            seed = int(loaded["seed"].item()) if "seed" in loaded else None
    else:
        actions = loaded.astype(np.float32)
        phases = None
        seed = None

    if actions.ndim != 2 or actions.shape[1] != 26:
        raise ValueError(f"expected actions with shape (N, 26), got {actions.shape}")
    return np.clip(actions, -1.0, 1.0), phases, seed


def strike_index_from_phase(phase: str) -> int | None:
    if not phase.startswith("strike_") or "_swing" not in phase:
        return None
    try:
        return int(phase.split("_", maxsplit=2)[1]) - 1
    except (ValueError, IndexError):
        return None


def build_controller_action(
    env: gym.Env,
    action: np.ndarray,
    local_step: int,
    reference_offsets: list[np.ndarray],
    lookahead_steps: int,
) -> np.ndarray:
    if len(reference_offsets) < STRIKE_CYCLE_STEPS:
        return action
    return select_closed_loop_strike_action(
        env,
        action,
        reference_offsets,
        local_step,
        lookahead_steps,
    )


def run_episode(
    seed: int,
    render: bool,
    sleep_s: float,
    verbose: bool,
    hide_info_pane: bool,
    lookahead_steps: int,
    settle_steps: int,
    stop_on_contact: bool,
    replay_actions: np.ndarray | None,
    replay_phases: list[str] | None,
    save_actions_path: Path | None,
) -> bool:
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=250)
    obs, _ = env.reset(seed=seed)
    if render and hide_info_pane:
        hide_mujoco_info_pane(env)

    hammer_head_geom_id, nail_geom_ids = get_hammer_nail_geom_ids(env)
    success = False
    first_success_step = None
    executed_actions: list[np.ndarray] = []
    phases: list[str] = []
    reference_offsets: list[np.ndarray] = []
    strike_min_distances = [np.inf for _ in range(NUM_STRIKES)]
    strike_head_nail_contacts = [False for _ in range(NUM_STRIKES)]
    strike_contact_steps: list[list[int]] = [[] for _ in range(NUM_STRIKES)]

    if replay_actions is not None:
        planned_actions = []
        for action_index, action in enumerate(replay_actions):
            phase = (
                replay_phases[action_index]
                if replay_phases is not None and action_index < len(replay_phases)
                else "replay"
            )
            planned_actions.append((action, phase, strike_index_from_phase(phase), None))
    else:
        planned_actions = []
        planned_actions.extend((action, "pickup", None, None) for action in PICKUP_ACTIONS)
        planned_actions.extend(
            (action, "go_to_x", None, local_step)
            for local_step, action in enumerate(GO_TO_X_ACTIONS)
        )
        planned_actions.extend((X_ACTION, "settle_at_x", None, None) for _ in range(settle_steps))

        for strike_index in range(NUM_STRIKES):
            planned_actions.extend(
                (action, f"strike_{strike_index + 1}_swing", strike_index, None)
                for action in SWING_ACTIONS
            )
            planned_actions.extend(
                (action, f"strike_{strike_index + 1}_retract_to_x", None, local_step)
                for local_step, action in enumerate(GO_TO_X_ACTIONS)
            )

    skip_remaining_swing = False
    for step, (base_action, phase, strike_index, x_local_step) in enumerate(planned_actions):
        if replay_actions is not None:
            action = base_action
        elif phase.endswith("_retract_to_x"):
            action = build_controller_action(
                env,
                base_action,
                int(x_local_step),
                reference_offsets,
                lookahead_steps,
            )
        elif phase.endswith("_swing") and skip_remaining_swing:
            continue
        else:
            action = base_action

        executed_actions.append(np.asarray(action, dtype=np.float32).copy())
        phases.append(phase)
        obs, reward, terminated, truncated, info = env.step(action)
        step_success = bool(info.get("success", False))
        if step_success and first_success_step is None:
            first_success_step = len(executed_actions) - 1
        success = success or step_success

        palm, tool, nail = get_task_sites(env)
        tool_nail_dist = float(np.linalg.norm(tool - nail))
        head_nail_contact, _ = hammer_nail_contact(env, hammer_head_geom_id, nail_geom_ids)
        action_step = len(executed_actions) - 1

        if replay_actions is None and phase == "go_to_x":
            reference_offsets.append(tool - nail)
        if replay_actions is None and phase == "strike_1_swing":
            reference_offsets.append(tool - nail)

        if strike_index is not None:
            strike_min_distances[strike_index] = min(
                strike_min_distances[strike_index],
                tool_nail_dist,
            )
            if head_nail_contact:
                strike_head_nail_contacts[strike_index] = True
                strike_contact_steps[strike_index].append(action_step)
                skip_remaining_swing = stop_on_contact
        else:
            skip_remaining_swing = False

        should_log = (
            verbose
            or action_step % 10 == 0
            or action_step == first_success_step
            or action_step == len(planned_actions) - 1
            or head_nail_contact
        )
        if should_log:
            print(
                f"{action_step:03d} {phase_name(action_step, phases):22s} "
                f"nail={obs[26]: .4f} tool_nail={tool_nail_dist: .4f} "
                f"tool_z={tool[2]: .4f} palm_z={palm[2]: .4f} "
                f"head_nail_contact={head_nail_contact} "
                f"reward={reward: .3f} success={step_success}"
            )

        if render and sleep_s > 0:
            time.sleep(sleep_s)
        if terminated or truncated:
            break

    env.close()
    final_nail_displacement = float(obs[26])
    strike_site_contacts = [bool(distance <= CONTACT_DISTANCE) for distance in strike_min_distances]
    print(
        f"finished seed={seed} success={success} "
        f"first_success_step={first_success_step} "
        f"final_nail_displacement={final_nail_displacement:.4f} "
        f"strike_min_tool_nail={[round(float(d), 4) for d in strike_min_distances]} "
        f"strike_site_contacts={strike_site_contacts} "
        f"strike_head_nail_contacts={strike_head_nail_contacts} "
        f"strike_contact_steps={strike_contact_steps}"
    )

    if save_actions_path is not None:
        save_action_trajectory(
            save_actions_path,
            np.asarray(executed_actions, dtype=np.float32),
            phases,
            seed=seed,
            settle_steps=settle_steps,
            final_nail_displacement=final_nail_displacement,
            strike_min_distances=[float(distance) for distance in strike_min_distances],
            strike_head_nail_contacts=strike_head_nail_contacts,
            strike_contact_steps=strike_contact_steps,
        )
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--show-info-pane", action="store_true")
    parser.add_argument("--lookahead", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--stop-on-contact", action="store_true")
    parser.add_argument("--save-actions", type=Path)
    parser.add_argument("--load-actions", type=Path)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay_actions = None
    replay_phases = None
    if args.load_actions is not None:
        load_path = resolve_load_path(args.load_actions)
        replay_actions, replay_phases, saved_seed = load_replay_trajectory(load_path)
        print(f"loaded {len(replay_actions)} actions from {load_path}")
        if saved_seed is not None and saved_seed != args.seed:
            print(
                f"warning: trajectory was saved with seed={saved_seed}, "
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
            lookahead_steps=args.lookahead,
            settle_steps=args.settle_steps,
            stop_on_contact=args.stop_on_contact,
            replay_actions=replay_actions,
            replay_phases=replay_phases,
            save_actions_path=(
                episode_save_path(save_actions_path, episode_seed, args.episodes)
                if save_actions_path is not None
                else None
            ),
        )
