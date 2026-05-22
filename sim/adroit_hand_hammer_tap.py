"""Tap-oriented Adroit hammer environment wrapper.

``AdroitHandHammer-vPEMPTap`` starts from the PEMP hammer-in-hand state, moves
the hand to a raised pre-strike pose during reset, then places the nail at a
slightly inserted level.  This creates clearance for the first visible swing:
the hammer approaches freely and contacts the nail late in the downstroke
instead of grazing it early and pushing.

The reset-time setup is intentionally outside the saved demonstration
trajectory.  After reset, controller actions are ordinary ``env.step`` actions.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np

import adroit_hand_hammer_updated  # noqa: F401  Registers AdroitHandHammer-vPEMP.


gym.register_envs(gymnasium_robotics)

ENV_ID = "AdroitHandHammer-vPEMPTap"
NAIL_QPOS_INDEX = 26
NAIL_DOF_INDEX = 26

RAISED_ARM_QPOS = np.array([-0.081, -0.209, 0.030, -0.712], dtype=np.float32)
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


def arm_action_for_qpos(base_env, qpos4: np.ndarray) -> np.ndarray:
    return np.asarray((qpos4 - base_env.act_mean[:4]) / base_env.act_rng[:4], dtype=np.float32)


def full_action(arm4: np.ndarray) -> np.ndarray:
    action = np.empty(26, dtype=np.float32)
    action[:4] = arm4
    action[4:] = GRASP_ACTION
    return np.clip(action, -1.0, 1.0)


def find_body_id(base_env, name: str) -> int | None:
    for body_index in range(base_env.model.nbody):
        if base_env.model.body(body_index).name == name:
            return body_index
    return None


class AdroitHandHammerTapWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        nail_start_qpos: float = 0.04,
        nail_frictionloss: float = 15.0,
        handle_sliding_friction: float = 2.5,
        head_nail_sliding_friction: float = 0.05,
        board_pos: np.ndarray | None = None,
        warmup_steps: int = 40,
    ):
        super().__init__(env)
        self.nail_start_qpos = float(nail_start_qpos)
        self.nail_frictionloss = float(nail_frictionloss)
        self.handle_sliding_friction = float(handle_sliding_friction)
        self.head_nail_sliding_friction = float(head_nail_sliding_friction)
        self.board_pos = None if board_pos is None else np.asarray(board_pos, dtype=np.float64)
        self.warmup_steps = int(warmup_steps)

    def configure_model(self) -> None:
        base_env = self.unwrapped
        model = base_env.model

        model.dof_frictionloss[NAIL_DOF_INDEX] = self.nail_frictionloss

        object_body_id = find_body_id(base_env, "Object")
        nail_body_id = find_body_id(base_env, "nail")
        board_body_id = find_body_id(base_env, "nail_board")

        if self.board_pos is not None and board_body_id is not None:
            model.body_pos[board_body_id] = self.board_pos

        for geom_id in range(model.ngeom):
            geom_body_id = int(model.geom_bodyid[geom_id])
            geom_name = model.geom(geom_id).name
            if geom_body_id == object_body_id:
                if geom_name == "handle":
                    model.geom_friction[geom_id, 0] = self.handle_sliding_friction
                else:
                    model.geom_friction[geom_id, 0] = self.head_nail_sliding_friction
            elif geom_body_id == nail_body_id:
                model.geom_friction[geom_id, 0] = self.head_nail_sliding_friction

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        base_env = self.unwrapped
        self.configure_model()

        reset4 = arm_action_for_qpos(base_env, base_env.data.qpos[:4].copy())
        raised4 = arm_action_for_qpos(base_env, RAISED_ARM_QPOS)
        for step in range(self.warmup_steps):
            alpha = (step + 1) / self.warmup_steps
            self.env.step(full_action(reset4 + alpha * (raised4 - reset4)))

        base_env.data.qpos[NAIL_QPOS_INDEX] = self.nail_start_qpos
        base_env.data.qvel[NAIL_DOF_INDEX] = 0.0
        mujoco.mj_forward(base_env.model, base_env.data)
        obs = base_env._get_obs()
        info["nail_start_qpos"] = self.nail_start_qpos
        return obs, info


def make_adroit_hammer_pemp_tap(
    nail_start_qpos: float = 0.04,
    nail_frictionloss: float = 12.0,
    handle_sliding_friction: float = 2.5,
    head_nail_sliding_friction: float = 0.05,
    board_pos: np.ndarray | None = None,
    warmup_steps: int = 40,
    **kwargs,
):
    env = gym.make("AdroitHandHammer-vPEMP", **kwargs)
    return AdroitHandHammerTapWrapper(
        env,
        nail_start_qpos=nail_start_qpos,
        nail_frictionloss=nail_frictionloss,
        handle_sliding_friction=handle_sliding_friction,
        head_nail_sliding_friction=head_nail_sliding_friction,
        board_pos=board_pos,
        warmup_steps=warmup_steps,
    )


if ENV_ID not in gym.envs.registry:
    gym.envs.registration.register(
        id=ENV_ID,
        entry_point="adroit_hand_hammer_tap:make_adroit_hammer_pemp_tap",
    )
