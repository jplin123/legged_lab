"""Functions to specify symmetry in the observation and action space for Agibot X2 29dof."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Expected articulation joint order used by policy obs/actions.
EXPECTED_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

_JOINT_INDEX = {name: idx for idx, name in enumerate(EXPECTED_JOINT_ORDER)}
_LEFT_RIGHT_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_hip_roll_joint", "right_hip_roll_joint"),
    ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
    ("left_ankle_roll_joint", "right_ankle_roll_joint"),
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint"),
    ("left_elbow_joint", "right_elbow_joint"),
    ("left_wrist_roll_joint", "right_wrist_roll_joint"),
    ("left_wrist_pitch_joint", "right_wrist_pitch_joint"),
    ("left_wrist_yaw_joint", "right_wrist_yaw_joint"),
]
_ROLL_SIGN_JOINTS = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]
_YAW_SIGN_JOINTS = [
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
_WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
_WAIST_SIGN_JOINTS = ["waist_yaw_joint", "waist_roll_joint"]

# Cache validated env objects to avoid repeated checks every step.
_VALIDATED_ENVS: set[int] = set()


def _validate_joint_order_once(env: ManagerBasedRLEnv):
    env_key = id(env)
    if env_key in _VALIDATED_ENVS:
        return

    robot = env.scene["robot"]
    actual_joint_order = list(robot.joint_names)

    if len(actual_joint_order) != len(EXPECTED_JOINT_ORDER):
        raise ValueError(
            f"X2 symmetry expects {len(EXPECTED_JOINT_ORDER)} joints, got {len(actual_joint_order)}. "
            f"First joints: {actual_joint_order[:8]}"
        )

    mismatches = [
        (idx, expected, actual)
        for idx, (expected, actual) in enumerate(zip(EXPECTED_JOINT_ORDER, actual_joint_order))
        if expected != actual
    ]
    if mismatches:
        details = ", ".join([f"{i}:{e}!={a}" for i, e, a in mismatches[:6]])
        raise ValueError(
            "X2 symmetry joint order mismatch. "
            f"First mismatches: {details}. "
            "Update symmetry indices or enforce expected joint order."
        )

    print(f"[X2 symmetry] Runtime joint order verified: {actual_joint_order}")
    _VALIDATED_ENVS.add(env_key)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations/actions with original + left-right mirrored samples."""
    _validate_joint_order_once(env.unwrapped)

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry transform to policy observation tensor."""
    obs = obs.clone()
    device = obs.device
    joint_num = 29
    key_body_num = 6

    HISTORY_LEN = 5
    ANG_VEL_DIM = 3
    ROT_TAN_NORM = 6
    VEL_CMD_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    KEY_BODY_POS_DIM = key_body_num * 3

    end_idx = 0
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ROT_TAN_NORM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1, 1, -1, 1], device=device)

    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)

    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_x2_29dof_joints_left_right(obs[:, start_idx:end_idx])

    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_x2_29dof_joints_left_right(obs[:, start_idx:end_idx])

    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_x2_29dof_joints_left_right(obs[:, start_idx:end_idx])

    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + KEY_BODY_POS_DIM
        obs[:, start_idx:end_idx] = _switch_x2_29dof_key_body_pos_left_right(obs[:, start_idx:end_idx])

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry transform to action tensor."""
    actions = actions.clone()
    actions[:] = _switch_x2_29dof_joints_left_right(actions[:])
    return actions


def _switch_x2_29dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right joints and flip signs for roll/yaw-related channels."""
    joint_data_switched = torch.zeros_like(joint_data)

    waist_indices = [_JOINT_INDEX[name] for name in _WAIST_JOINTS]
    joint_data_switched[..., waist_indices] = joint_data[..., waist_indices]

    for left_name, right_name in _LEFT_RIGHT_PAIRS:
        left_idx = _JOINT_INDEX[left_name]
        right_idx = _JOINT_INDEX[right_name]
        joint_data_switched[..., left_idx] = joint_data[..., right_idx]
        joint_data_switched[..., right_idx] = joint_data[..., left_idx]

    roll_indices = [_JOINT_INDEX[name] for name in _ROLL_SIGN_JOINTS]
    yaw_indices = [_JOINT_INDEX[name] for name in _YAW_SIGN_JOINTS]
    waist_sign_indices = [_JOINT_INDEX[name] for name in _WAIST_SIGN_JOINTS]

    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0
    joint_data_switched[..., waist_sign_indices] *= -1.0

    return joint_data_switched


def _switch_x2_29dof_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Swap left/right key-body positions and mirror lateral axis."""
    key_body_pos_switched = key_body_pos.clone()
    num_key_bodies = key_body_pos.shape[-1] // 3

    for i in range(num_key_bodies // 2):
        left_idx = i * 2
        right_idx = i * 2 + 1

        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
            ..., right_idx * 3 : right_idx * 3 + 3
        ]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
            ..., left_idx * 3 : left_idx * 3 + 3
        ]

        key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

    return key_body_pos_switched
