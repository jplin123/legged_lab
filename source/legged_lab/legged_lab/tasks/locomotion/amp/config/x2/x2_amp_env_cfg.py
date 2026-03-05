import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.agibot import AGIBOT_X2_ULTRA_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg


KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4


def _build_motion_weights(motion_dir: str) -> dict[str, float]:
    return {
        os.path.splitext(file_name)[0]: 1.0
        for file_name in sorted(os.listdir(motion_dir))
        if file_name.endswith(".pkl")
    }


@configclass
class X2AmpRewards:
    """Reward terms for the X2 AMP MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class X2AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the X2 AMP environment."""

    rewards: X2AmpRewards = X2AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = AGIBOT_X2_ULTRA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        motion_dir = os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", "x2", "amp", "walk_and_run")
        self.motion_data.motion_dataset.motion_data_dir = motion_dir
        self.motion_data.motion_dataset.motion_data_weights = _build_motion_weights(motion_dir)

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.1}

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # Per-joint action scaling for X2.
        # Joints not listed below keep the default scale (0.25).
        self.actions.joint_pos.scale = {
            "left_hip_pitch_joint": 0.35,
            "left_hip_roll_joint": 0.35,
            "left_hip_yaw_joint": 0.35,
            "left_knee_joint": 0.35,
            "left_ankle_pitch_joint": 0.2,
            "left_ankle_roll_joint": 0.02,
            "right_hip_pitch_joint": 0.35,
            "right_hip_roll_joint": 0.35,
            "right_hip_yaw_joint": 0.35,
            "right_knee_joint": 0.35,
            "right_ankle_pitch_joint": 0.2,
            "right_ankle_roll_joint": 0.02,
            "waist_yaw_joint": 0.2,
            "waist_pitch_joint": 0.2,
            "waist_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.2,
            "right_shoulder_yaw_joint": 0.2,
            "left_elbow_joint": 0.35,
            "right_elbow_joint": 0.35,
            "left_wrist_roll_joint": 0.2,
            "right_wrist_roll_joint": 0.2,
            "left_wrist_pitch_joint": 0.2,
            "right_wrist_pitch_joint": 0.2,
            "left_wrist_yaw_joint": 0.2,
            "right_wrist_yaw_joint": 0.2
        }

        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        self.terminations.base_contact = None


@configclass
class X2AmpEnvCfg_PLAY(X2AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        # Keep PLAY command ranges consistent with training for faithful evaluation/deploy export.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.events.reset_from_ref = None
