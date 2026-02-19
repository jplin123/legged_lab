"""Configuration for Agibot robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR


@configclass
class AgibotArticulationCfg(ArticulationCfg):
    """Configuration for Agibot articulations."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


@configclass
class AgibotUrdfFileCfg(sim_utils.UrdfFileCfg):
    activate_contact_sensors: bool = True
    fix_base: bool = False
    merge_fixed_joints: bool = False
    make_instanceable: bool = False
    self_collision: bool = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
    )


AGIBOT_X2_ULTRA_CFG = AgibotArticulationCfg(
    spawn=AgibotUrdfFileCfg(
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Agibot/x2/x2_ultra_simple_collision.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            # Locomotion defaults from X2 CPGWalk/CPGRun v2 config.
            "left_hip_pitch_joint": -0.2480,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.5303,
            "left_ankle_pitch_joint": -0.2823,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.2480,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.5303,
            "right_ankle_pitch_joint": -0.2823,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            # Keep non-locomotion joints in a neutral pose.
            ".*_elbow_joint": 0.0,
            ".*_wrist_.*_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            # "head_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "x2_hip_yaw_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=118.0,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_hip_pitch_.*": 120.0,
                ".*_hip_yaw_.*": 120.0,
                "waist_yaw_joint": 160.0,
            },
            damping={
                ".*_hip_pitch_.*": 5.0,
                ".*_hip_yaw_.*": 5.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "x2_hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=118.0,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_hip_roll_.*": 120.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 5.0,
                ".*_knee_.*": 5.0,
            },
            armature=0.01,
        ),
        "x2_ankle_shoulder_elbow_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=48.0,
            velocity_limit_sim=13.0,
            stiffness={
                ".*_ankle_pitch_.*": 40.0,
                ".*_ankle_roll_.*": 30.0,
                "waist_pitch_joint": 80.0,
                "waist_roll_joint": 80.0,
                "left_shoulder_pitch_joint": 80.0,
                "right_shoulder_pitch_joint": 80.0,
                ".*_shoulder_roll_.*": 40.0,
                ".*_shoulder_yaw_.*": 40.0,
                ".*_elbow_.*": 40.0,
            },
            damping={
                ".*_ankle_pitch_.*": 3.0,
                ".*_ankle_roll_.*": 2.0,
                "waist_pitch_joint": 5.0,
                "waist_roll_joint": 5.0,
                "left_shoulder_pitch_joint": 4.0,
                "right_shoulder_pitch_joint": 4.0,
                ".*_shoulder_roll_.*": 1.0,
                ".*_shoulder_yaw_.*": 1.0,
                ".*_elbow_.*": 1.0,
            },
            armature=0.01,
        ),
        "x2_wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll.*", ".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=24.0,
            velocity_limit_sim=22.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)
