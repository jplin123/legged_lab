import os

from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.agibot import AGIBOT_X2_ULTRA_CFG
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1AmpEnvCfg


def _build_motion_weights(motion_dir: str) -> dict[str, float]:
    return {
        os.path.splitext(file_name)[0]: 1.0
        for file_name in sorted(os.listdir(motion_dir))
        if file_name.endswith(".pkl")
    }


@configclass
class X2AmpEnvCfg(G1AmpEnvCfg):
    """Configuration for the X2 AMP environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AGIBOT_X2_ULTRA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # X2-stabilization baseline: start with milder commands/actions than G1.
        self.actions.joint_pos.scale = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-1.57, 1.57)
        self.events.reset_from_ref.params["height_offset"] = 0.15
        # Disable pushes for initial stabilization; re-enable later once gait is stable.
        self.events.push_robot = None

        # Reuse the baseline AMP setup and only swap the motion dataset to X2.
        motion_dir = os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", "x2", "amp", "walk_and_run")
        self.motion_data.motion_dataset.motion_data_dir = motion_dir
        self.motion_data.motion_dataset.motion_data_weights = _build_motion_weights(motion_dir)


@configclass
class X2AmpEnvCfg_PLAY(X2AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.reset_from_ref = None
