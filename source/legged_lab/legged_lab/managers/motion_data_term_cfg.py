from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

@configclass 
class MotionDataTermCfg:
    """
    Configuration for the motion data term in the motion data manager.
    """
    
    weight: float = 1.0
    """Weight of this term in the motion data manager."""
    
    motion_data_dir: str = MISSING
    """Directory containing motion data files.
    
    Only supports reading .pkl files from this directory.
    """
    
    motion_data_weights: dict[str, float] = MISSING
    """Weights for the motion data in this term."""

    motion_style_groups: dict[str, list[str]] | None = None
    """Optional motion-name groups for speed-conditioned sampling, e.g. {"walk": [...], "run": [...]}."""

    command_name: str | None = None
    """Command used for style-conditioned sampling."""

    walk_end_speed: float = 1.2
    """Forward speed at or below which style sampling is fully in walk mode."""

    run_start_speed: float = 1.8
    """Forward speed at or above which style sampling is fully in run mode."""

    backward_threshold: float = 0.1
    """Forward command below `-backward_threshold` is treated as backward locomotion."""

    side_step_threshold: float = 0.2
    """Lateral command magnitude above this threshold is treated as side stepping."""

    turn_threshold: float = 0.5
    """Yaw command magnitude above this threshold is treated as turning."""

    transition_band_enabled: bool = True
    """Whether to prefer a dedicated transition motion group inside the walk/run transition band."""
    
