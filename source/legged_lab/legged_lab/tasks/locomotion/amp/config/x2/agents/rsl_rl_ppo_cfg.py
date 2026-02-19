from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import G1RslRlOnPolicyRunnerAmpCfg


@configclass
class X2RslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "x2_amp"

    def __post_init__(self):
        super().__post_init__()
        # Start without any symmetry assumptions for X2.
        self.algorithm.symmetry_cfg = None
