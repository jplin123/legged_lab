# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import re
import time
import torch
import yaml
from collections import OrderedDict

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import Articulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)


def _represent_ordereddict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, _represent_ordereddict)


def _export_deploy_yaml(env, env_cfg, output_file: str):
    """Export deploy configuration for the current run."""
    output_cfg = OrderedDict()

    obs_term_names = env.observation_manager.active_terms["policy"]
    obs_term_cfg = env.observation_manager._group_obs_term_cfgs["policy"]  # type: ignore[attr-defined]
    output_cfg["obs_names"] = obs_term_names
    output_cfg["obs_cfg"] = OrderedDict()
    for term_name, term_cfg in zip(obs_term_names, obs_term_cfg):
        output_cfg["obs_cfg"][term_name] = OrderedDict(
            [
                ("clip", term_cfg.clip if term_cfg.clip else [0.0]),
                ("scale", float(term_cfg.scale) if term_cfg.scale else 1.0),
                ("history_length", term_cfg.history_length),
            ]
        )

    robot: Articulation = env.scene["robot"]
    action_cfg = env_cfg.actions.joint_pos
    action_term: JointPositionAction = env.action_manager.get_term("joint_pos")

    joint_names = robot.joint_names
    for i, jnt_name in enumerate(joint_names):
        if jnt_name != action_term._joint_names[i]:  # type: ignore[attr-defined]
            raise ValueError(f"Joint name mismatch: {jnt_name} != {action_term._joint_names[i]}")  # type: ignore[attr-defined]

    output_cfg["joint_names"] = joint_names
    output_cfg["action_cfg"] = OrderedDict()

    for i, jnt_name in enumerate(joint_names):
        if action_cfg.clip is not None:
            clip = action_term._clip[0, i, :].cpu().tolist()  # type: ignore[attr-defined]
        else:
            clip = [0.0]

        if isinstance(action_cfg.scale, (float, int)):
            scale = float(action_cfg.scale)
        elif isinstance(action_cfg.scale, dict):
            scale = action_term._scale[0, i].item()  # type: ignore[attr-defined]
        else:
            scale = 1.0

        found = False
        kp = 0.0
        kd = 0.0
        for value in env_cfg.scene.robot.actuators.values():
            for expr in value.joint_names_expr:
                if re.fullmatch(expr, jnt_name):
                    found = True
                    if isinstance(value.stiffness, float):
                        kp = value.stiffness
                    elif isinstance(value.stiffness, dict):
                        for k, v in value.stiffness.items():
                            if re.fullmatch(k, jnt_name):
                                kp = v
                                break
                    else:
                        raise ValueError(f"Unsupported stiffness type for joint {jnt_name}: {type(value.stiffness)}")

                    if isinstance(value.damping, float):
                        kd = value.damping
                    elif isinstance(value.damping, dict):
                        for k, v in value.damping.items():
                            if re.fullmatch(k, jnt_name):
                                kd = v
                                break
                    else:
                        raise ValueError(f"Unsupported damping type for joint {jnt_name}: {type(value.damping)}")
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Joint {jnt_name} not found in robot actuator config.")

        default_pos = robot.data.default_joint_pos[0, i].item()
        output_cfg["action_cfg"][jnt_name] = OrderedDict(
            [("clip", clip), ("scale", scale), ("kp", kp), ("kd", kd), ("default_pos", default_pos)]
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(output_cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] Deploy config exported to: {output_file}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # export deploy config
    deploy_yaml_path = os.path.join(log_dir, "params", "deploy.yaml")
    try:
        _export_deploy_yaml(env.unwrapped, env_cfg, deploy_yaml_path)
    except Exception as exc:
        print(f"[WARN] Failed to export deploy config: {exc}")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner
        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
