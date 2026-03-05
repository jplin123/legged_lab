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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.string import resolve_matching_names

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


def _to_builtin(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return str(value)


def _format_value(value):
    if isinstance(value, float):
        return float(f"{value:.3g}")
    if isinstance(value, list):
        return [_format_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _format_value(v) for k, v in value.items()}
    return value


def _export_deploy_yaml(env, env_cfg, output_file: str):
    """Export deploy configuration aligned with Unitree RL Lab schema."""
    robot: Articulation = env.scene["robot"]
    asset_joint_names = list(robot.data.joint_names)
    joint_sdk_names = getattr(env_cfg.scene.robot, "joint_sdk_names", None)
    if not joint_sdk_names:
        joint_sdk_names = list(asset_joint_names)
        joint_ids_map = list(range(len(joint_sdk_names)))
    else:
        joint_ids_map, _ = resolve_matching_names(asset_joint_names, joint_sdk_names, preserve_order=True)

    stiffness = [0.0] * len(joint_sdk_names)
    damping = [0.0] * len(joint_sdk_names)
    default_stiffness = robot.data.default_joint_stiffness[0].detach().cpu().tolist()
    default_damping = robot.data.default_joint_damping[0].detach().cpu().tolist()
    for asset_idx, sdk_idx in enumerate(joint_ids_map):
        stiffness[sdk_idx] = float(default_stiffness[asset_idx])
        damping[sdk_idx] = float(default_damping[asset_idx])

    output_cfg = OrderedDict()
    output_cfg["joint_ids_map"] = [int(x) for x in joint_ids_map]
    output_cfg["joint_names_asset_order"] = asset_joint_names
    output_cfg["joint_names_sdk_order"] = list(joint_sdk_names)
    output_cfg["joint_name_map"] = [
        OrderedDict(
            {
                "asset_idx": int(asset_idx),
                "asset_name": asset_joint_names[asset_idx],
                "sdk_idx": int(sdk_idx),
                "sdk_name": joint_sdk_names[sdk_idx],
            }
        )
        for asset_idx, sdk_idx in enumerate(joint_ids_map)
    ]
    output_cfg["step_dt"] = float(env.step_dt)
    output_cfg["stiffness"] = stiffness
    output_cfg["damping"] = damping
    output_cfg["default_joint_pos"] = robot.data.default_joint_pos[0].detach().cpu().tolist()

    # Commands
    commands = {}
    if hasattr(env_cfg.commands, "base_velocity"):
        commands["base_velocity"] = {}
        if hasattr(env_cfg.commands.base_velocity, "limit_ranges"):
            ranges = env_cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env_cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        commands["base_velocity"]["ranges"] = ranges
    output_cfg["commands"] = commands

    # Actions
    actions = {}
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    for _, action_term in action_terms:
        action_name = action_term.__class__.__name__
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, (float, int)):
            term_cfg.scale = [float(term_cfg.scale) for _ in range(action_term.action_dim)]
        else:
            term_cfg.scale = action_term._scale[0].detach().cpu().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        term_cfg = term_cfg.to_dict()
        for key in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            term_cfg.pop(key, None)
        if action_term._joint_ids == slice(None):
            term_cfg["joint_ids"] = None
        else:
            term_cfg["joint_ids"] = _to_builtin(action_term._joint_ids)
        if hasattr(action_term, "_joint_names"):
            term_cfg["resolved_joint_names"] = list(action_term._joint_names)
        actions[action_name] = term_cfg
    output_cfg["actions"] = actions

    # Observations
    observations = {}
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]  # type: ignore[attr-defined]
    obs_term_layout = []
    running_offset = 0
    for obs_name, obs_cfg in zip(obs_names, obs_cfgs):
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        term_dim = int(obs_dims[1])
        if term_cfg.scale is not None:
            scale = _to_builtin(term_cfg.scale)
            if isinstance(scale, (float, int)):
                term_cfg.scale = [float(scale) for _ in range(term_dim)]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(term_dim)]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        history_length = int(term_cfg.history_length)
        flatten_history = bool(term_cfg.flatten_history_dim)
        slice_start = running_offset
        slice_end = running_offset + term_dim
        running_offset = slice_end
        obs_term_layout.append(
            OrderedDict(
                {
                    "name": obs_name,
                    "slice": [slice_start, slice_end],
                    "term_dim": term_dim,
                    "history_length": history_length,
                    "flatten_history_dim": flatten_history,
                    "layout": (
                        "term-major, history-oldest-to-newest (xxxxx yyyyy style)"
                        if flatten_history
                        else "term-major, explicit history dimension"
                    ),
                }
            )
        )

        term_cfg = term_cfg.to_dict()
        for key in ["func", "modifiers", "noise", "flatten_history_dim"]:
            term_cfg.pop(key, None)
        observations[obs_name] = term_cfg
    output_cfg["observations"] = observations
    output_cfg["observation_layout"] = OrderedDict(
        {
            "policy_concatenate_terms": bool(env.observation_manager.group_obs_concatenate["policy"]),
            "policy_concat_order": "term-major (not interleaved xyxyxy)",
            "policy_terms": obs_term_layout,
        }
    )

    output_cfg = _format_value(output_cfg)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(output_cfg, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Deploy config exported to: {output_file}")
    print("[INFO] Policy observation flatten order: term-major blocks (xxxxx yyyyy), not interleaved (xyxyxy).")
    for item in obs_term_layout:
        print(
            f"[INFO]   {item['name']}: slice={item['slice']} dim={item['term_dim']} "
            f"history={item['history_length']} flatten={item['flatten_history_dim']}"
        )


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
