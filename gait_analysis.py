"""Deterministic headless dynamics evaluation for DreamWaQ/Boying.

The standard legged_gym arguments select the run/checkpoint.  Evaluation
settings are environment variables so this script remains compatible with
legged_gym's strict argument parser:

    GAIT_COMMAND=0.6,0,0 GAIT_WARMUP_STEPS=150 GAIT_STEPS=700 \
    GAIT_OUTPUT_DIR=/tmp/dreamwaq_eval \
    python gait_analysis.py --task=boying --headless \
        --load_run Jul28_22-17-43_ --checkpoint 30000
"""
import sys
sys.path.insert(0, '/home/wkl/DreamWaQ/legged_gym')
sys.path.insert(0, '/home/wkl/DreamWaQ/rsl_rl-1.0.2')
sys.path.insert(0, '/home/wkl/DreamWaQ/isaacgym/python')

import isaacgym  # must be before torch
import json
import os
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

COMMAND = np.asarray(
    [float(value) for value in os.getenv("GAIT_COMMAND", "0.6,0,0").split(",")],
    dtype=np.float32,
)
WARMUP_STEPS = int(os.getenv("GAIT_WARMUP_STEPS", "150"))
RECORD_STEPS = int(os.getenv("GAIT_STEPS", "700"))
OUTPUT_DIR = os.getenv("GAIT_OUTPUT_DIR", "/tmp/dreamwaq_eval")
TERRAIN = os.getenv("GAIT_TERRAIN", "plane")
ROBOT_INDEX = int(os.getenv("GAIT_ROBOT_INDEX", "0"))
TERRAIN_LEVEL = int(os.getenv("GAIT_TERRAIN_LEVEL", "5"))

if COMMAND.shape != (3,):
    raise ValueError("GAIT_COMMAND must contain vx,vy,yaw_rate")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def _safe_corr(left, right):
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return None
    return float(np.corrcoef(left, right)[0, 1])

def collect_gait(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # A plane isolates policy/checkpoint dynamics. In rough mode one robot is
    # created per curriculum terrain column so GAIT_ROBOT_INDEX deterministically
    # selects the terrain family (0-1 slopes, 2-3 rough slopes, 4-17 stairs,
    # 18-19 discrete obstacles).
    if TERRAIN == "plane":
        env_cfg.env.num_envs = 1
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
    elif TERRAIN == "rough":
        env_cfg.env.num_envs = env_cfg.terrain.num_cols
        env_cfg.terrain.mesh_type = "trimesh"
        env_cfg.terrain.curriculum = True
    else:
        raise ValueError("GAIT_TERRAIN must be 'plane' or 'rough'")
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_Kp_factor = False
    env_cfg.domain_rand.randomize_Kd_factor = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not 0 <= ROBOT_INDEX < env.num_envs:
        raise ValueError(
            f"GAIT_ROBOT_INDEX={ROBOT_INDEX} outside [0, {env.num_envs})"
        )
    if TERRAIN == "rough":
        level = int(np.clip(TERRAIN_LEVEL, 0, env.max_terrain_level - 1))
        env.terrain_levels[:] = level
        env.env_origins[:] = env.terrain_origins[
            env.terrain_levels, env.terrain_types
        ]
        env.reset_idx(torch.arange(env.num_envs, device=env.device))

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    command_tensor = torch.as_tensor(COMMAND, device=env.device)

    def force_command_and_observe():
        env.commands[:, :3] = command_tensor
        env.compute_observations()
        return env.get_observations()

    obs, obs_hist = force_command_and_observe()
    for _ in range(WARMUP_STEPS):
        env.commands[:, :3] = command_tensor
        with torch.inference_mode():
            actions = policy(obs.detach(), obs_hist.detach())
        obs, _, _, obs_hist, _, _, _ = env.step(actions.detach())

    shape = (RECORD_STEPS,)
    data = {
        "contact_force": np.zeros((RECORD_STEPS, 4, 3)),
        "foot_position": np.zeros((RECORD_STEPS, 4, 3)),
        "foot_velocity": np.zeros((RECORD_STEPS, 4, 3)),
        "base_lin_vel": np.zeros((RECORD_STEPS, 3)),
        "base_ang_vel": np.zeros((RECORD_STEPS, 3)),
        "projected_gravity": np.zeros((RECORD_STEPS, 3)),
        "base_height": np.zeros(shape),
        "dof_pos": np.zeros((RECORD_STEPS, 12)),
        "dof_vel": np.zeros((RECORD_STEPS, 12)),
        "dof_acc": np.zeros((RECORD_STEPS, 12)),
        "torque": np.zeros((RECORD_STEPS, 12)),
        "action": np.zeros((RECORD_STEPS, 12)),
        "command": np.repeat(COMMAND[None, :], RECORD_STEPS, axis=0),
        "done": np.zeros(shape, dtype=np.bool_),
    }

    robot_idx = ROBOT_INDEX
    previous_dof_vel = env.dof_vel[robot_idx].detach().cpu().numpy().copy()
    for i in range(RECORD_STEPS):
        env.commands[:, :3] = command_tensor
        with torch.inference_mode():
            actions = policy(obs.detach(), obs_hist.detach())
        obs, _, _, obs_hist, _, done, _ = env.step(actions.detach())

        dof_vel = env.dof_vel[robot_idx].detach().cpu().numpy()
        data["contact_force"][i] = (
            env.contact_forces[robot_idx, env.feet_indices].detach().cpu().numpy()
        )
        data["foot_position"][i] = (
            env.foot_positions[robot_idx].detach().cpu().numpy()
        )
        data["foot_velocity"][i] = (
            env.foot_velocities[robot_idx].detach().cpu().numpy()
        )
        data["base_lin_vel"][i] = env.base_lin_vel[robot_idx].detach().cpu().numpy()
        data["base_ang_vel"][i] = env.base_ang_vel[robot_idx].detach().cpu().numpy()
        data["projected_gravity"][i] = (
            env.projected_gravity[robot_idx].detach().cpu().numpy()
        )
        data["base_height"][i] = float(env.root_states[robot_idx, 2].item())
        data["dof_pos"][i] = env.dof_pos[robot_idx].detach().cpu().numpy()
        data["dof_vel"][i] = dof_vel
        data["dof_acc"][i] = (dof_vel - previous_dof_vel) / env.dt
        data["torque"][i] = env.torques[robot_idx].detach().cpu().numpy()
        data["action"][i] = actions[robot_idx].detach().cpu().numpy()
        data["done"][i] = bool(done[robot_idx].item())
        previous_dof_vel = dof_vel.copy()

    ckpt = args.checkpoint
    run = args.load_run
    command_name = "_".join(f"{value:g}" for value in COMMAND)
    terrain_name = (
        "plane"
        if TERRAIN == "plane"
        else f"rough_type{ROBOT_INDEX}_level{TERRAIN_LEVEL}"
    )
    out_prefix = os.path.join(
        OUTPUT_DIR,
        f"gait_{run}_ckpt{ckpt}_{terrain_name}_cmd_{command_name}",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(f"{out_prefix}.npz", **data)

    contact_norm = np.linalg.norm(data["contact_force"], axis=2)
    contact = data["contact_force"][:, :, 2] > 10.0
    power = np.abs(data["torque"] * data["dof_vel"])
    torque_limits = env.torque_limits.detach().cpu().numpy()
    velocity_limits = env.dof_vel_limits.detach().cpu().numpy()
    lower_limits = env.dof_pos_limits[:, 0].detach().cpu().numpy()
    upper_limits = env.dof_pos_limits[:, 1].detach().cpu().numpy()
    position_margin = np.minimum(
        data["dof_pos"] - lower_limits, upper_limits - data["dof_pos"]
    )
    planar_error = data["base_lin_vel"][:, :2] - COMMAND[:2]
    yaw_error = data["base_ang_vel"][:, 2] - COMMAND[2]
    stance_slip = np.linalg.norm(data["foot_velocity"][:, :, :2], axis=2)[contact]

    summary = {
        "run": run,
        "checkpoint": int(ckpt),
        "command": COMMAND.tolist(),
        "terrain": TERRAIN,
        "terrain_level": (
            int(env.terrain_levels[robot_idx].item())
            if TERRAIN == "rough" else None
        ),
        "terrain_type": (
            int(env.terrain_types[robot_idx].item())
            if TERRAIN == "rough" else None
        ),
        "robot_index": robot_idx,
        "warmup_steps": WARMUP_STEPS,
        "record_steps": RECORD_STEPS,
        "dt": float(env.dt),
        "reset_count": int(data["done"].sum()),
        "tracking_planar_error": _summary(planar_error),
        "tracking_yaw_error": _summary(yaw_error),
        "base_height": _summary(data["base_height"]),
        "base_vertical_velocity": _summary(data["base_lin_vel"][:, 2]),
        "base_roll_pitch_rate": _summary(data["base_ang_vel"][:, :2]),
        "projected_gravity_xy": _summary(data["projected_gravity"][:, :2]),
        "action_rate": _summary(np.diff(data["action"], axis=0) / env.dt),
        "dof_acceleration": _summary(data["dof_acc"][1:]),
        "joint_power": _summary(power),
        "total_absolute_power": _summary(power.sum(axis=1)),
        "torque_utilization": _summary(data["torque"] / torque_limits),
        "velocity_utilization": _summary(data["dof_vel"] / velocity_limits),
        "minimum_position_margin_rad": float(np.min(position_margin)),
        "position_limit_violation_count": int(np.sum(position_margin < 0.0)),
        "contact_ratio_per_foot": contact.mean(axis=0).tolist(),
        "contact_force_per_foot": [
            _summary(contact_norm[:, foot]) for foot in range(contact.shape[1])
        ],
        "foot_height_per_foot": [
            _summary(data["foot_position"][:, foot, 2])
            for foot in range(contact.shape[1])
        ],
        "stance_foot_slip_speed": (
            _summary(stance_slip) if stance_slip.size else None
        ),
        "contact_correlation": {
            "FL_RR": _safe_corr(contact[:, 0], contact[:, 3]),
            "FR_RL": _safe_corr(contact[:, 1], contact[:, 2]),
            "FL_FR": _safe_corr(contact[:, 0], contact[:, 1]),
            "RL_RR": _safe_corr(contact[:, 2], contact[:, 3]),
        },
    }
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    print(f"[DONE] Saved {RECORD_STEPS} steps -> {out_prefix}.npz")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    args = get_args()
    collect_gait(args)
