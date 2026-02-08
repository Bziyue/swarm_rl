"""
Script to profile an RL agent (skrl + Isaac Lab) using torch.profiler.
Usage: python profile_agent.py --task <TASK_NAME> --checkpoint <PATH>
"""

import argparse
import sys

# --- 1. Launch Isaac Sim (Must be done before other imports) ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Profile an RL agent using torch.profiler.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="RL algorithm.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--log_dir", type=str, default="./outputs/torch_profile", help="Output directory for torch profile logs.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Late Imports ---
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import skrl
from skrl.utils.runner.torch import Runner
from packaging import version
# Safer version check
if version.parse(skrl.__version__) < version.parse("2.0.0"):
    raise ImportError("skrl>=2.0.0 is required.")

import swarm_rl.envs  # noqa: F401

def main():
    """Main profiling execution."""
    
    # --- Configuration ---
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # --- Setup ---
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Disable training overhead
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(env, experiment_cfg, verbose=True)
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)
    runner.agent.enable_models_training_mode(False)

    obs, _ = env.reset()
    states = env.state()

    # --- Helper Function to Avoid Code Duplication ---
    def run_step(current_obs, current_states):
        """Executes one single step of Policy + Env."""
        # A. Policy Inference
        with record_function("1_Policy_Inference"):
            outputs = runner.agent.act(current_obs, current_states, timestep=0, timesteps=0)
            
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

        # B. Environment Step
        with record_function("2_Env_Step_Total"):
            new_obs, _, _, _, _ = env.step(actions)
            new_states = env.state()
            
        return new_obs, new_states

    # --- 3. Warmup Phase ---
    WARMUP_STEPS = 20
    print(f"[INFO] Warming up GPU ({WARMUP_STEPS} steps)...")
    for _ in range(WARMUP_STEPS):
        with torch.inference_mode():
            obs, states = run_step(obs, states)

    # --- 4. Profiling Phase ---
    PROFILE_STEPS = 10
    print(f"[INFO] Starting Profiling ({PROFILE_STEPS} steps)...")

    # Setup Profiler
    my_schedule = schedule(wait=0, warmup=0, active=PROFILE_STEPS, repeat=1)
    handler = tensorboard_trace_handler(args_cli.log_dir)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True
    ) as prof:
        
        torch.cuda.synchronize()

        for _ in range(PROFILE_STEPS):
            with torch.inference_mode():
                obs, states = run_step(obs, states)
            prof.step()

    print("-" * 60)
    print(f"[SUCCESS] Profiling complete.")
    print(f"Run: tensorboard --logdir={args_cli.log_dir}")
    print("-" * 60)

    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()