"""
Training Script for Workload Placement Agent
Optimized for AMD Cloud with ROCm

Usage:
    python train.py --data-dir ./data --epochs 100 --batch-size 64
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RL Libraries
try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import our modules
from simulator import ClusterSchedulerEnv, Machine, Job, BaselineScheduler
from data_loader import (
    load_alibaba_trace_sample, 
    preprocess_alibaba_data,
    load_processed_data
)


class TrainingMonitorCallback(BaseCallback):
    """Custom callback to monitor training progress."""

    def __init__(self, log_dir: str, save_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_costs = []
        self.episode_energies = []

    def _on_step(self) -> bool:
        # Log every N steps
        if self.n_calls % self.save_freq == 0:
            # Get current stats from environment
            info = self.locals.get("infos", [{}])[0]

            if info:
                self.episode_rewards.append(info.get("episode", {}).get("r", 0))
                self.episode_lengths.append(info.get("episode", {}).get("l", 0))

            if self.verbose > 0 and self.n_calls % (self.save_freq * 5) == 0:
                print(f"Step {self.n_calls}: Training in progress...")

        return True

    def _on_training_end(self):
        # Save final statistics
        stats = {
            "total_steps": self.n_calls,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_length": np.mean(self.episode_lengths) if self.episode_lengths else 0
        }

        with open(self.log_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)


def create_env_from_data(data_dir: str, max_steps: int = 1000, render_mode: str = None):
    """Create environment from preprocessed data."""
    try:
        data = load_processed_data(data_dir)
    except FileNotFoundError:
        print("Processed data not found. Creating synthetic data...")
        from data_loader import create_synthetic_machines, create_synthetic_jobs, preprocess_alibaba_data

        machine_meta = create_synthetic_machines(10)
        batch_task = create_synthetic_jobs(1000)
        batch_instance = create_synthetic_instances(batch_task)

        data = preprocess_alibaba_data(machine_meta, batch_task, batch_instance, data_dir)

    env = ClusterSchedulerEnv(
        machines=data["machines"],
        jobs=data["jobs"],
        max_steps=max_steps,
        render_mode=render_mode
    )

    return env


def make_env(data_dir: str, max_steps: int = 1000, seed: int = 0):
    """Factory function for vectorized environments."""
    def _init():
        env = create_env_from_data(data_dir, max_steps)
        env.reset(seed=seed)
        env = Monitor(env)
        return env
    return _init


def train_agent(
    data_dir: str = "./data",
    output_dir: str = "./models",
    algorithm: str = "PPO",
    total_timesteps: int = 500000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    max_steps_per_episode: int = 1000,
    seed: int = 42,
    device: str = "auto"
):
    """
    Train the workload placement agent.

    Args:
        data_dir: Directory with preprocessed data
        output_dir: Directory to save models
        algorithm: RL algorithm (PPO, DQN, A2C)
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        n_steps: Steps per update (PPO)
        batch_size: Minibatch size
        n_epochs: Number of epochs per update (PPO)
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        max_steps_per_episode: Max steps per episode
        seed: Random seed
        device: Device for training (cpu, cuda, auto)
    """

    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_dir = output_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create vectorized environment
    print(f"\n{'='*60}")
    print(f"Creating {n_envs} parallel environments...")
    print(f"{'='*60}")

    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(data_dir, max_steps_per_episode, seed + i) 
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(data_dir, max_steps_per_episode, seed)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(data_dir, max_steps_per_episode, seed + 999)])

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best_model"),
        log_path=str(log_dir),
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    monitor_callback = TrainingMonitorCallback(
        log_dir=str(log_dir),
        save_freq=5000
    )

    callbacks = [eval_callback, monitor_callback]

    # Initialize RL algorithm
    print(f"\nInitializing {algorithm} agent...")
    print(f"  Device: {device}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")

    if algorithm.upper() == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device=device,
            seed=seed
        )
    elif algorithm.upper() == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=batch_size,
            gamma=gamma,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device=device,
            seed=seed
        )
    elif algorithm.upper() == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device=device,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Train
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    training_time = time.time() - start_time

    # Save final model
    final_model_path = output_path / f"{algorithm.lower()}_final_model"
    model.save(str(final_model_path))
    print(f"\n✅ Model saved to {final_model_path}")

    # Save training config
    config = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "max_steps_per_episode": max_steps_per_episode,
        "seed": seed,
        "device": str(device),
        "training_time_seconds": training_time,
        "training_time_formatted": f"{training_time/3600:.2f} hours"
    }

    with open(output_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"  Total time: {training_time/60:.1f} minutes")
    print(f"  Model saved: {final_model_path}")
    print(f"  Best model: {output_path / 'best_model'}")

    return model, config


def evaluate_baselines(data_dir: str = "./data", n_episodes: int = 10):
    """Evaluate baseline schedulers for comparison."""
    print(f"\n{'='*60}")
    print(f"Evaluating Baseline Schedulers")
    print(f"{'='*60}")

    env = create_env_from_data(data_dir, max_steps=1000)

    baselines = {
        "First Fit": BaselineScheduler.first_fit,
        "Best Fit": BaselineScheduler.best_fit,
        "Least Loaded": BaselineScheduler.least_loaded,
        "Cheapest Fit": BaselineScheduler.cheapest_fit
    }

    results = {}

    for name, scheduler_fn in baselines.items():
        print(f"\nEvaluating {name}...")
        episode_rewards = []
        episode_costs = []
        episode_energies = []
        episode_sla_violations = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            total_reward = 0

            while not done:
                action = scheduler_fn(env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_costs.append(info["total_cost"])
            episode_energies.append(info["total_energy_kwh"])
            episode_sla_violations.append(info["sla_violations"])

        results[name] = {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "avg_cost": np.mean(episode_costs),
            "avg_energy": np.mean(episode_energies),
            "avg_sla_violations": np.mean(episode_sla_violations)
        }

        print(f"  Avg Reward: {results[name]['avg_reward']:.2f} ± {results[name]['std_reward']:.2f}")
        print(f"  Avg Cost: ${results[name]['avg_cost']:.2f}")
        print(f"  Avg Energy: {results[name]['avg_energy']:.3f} kWh")
        print(f"  Avg SLA Violations: {results[name]['avg_sla_violations']:.1f}")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv("baseline_results.csv")
    print(f"\n✅ Baseline results saved to baseline_results.csv")

    return results


def plot_training_results(log_dir: str = "./models/logs"):
    """Plot training curves from tensorboard logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        log_path = Path(log_dir)
        event_files = list(log_path.glob("events.out.tfevents.*"))

        if not event_files:
            print("No tensorboard event files found.")
            return

        ea = event_accumulator.EventAccumulator(str(event_files[0]))
        ea.Reload()

        # Plot rewards
        if "rollout/ep_rew_mean" in ea.Tags()["scalars"]:
            rewards = ea.Scalars("rollout/ep_rew_mean")
            steps = [x.step for x in rewards]
            values = [x.value for x in rewards]

            plt.figure(figsize=(10, 6))
            plt.plot(steps, values)
            plt.xlabel("Training Steps")
            plt.ylabel("Mean Episode Reward")
            plt.title("Training Progress")
            plt.grid(True)
            plt.savefig("training_rewards.png", dpi=150, bbox_inches="tight")
            print("✅ Training plot saved to training_rewards.png")

    except Exception as e:
        print(f"Could not plot results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Workload Placement Agent")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN", "A2C"])
    parser.add_argument("--total-timesteps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-baselines", action="store_true", help="Evaluate baseline schedulers")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate")

    args = parser.parse_args()

    # Check for AMD GPU
    if args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA/ROCm available: {torch.cuda.get_device_name(0)}")
                args.device = "cuda"
            else:
                print("⚠️ No GPU detected, using CPU")
                args.device = "cpu"
        except:
            args.device = "cpu"

    print(f"\n{'='*60}")
    print(f"Workload Placement Agent - Training")
    print(f"{'='*60}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Device: {args.device}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")

    # Evaluate baselines first
    if args.eval_baselines:
        baseline_results = evaluate_baselines(args.data_dir)

    # Train agent
    if not args.skip_training:
        model, config = train_agent(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            device=args.device,
            seed=args.seed
        )

        # Plot results
        plot_training_results()

    print(f"\n{'='*60}")
    print(f"All tasks completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
