"""
Evaluation and Demo Script for Workload Placement Agent

Usage:
    python evaluate.py --model-path ./models/PPO_final_model.zip --n-episodes 10
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from simulator import ClusterSchedulerEnv, Machine, Job, BaselineScheduler
from data_loader import load_processed_data, create_synthetic_machines, create_synthetic_jobs, preprocess_alibaba_data


def load_trained_model(model_path: str, algorithm: str = "PPO"):
    """Load a trained RL model."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading {algorithm} model from {model_path}...")

    if algorithm.upper() == "PPO":
        model = PPO.load(str(model_path))
    elif algorithm.upper() == "DQN":
        model = DQN.load(str(model_path))
    elif algorithm.upper() == "A2C":
        model = A2C.load(str(model_path))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"✅ Model loaded successfully")
    return model


def evaluate_agent(model, env: ClusterSchedulerEnv, n_episodes: int = 10, render: bool = False):
    """
    Evaluate trained agent against environment.

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Trained Agent ({n_episodes} episodes)")
    print(f"{'='*60}")

    episode_rewards = []
    episode_lengths = []
    episode_costs = []
    episode_energies = []
    episode_sla_violations = []
    episode_utilizations = []

    all_placements = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1
            done = terminated or truncated

            if render and step_count % 100 == 0:
                env.render()

        # Collect metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_costs.append(info["total_cost"])
        episode_energies.append(info["total_energy_kwh"])
        episode_sla_violations.append(info["sla_violations"])
        episode_utilizations.append(info["avg_cpu_util"])

        # Collect placements
        placements = env.get_placement_summary()
        placements["episode"] = ep
        all_placements.append(placements)

        print(f"  Episode {ep+1}: Reward={total_reward:.2f}, Cost=${info['total_cost']:.2f}, "
              f"Energy={info['total_energy_kwh']:.3f}kWh, SLA_violations={info['sla_violations']}")

    # Aggregate results
    results = {
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_length": np.mean(episode_lengths),
        "avg_cost": np.mean(episode_costs),
        "std_cost": np.std(episode_costs),
        "avg_energy_kwh": np.mean(episode_energies),
        "std_energy": np.std(episode_energies),
        "avg_sla_violations": np.mean(episode_sla_violations),
        "avg_cpu_utilization": np.mean(episode_utilizations),
        "total_episodes": n_episodes
    }

    # Combine all placements
    all_placements_df = pd.concat(all_placements, ignore_index=True) if all_placements else pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Avg Cost: ${results['avg_cost']:.2f} ± ${results['std_cost']:.2f}")
    print(f"  Avg Energy: {results['avg_energy_kwh']:.3f} ± {results['std_energy']:.3f} kWh")
    print(f"  Avg SLA Violations: {results['avg_sla_violations']:.1f}")
    print(f"  Avg CPU Utilization: {results['avg_cpu_utilization']:.1%}")

    return results, all_placements_df


def compare_with_baselines(model, env: ClusterSchedulerEnv, n_episodes: int = 10):
    """Compare trained agent with baseline schedulers."""
    print(f"\n{'='*60}")
    print(f"Comparing Agent vs Baselines")
    print(f"{'='*60}")

    baselines = {
        "RL Agent (Ours)": lambda e: model.predict(e._get_observation(), deterministic=True)[0],
        "First Fit": BaselineScheduler.first_fit,
        "Best Fit": BaselineScheduler.best_fit,
        "Least Loaded": BaselineScheduler.least_loaded,
        "Cheapest Fit": BaselineScheduler.cheapest_fit
    }

    comparison_results = {}

    for name, scheduler_fn in baselines.items():
        print(f"\nEvaluating {name}...")

        episode_rewards = []
        episode_costs = []
        episode_energies = []
        episode_sla_violations = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep + 5000)
            done = False
            total_reward = 0

            while not done:
                if name == "RL Agent (Ours)":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = scheduler_fn(env)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_costs.append(info["total_cost"])
            episode_energies.append(info["total_energy_kwh"])
            episode_sla_violations.append(info["sla_violations"])

        comparison_results[name] = {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "avg_cost": np.mean(episode_costs),
            "std_cost": np.std(episode_costs),
            "avg_energy": np.mean(episode_energies),
            "avg_sla": np.mean(episode_sla_violations)
        }

    # Create comparison DataFrame
    df = pd.DataFrame(comparison_results).T

    print(f"\n{'='*60}")
    print(f"Comparison Results")
    print(f"{'='*60}")
    print(df.to_string())

    # Calculate improvements
    if "RL Agent (Ours)" in comparison_results:
        our_cost = comparison_results["RL Agent (Ours)"]["avg_cost"]
        our_energy = comparison_results["RL Agent (Ours)"]["avg_energy"]
        our_sla = comparison_results["RL Agent (Ours)"]["avg_sla"]

        print(f"\n{'='*60}")
        print(f"Improvements over Best Baseline")
        print(f"{'='*60}")

        baseline_costs = {k: v["avg_cost"] for k, v in comparison_results.items() if k != "RL Agent (Ours)"}
        best_baseline_cost = min(baseline_costs.values())
        cost_improvement = (best_baseline_cost - our_cost) / best_baseline_cost * 100

        baseline_energies = {k: v["avg_energy"] for k, v in comparison_results.items() if k != "RL Agent (Ours)"}
        best_baseline_energy = min(baseline_energies.values())
        energy_improvement = (best_baseline_energy - our_energy) / best_baseline_energy * 100

        print(f"  Cost reduction: {cost_improvement:.1f}% vs best baseline")
        print(f"  Energy reduction: {energy_improvement:.1f}% vs best baseline")
        print(f"  SLA violations: {our_sla:.1f} (lower is better)")

    return comparison_results, df


def visualize_results(comparison_df: pd.DataFrame, output_dir: str = "./results"):
    """Create visualization of comparison results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 10)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cost comparison
    ax = axes[0, 0]
    costs = comparison_df["avg_cost"]
    colors = ["#e74c3c" if "RL Agent" in name else "#3498db" for name in costs.index]
    bars = ax.bar(range(len(costs)), costs.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(costs)))
    ax.set_xticklabels(costs.index, rotation=45, ha="right")
    ax.set_ylabel("Average Cost ($)")
    ax.set_title("Cost Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontsize=9)

    # 2. Energy comparison
    ax = axes[0, 1]
    energies = comparison_df["avg_energy"]
    colors = ["#e74c3c" if "RL Agent" in name else "#2ecc71" for name in energies.index]
    bars = ax.bar(range(len(energies)), energies.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(energies)))
    ax.set_xticklabels(energies.index, rotation=45, ha="right")
    ax.set_ylabel("Average Energy (kWh)")
    ax.set_title("Energy Consumption Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Reward comparison
    ax = axes[1, 0]
    rewards = comparison_df["avg_reward"]
    colors = ["#e74c3c" if "RL Agent" in name else "#9b59b6" for name in rewards.index]
    bars = ax.bar(range(len(rewards)), rewards.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(rewards)))
    ax.set_xticklabels(rewards.index, rotation=45, ha="right")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward Comparison (Higher is Better)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 4. SLA violations
    ax = axes[1, 1]
    slas = comparison_df["avg_sla"]
    colors = ["#e74c3c" if "RL Agent" in name else "#f39c12" for name in slas.index]
    bars = ax.bar(range(len(slas)), slas.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(slas)))
    ax.set_xticklabels(slas.index, rotation=45, ha="right")
    ax.set_ylabel("Average SLA Violations")
    ax.set_title("SLA Violations (Lower is Better)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "comparison_results.png", dpi=150, bbox_inches="tight")
    print(f"✅ Comparison plot saved to {output_path / 'comparison_results.png'}")

    # Create a summary table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for idx, row in comparison_df.iterrows():
        table_data.append([
            idx,
            f"${row['avg_cost']:.2f} ± ${row['std_cost']:.2f}",
            f"{row['avg_energy']:.3f} kWh",
            f"{row['avg_reward']:.2f}",
            f"{row['avg_sla']:.1f}"
        ])

    table = ax.table(cellText=table_data,
                     colLabels=["Scheduler", "Avg Cost", "Avg Energy", "Avg Reward", "SLA Violations"],
                     cellLoc='center',
                     loc='center',
                     colColours=['#3498db']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color the RL Agent row
    for i in range(len(table_data)):
        if "RL Agent" in table_data[i][0]:
            for j in range(5):
                table[(i+1, j)].set_facecolor('#e8f8f5')

    plt.savefig(output_path / "comparison_table.png", dpi=150, bbox_inches="tight")
    print(f"✅ Comparison table saved to {output_path / 'comparison_table.png'}")

    plt.close('all')


def generate_demo_scenario(model, env: ClusterSchedulerEnv, output_dir: str = "./results"):
    """Generate a demo scenario showing agent decision-making."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating Demo Scenario")
    print(f"{'='*60}")

    obs, _ = env.reset(seed=9999)
    done = False
    step = 0

    demo_log = []

    while not done and step < 20:  # Show first 20 decisions
        action, _ = model.predict(obs, deterministic=True)

        # Get current job info
        if env.current_job_idx < env.num_jobs:
            job = env.jobs[env.current_job_idx]
            machine = env.machines[action] if action < len(env.machines) else None

            demo_log.append({
                "step": step + 1,
                "job_id": job.job_id,
                "job_type": job.task_type,
                "cpu_req": job.plan_cpu,
                "mem_req": job.plan_mem,
                "gpu_req": job.plan_gpu,
                "latency_sensitive": job.latency_sensitive,
                "chosen_machine": machine.machine_id if machine else "INVALID",
                "machine_cost": machine.cost_per_hour if machine else 0,
                "machine_cpu_free": machine.free_cpu if machine else 0,
                "machine_ram_free": machine.free_ram if machine else 0
            })

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    demo_df = pd.DataFrame(demo_log)
    demo_df.to_csv(output_path / "demo_scenario.csv", index=False)

    print("\nDemo Scenario (First 20 placements):")
    print(demo_df.to_string(index=False))
    print(f"\n✅ Demo scenario saved to {output_path / 'demo_scenario.csv'}")

    return demo_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Workload Placement Agent")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN", "A2C"])
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--compare-baselines", action="store_true", help="Compare with baseline schedulers")
    parser.add_argument("--generate-demo", action="store_true", help="Generate demo scenario")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Workload Placement Agent - Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes: {args.n_episodes}")

    # Load model
    model = load_trained_model(args.model_path, args.algorithm)

    # Create environment
    try:
        data = load_processed_data(args.data_dir)
    except FileNotFoundError:
        print("Processed data not found. Creating synthetic data...")
        from data_loader import create_synthetic_machines, create_synthetic_jobs, create_synthetic_instances
        machine_meta = create_synthetic_machines(10)
        batch_task = create_synthetic_jobs(1000)
        batch_instance = create_synthetic_instances(batch_task)
        data = preprocess_alibaba_data(machine_meta, batch_task, batch_instance, args.data_dir)

    env = ClusterSchedulerEnv(
        machines=data["machines"],
        jobs=data["jobs"],
        max_steps=1000,
        render_mode="human" if args.render else None
    )

    # Evaluate agent
    results, placements_df = evaluate_agent(
        model, env, 
        n_episodes=args.n_episodes,
        render=args.render
    )

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if not placements_df.empty:
        placements_df.to_csv(output_path / "placements.csv", index=False)

    print(f"\n✅ Results saved to {output_path}")

    # Compare with baselines
    if args.compare_baselines:
        comparison_results, comparison_df = compare_with_baselines(model, env, args.n_episodes)
        comparison_df.to_csv(output_path / "comparison.csv")
        visualize_results(comparison_df, args.output_dir)

    # Generate demo
    if args.generate_demo:
        demo_df = generate_demo_scenario(model, env, args.output_dir)

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
