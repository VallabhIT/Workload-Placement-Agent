#!/usr/bin/env python3
"""
Quick Start Script - Test the entire pipeline locally
Run this to verify everything works before deploying to AMD Cloud.
"""

import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import ClusterSchedulerEnv, Machine, Job, BaselineScheduler
from data_loader import (
    create_synthetic_machines, create_synthetic_jobs, 
    create_synthetic_instances, preprocess_alibaba_data
)

import numpy as np
import pandas as pd

print("=" * 70)
print("🚀 WORKLOAD PLACEMENT AGENT - QUICK START TEST")
print("=" * 70)

# Step 1: Create synthetic data
print("\n📦 Step 1: Creating synthetic data...")
machine_meta = create_synthetic_machines(10)
batch_task = create_synthetic_jobs(500)
batch_instance = create_synthetic_instances(batch_task)
data = preprocess_alibaba_data(machine_meta, batch_task, batch_instance, './data')
print(f"   ✅ Created {data['n_machines']} machines, {data['n_jobs']} jobs")

# Step 2: Create environment
print("\n🏗️  Step 2: Creating environment...")
env = ClusterSchedulerEnv(
    machines=data['machines'],
    jobs=data['jobs'],
    max_steps=500,
    render_mode=None
)
print(f"   ✅ Environment created: {env.observation_space.shape[0]}D state, {env.action_space.n} actions")

# Step 3: Test baselines
print("\n📊 Step 3: Testing baseline schedulers...")
baselines = {
    "First Fit": BaselineScheduler.first_fit,
    "Best Fit": BaselineScheduler.best_fit,
    "Least Loaded": BaselineScheduler.least_loaded,
    "Cheapest Fit": BaselineScheduler.cheapest_fit
}

baseline_results = {}
for name, scheduler_fn in baselines.items():
    obs, _ = env.reset(seed=42)
    done = False
    total_reward = 0

    while not done:
        action = scheduler_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    baseline_results[name] = {
        'reward': total_reward,
        'cost': info['total_cost'],
        'energy': info['total_energy_kwh'],
        'sla_violations': info['sla_violations']
    }
    print(f"   {name:15s}: Reward={total_reward:8.2f}, Cost=${info['total_cost']:6.2f}, "
          f"Energy={info['total_energy_kwh']:6.3f}kWh, SLAs={info['sla_violations']}")

# Step 4: Quick RL training (very short for testing)
print("\n🤖 Step 4: Training RL agent (quick test - 5000 steps)...")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    train_env = ClusterSchedulerEnv(
        machines=data['machines'],
        jobs=data['jobs'],
        max_steps=500
    )
    train_env = Monitor(train_env)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        verbose=0,
        device="cpu"
    )

    model.learn(total_timesteps=5000, progress_bar=True)

    # Evaluate trained agent
    eval_env = ClusterSchedulerEnv(
        machines=data['machines'],
        jobs=data['jobs'],
        max_steps=500
    )

    obs, _ = eval_env.reset(seed=999)
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    baseline_results["RL Agent"] = {
        'reward': total_reward,
        'cost': info['total_cost'],
        'energy': info['total_energy_kwh'],
        'sla_violations': info['sla_violations']
    }

    print(f"   {'RL Agent':15s}: Reward={total_reward:8.2f}, Cost=${info['total_cost']:6.2f}, "
          f"Energy={info['total_energy_kwh']:6.3f}kWh, SLAs={info['sla_violations']}")

    # Save model
    os.makedirs('./models', exist_ok=True)
    model.save('./models/quick_test_model')
    print(f"   ✅ Model saved to ./models/quick_test_model")

except Exception as e:
    print(f"   ⚠️  RL training skipped: {e}")
    print("   (This is OK - baselines work fine for testing)")

# Step 5: Summary
print("\n" + "=" * 70)
print("📈 RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(baseline_results).T
print(results_df.to_string())

print("\n" + "=" * 70)
print("✅ QUICK START COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run full training on AMD Cloud: python train.py")
print("  2. Evaluate trained model: python evaluate.py --model-path ./models/PPO_final_model.zip")
print("  3. Interactive demo: jupyter notebook notebooks/demo.ipynb")
print("\nFor detailed setup, see SETUP_GUIDE.md")
