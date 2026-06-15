"""
Data Loader for Alibaba Cluster Trace v2018
Handles large files efficiently with sampling and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import json


def load_alibaba_trace_sample(
    data_dir: str,
    machine_meta_file: str = "machine_meta.csv",
    batch_task_file: str = "batch_task.csv",
    batch_instance_file: str = "batch_instance.csv",
    sample_fraction: float = 0.01,  # Use 1% of data by default
    max_jobs: int = 5000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and sample Alibaba trace data.

    Args:
        data_dir: Directory containing the CSV files
        machine_meta_file: Machine metadata filename
        batch_task_file: Batch task filename  
        batch_instance_file: Batch instance filename
        sample_fraction: Fraction of jobs to sample (0-1)
        max_jobs: Maximum number of jobs to load
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (machine_meta, batch_task, batch_instance) DataFrames
    """
    data_path = Path(data_dir)

    print(f"Loading Alibaba trace data from {data_dir}...")

    # Load machine metadata (small file, load fully)
    machine_meta_path = data_path / machine_meta_file
    if machine_meta_path.exists():
        machine_meta = pd.read_csv(machine_meta_path)
        print(f"✅ Loaded {len(machine_meta)} machines from machine_meta")
    else:
        print(f"⚠️ machine_meta.csv not found, creating synthetic machines")
        machine_meta = create_synthetic_machines()

    # Load batch tasks with sampling
    batch_task_path = data_path / batch_task_file
    if batch_task_path.exists():
        # For large files, use chunking and sampling
        print(f"Loading batch_task (sampling {sample_fraction*100:.1f}%)...")
        batch_task = sample_large_csv(batch_task_path, sample_fraction, max_jobs, random_seed)
        print(f"✅ Loaded {len(batch_task)} tasks from batch_task")
    else:
        print(f"⚠️ batch_task.csv not found, creating synthetic jobs")
        batch_task = create_synthetic_jobs(max_jobs)

    # Load batch instances (filter to match sampled tasks)
    batch_instance_path = data_path / batch_instance_file
    if batch_instance_path.exists() and len(batch_task) > 0:
        print(f"Loading batch_instance (filtering to sampled tasks)...")
        task_names = set(batch_task['task_name'].unique())
        batch_instance = filter_large_csv(batch_instance_path, task_names)
        print(f"✅ Loaded {len(batch_instance)} instances")
    else:
        print(f"⚠️ batch_instance.csv not found or no tasks, creating synthetic instances")
        batch_instance = create_synthetic_instances(batch_task)

    return machine_meta, batch_task, batch_instance


def sample_large_csv(file_path: Path, fraction: float, max_rows: int, seed: int) -> pd.DataFrame:
    """Sample rows from a large CSV file without loading it entirely."""

    # First, count total rows
    print(f"  Counting rows in {file_path.name}...")
    total_rows = sum(1 for _ in open(file_path, 'r')) - 1  # minus header
    print(f"  Total rows: {total_rows:,}")

    # Calculate skip rows
    np.random.seed(seed)
    n_samples = min(int(total_rows * fraction), max_rows)
    skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                       size=total_rows - n_samples, 
                                       replace=False))

    print(f"  Sampling {n_samples:,} rows...")
    df = pd.read_csv(file_path, skiprows=skip_rows)

    return df


def filter_large_csv(file_path: Path, task_names: set) -> pd.DataFrame:
    """Filter CSV to only include rows with matching task names."""

    chunks = []
    chunk_size = 100000

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        filtered = chunk[chunk['task_name'].isin(task_names)]
        if len(filtered) > 0:
            chunks.append(filtered)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.DataFrame()


def create_synthetic_machines(n_machines: int = 10) -> pd.DataFrame:
    """Create synthetic machine metadata for testing."""
    np.random.seed(42)

    machines = []
    machine_types = [
        {"type": "cpu-small", "cpu": 8, "ram": 32, "gpus": 0, "cost": 0.5, "energy": 150},
        {"type": "cpu-medium", "cpu": 16, "ram": 64, "gpus": 0, "cost": 1.0, "energy": 250},
        {"type": "cpu-large", "cpu": 32, "ram": 128, "gpus": 0, "cost": 2.0, "energy": 400},
        {"type": "gpu-basic", "cpu": 8, "ram": 32, "gpus": 1, "cost": 1.5, "energy": 300, "gpu_model": "MI100"},
        {"type": "gpu-fast", "cpu": 16, "ram": 64, "gpus": 1, "cost": 3.0, "energy": 500, "gpu_model": "MI200"},
        {"type": "gpu-huge", "cpu": 32, "ram": 128, "gpus": 2, "cost": 6.0, "energy": 800, "gpu_model": "MI300X"},
    ]

    for i in range(n_machines):
        mt = machine_types[i % len(machine_types)]
        machines.append({
            "machine_id": f"m{i:04d}",
            "time_stamp": 0,
            "failure_domain_1": i % 3,
            "failure_domain_2": f"rack_{i % 5}",
            "cpu_num": mt["cpu"],
            "mem_size": mt["ram"],
            "status": "active",
            "gpus": mt["gpus"],
            "cost_per_hour": mt["cost"],
            "energy_watt": mt["energy"],
            "gpu_model": mt.get("gpu_model", "")
        })

    return pd.DataFrame(machines)


def create_synthetic_jobs(n_jobs: int = 1000) -> pd.DataFrame:
    """Create synthetic job data for testing."""
    np.random.seed(42)

    job_types = [
        {"name": "batch_compute", "cpu": 200, "mem": 30, "gpu": 0, "duration": 600, "priority": 1},
        {"name": "ml_training", "cpu": 400, "mem": 60, "gpu": 1, "duration": 1800, "priority": 2},
        {"name": "inference", "cpu": 100, "mem": 15, "gpu": 0, "duration": 60, "priority": 3, "latency": True},
        {"name": "data_processing", "cpu": 300, "mem": 40, "gpu": 0, "duration": 1200, "priority": 1},
        {"name": "gpu_inference", "cpu": 200, "mem": 25, "gpu": 1, "duration": 120, "priority": 3, "latency": True},
    ]

    jobs = []
    for i in range(n_jobs):
        jt = np.random.choice(job_types)
        jobs.append({
            "task_name": f"task_{i:06d}",
            "instance_num": np.random.randint(1, 4),
            "job_name": f"job_{i//10:04d}",
            "task_type": jt["name"],
            "status": "Ready",
            "start_time": i * 30 + np.random.randint(0, 30),  # Every ~30s
            "end_time": 0,
            "plan_cpu": jt["cpu"] + np.random.randint(-50, 50),
            "plan_mem": jt["mem"] + np.random.randint(-10, 10),
            "plan_gpu": jt["gpu"],
            "duration": jt["duration"] + np.random.randint(-60, 60),
            "priority": jt["priority"],
            "latency_sensitive": jt.get("latency", False)
        })

    return pd.DataFrame(jobs)


def create_synthetic_instances(batch_task: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic instance data matching batch tasks."""
    instances = []

    for _, task in batch_task.iterrows():
        for j in range(task.get("instance_num", 1)):
            instances.append({
                "instance_name": f"{task['task_name']}_inst{j}",
                "task_name": task["task_name"],
                "job_name": task["job_name"],
                "task_type": task["task_type"],
                "status": "Terminated",
                "start_time": task["start_time"],
                "end_time": task["start_time"] + task.get("duration", 300),
                "machine_id": f"m{np.random.randint(0, 10):04d}",
                "seq_no": j,
                "total_seq_no": task.get("instance_num", 1),
                "cpu_avg": task["plan_cpu"] * 0.8,
                "cpu_max": task["plan_cpu"] * 1.2,
                "mem_avg": task["plan_mem"] * 0.8,
                "mem_max": task["plan_mem"] * 1.2
            })

    return pd.DataFrame(instances)


def preprocess_alibaba_data(
    machine_meta: pd.DataFrame,
    batch_task: pd.DataFrame, 
    batch_instance: pd.DataFrame,
    output_dir: str = "./processed_data"
) -> dict:
    """
    Preprocess Alibaba trace data into format suitable for RL environment.

    Returns:
        Dictionary with 'machines', 'jobs', 'placements' keys
    """
    from simulator import Machine, Job

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Process machines
    machines = []
    for _, row in machine_meta.iterrows():
        machines.append(Machine(
            machine_id=str(row.get("machine_id", f"m{len(machines):04d}")),
            cpu_cores=int(row.get("cpu_num", 8)),
            ram_gb=float(row.get("mem_size", 32)),
            gpus=int(row.get("gpus", 0)),
            gpu_model=str(row.get("gpu_model", "")),
            cost_per_hour=float(row.get("cost_per_hour", 1.0)),
            energy_watt=float(row.get("energy_watt", 200)),
            failure_domain_1=int(row.get("failure_domain_1", 0)),
            failure_domain_2=str(row.get("failure_domain_2", ""))
        ))

    # Process jobs from batch tasks
    jobs = []
    for _, row in batch_task.iterrows():
        # Estimate duration from instances if available
        duration = 300  # default 5 minutes
        if "end_time" in row and "start_time" in row and row["end_time"] > row["start_time"]:
            duration = row["end_time"] - row["start_time"]

        jobs.append(Job(
            job_id=str(row.get("task_name", f"job_{len(jobs):06d}")),
            task_name=str(row.get("task_name", "")),
            job_name=str(row.get("job_name", "")),
            task_type=str(row.get("task_type", "batch")),
            submit_time=float(row.get("start_time", len(jobs) * 30)),
            plan_cpu=float(row.get("plan_cpu", 100)),
            plan_mem=float(row.get("plan_mem", 20)),
            plan_gpu=int(row.get("plan_gpu", 0)),
            duration_estimate=max(duration, 60),  # minimum 1 minute
            priority=int(row.get("priority", 1)),
            latency_sensitive=bool(row.get("latency_sensitive", False))
        ))

    # Process actual placements from instances
    placements = {}
    if not batch_instance.empty:
        for _, row in batch_instance.iterrows():
            task_name = str(row.get("task_name", ""))
            if task_name not in placements:
                placements[task_name] = []
            placements[task_name].append({
                "machine_id": str(row.get("machine_id", "")),
                "cpu_avg": float(row.get("cpu_avg", 0)),
                "mem_avg": float(row.get("mem_avg", 0)),
                "start_time": float(row.get("start_time", 0)),
                "end_time": float(row.get("end_time", 0))
            })

    # Save processed data
    processed = {
        "machines": machines,
        "jobs": jobs,
        "placements": placements,
        "n_machines": len(machines),
        "n_jobs": len(jobs)
    }

    with open(output_path / "processed_data.pkl", "wb") as f:
        pickle.dump(processed, f)

    # Also save as JSON for inspection
    json_data = {
        "machines": [{"id": m.machine_id, "cpu": m.cpu_cores, "ram": m.ram_gb, 
                      "gpus": m.gpus, "cost": m.cost_per_hour} for m in machines],
        "jobs": [{"id": j.job_id, "cpu": j.plan_cpu, "mem": j.plan_mem, 
                  "gpu": j.plan_gpu, "duration": j.duration_estimate} for j in jobs[:100]],
        "n_machines": len(machines),
        "n_jobs": len(jobs)
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✅ Preprocessed data saved to {output_dir}")
    print(f"   Machines: {len(machines)}")
    print(f"   Jobs: {len(jobs)}")
    print(f"   Placements: {len(placements)}")

    return processed


def load_processed_data(data_dir: str = "./processed_data") -> dict:
    """Load preprocessed data from pickle file."""
    data_path = Path(data_dir) / "processed_data.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run preprocess_alibaba_data first.")

    with open(data_path, "rb") as f:
        return pickle.load(f)
