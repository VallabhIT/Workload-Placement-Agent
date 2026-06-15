"""
Workload Placement Simulator - Gymnasium Environment
Optimized for AMD Hackathon - Self-Optimizing Workload Placement Agent

This environment simulates a heterogeneous data center where an RL agent
learns to place incoming jobs on optimal machines based on cost, performance,
and energy efficiency.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Machine:
    """Represents a compute node in the cluster."""
    machine_id: str
    cpu_cores: int
    ram_gb: float
    gpus: int
    gpu_model: str
    cost_per_hour: float
    energy_watt: float
    failure_domain_1: int = 0
    failure_domain_2: str = ""

    # Runtime state
    used_cpu: float = 0.0
    used_ram: float = 0.0
    used_gpus: int = 0
    active_jobs: List[Dict] = None
    total_energy_consumed: float = 0.0
    total_cost_incurred: float = 0.0

    def __post_init__(self):
        if self.active_jobs is None:
            self.active_jobs = []

    @property
    def free_cpu(self) -> float:
        return self.cpu_cores - self.used_cpu

    @property
    def free_ram(self) -> float:
        return self.ram_gb - self.used_ram

    @property
    def free_gpus(self) -> int:
        return self.gpus - self.used_gpus

    @property
    def cpu_utilization(self) -> float:
        return self.used_cpu / self.cpu_cores if self.cpu_cores > 0 else 0

    @property
    def ram_utilization(self) -> float:
        return self.used_ram / self.ram_gb if self.ram_gb > 0 else 0

    def can_fit(self, cpu_req: float, ram_req: float, gpu_req: int) -> bool:
        """Check if machine has enough resources for a job."""
        return (self.free_cpu >= cpu_req and 
                self.free_ram >= ram_req and 
                self.free_gpus >= gpu_req)

    def allocate(self, job_id: str, cpu: float, ram: float, gpu: int, 
                 duration: float, timestamp: float):
        """Allocate resources for a job."""
        self.used_cpu += cpu
        self.used_ram += ram
        self.used_gpus += gpu
        self.active_jobs.append({
            'job_id': job_id,
            'cpu': cpu,
            'ram': ram,
            'gpu': gpu,
            'start_time': timestamp,
            'duration': duration,
            'end_time': timestamp + duration
        })

    def release_completed(self, current_time: float):
        """Release resources for jobs that have completed."""
        completed = [j for j in self.active_jobs if j['end_time'] <= current_time]
        for job in completed:
            self.used_cpu -= job['cpu']
            self.used_ram -= job['ram']
            self.used_gpus -= job['gpu']
            # Calculate energy and cost for completed job
            energy_kwh = (self.energy_watt * job['duration']) / 3600 / 1000
            self.total_energy_consumed += energy_kwh
            self.total_cost_incurred += self.cost_per_hour * (job['duration'] / 3600)
        self.active_jobs = [j for j in self.active_jobs if j['end_time'] > current_time]


@dataclass
class Job:
    """Represents a workload to be scheduled."""
    job_id: str
    task_name: str
    job_name: str
    task_type: str
    submit_time: float
    plan_cpu: float  # 100 = 1 core
    plan_mem: float  # normalized [0, 100]
    plan_gpu: int = 0
    duration_estimate: float = 300.0  # seconds
    priority: int = 1
    latency_sensitive: bool = False

    # Actual runtime (learned from historical data)
    actual_cpu_avg: float = None
    actual_mem_avg: float = None
    actual_duration: float = None

    # For tracking
    assigned_machine: str = None
    start_time: float = None
    end_time: float = None
    sla_met: bool = True


class ClusterSchedulerEnv(gym.Env):
    """
    Gymnasium environment for workload placement optimization.

    State: Current cluster utilization + next job requirements
    Action: Which machine to assign the next job to
    Reward: Negative cost (energy + instance cost + SLA penalties)
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, 
                 machines: List[Machine],
                 jobs: List[Job],
                 max_steps: int = 1000,
                 energy_price: float = 0.15,  # $/kWh
                 sla_penalty: float = 10.0,
                 fragmentation_penalty: float = 2.0,
                 render_mode: str = None):
        super().__init__()

        self.machines = machines
        self.jobs = jobs
        self.max_steps = max_steps
        self.energy_price = energy_price
        self.sla_penalty = sla_penalty
        self.fragmentation_penalty = fragmentation_penalty
        self.render_mode = render_mode

        self.num_machines = len(machines)
        self.num_jobs = len(jobs)

        # State dimensions
        # For each machine: [free_cpu, free_ram, free_gpus, cpu_util, ram_util, cost_per_hour, energy_watt]
        # Plus next job: [plan_cpu, plan_mem, plan_gpu, priority, latency_sensitive]
        self.machine_features = 7
        self.job_features = 5
        self.state_dim = self.num_machines * self.machine_features + self.job_features

        # Action space: which machine to place the job on
        self.action_space = spaces.Discrete(self.num_machines)

        # Observation space
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(self.state_dim,), dtype=np.float32
        )

        # Tracking
        self.current_step = 0
        self.current_job_idx = 0
        self.current_time = 0.0
        self.total_reward = 0.0
        self.total_cost = 0.0
        self.total_energy = 0.0
        self.sla_violations = 0
        self.placement_history = []

        # For reward shaping
        self.prev_avg_utilization = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # Reset machines
        for machine in self.machines:
            machine.used_cpu = 0.0
            machine.used_ram = 0.0
            machine.used_gpus = 0
            machine.active_jobs = []
            machine.total_energy_consumed = 0.0
            machine.total_cost_incurred = 0.0

        # Reset tracking
        self.current_step = 0
        self.current_job_idx = 0
        self.current_time = 0.0
        self.total_reward = 0.0
        self.total_cost = 0.0
        self.total_energy = 0.0
        self.sla_violations = 0
        self.placement_history = []
        self.prev_avg_utilization = 0.0

        # Sort jobs by submit time
        self.jobs.sort(key=lambda j: j.submit_time)

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Build state vector from current cluster state and next job."""
        state = []

        # Machine states
        for machine in self.machines:
            state.extend([
                machine.free_cpu,
                machine.free_ram,
                machine.free_gpus,
                machine.cpu_utilization * 100,
                machine.ram_utilization * 100,
                machine.cost_per_hour,
                machine.energy_watt / 100  # normalize
            ])

        # Next job features
        if self.current_job_idx < self.num_jobs:
            job = self.jobs[self.current_job_idx]
            state.extend([
                job.plan_cpu,
                job.plan_mem,
                job.plan_gpu,
                job.priority,
                1.0 if job.latency_sensitive else 0.0
            ])
        else:
            state.extend([0, 0, 0, 0, 0])

        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, machine: Machine, job: Job, 
                         placed: bool, wait_time: float = 0) -> float:
        """
        Calculate reward for a placement decision.

        Components:
        1. Cost penalty: hourly cost * duration
        2. Energy penalty: energy consumed * price
        3. SLA penalty: if latency-sensitive job placed on slow machine
        4. Fragmentation penalty: if placement causes high fragmentation
        5. Utilization bonus: reward for good resource packing
        """
        if not placed:
            return -100.0  # Heavy penalty for failed placement

        # Calculate job duration and cost
        duration_hours = job.duration_estimate / 3600
        cost = machine.cost_per_hour * duration_hours

        # Energy cost
        energy_kwh = (machine.energy_watt * job.duration_estimate) / 3600 / 1000
        energy_cost = energy_kwh * self.energy_price

        # SLA penalty for latency-sensitive jobs on slow machines
        sla_penalty = 0.0
        if job.latency_sensitive and machine.cost_per_hour < 1.0:
            # Placing latency-sensitive job on cheap/slow machine
            sla_penalty = self.sla_penalty
            self.sla_violations += 1

        # Fragmentation penalty (encourage tight packing)
        fragmentation = 0.0
        if machine.free_cpu < 1.0 and machine.free_cpu > 0:
            fragmentation += self.fragmentation_penalty  # Small CPU fragment
        if machine.free_ram < 1.0 and machine.free_ram > 0:
            fragmentation += self.fragmentation_penalty  # Small RAM fragment

        # Utilization bonus (reward for using resources efficiently)
        utilization_bonus = 0.0
        avg_util = np.mean([m.cpu_utilization for m in self.machines])
        if avg_util > self.prev_avg_utilization:
            utilization_bonus = 2.0  # Reward for increasing utilization
        self.prev_avg_utilization = avg_util

        # Wait time penalty
        wait_penalty = wait_time * 0.1

        # Total reward (negative because we want to minimize cost)
        reward = -(cost + energy_cost + sla_penalty + fragmentation + wait_penalty) + utilization_bonus

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one placement decision.

        Args:
            action: Index of machine to place the job on

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Get current job
        if self.current_job_idx >= self.num_jobs:
            # No more jobs, episode done
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()

        job = self.jobs[self.current_job_idx]
        self.current_time = max(self.current_time, job.submit_time)

        # Release completed jobs on all machines
        for machine in self.machines:
            machine.release_completed(self.current_time)

        # Try to place job
        placed = False
        wait_time = 0.0

        if 0 <= action < self.num_machines:
            machine = self.machines[action]

            # Check if machine can fit the job
            cpu_req = job.plan_cpu / 100  # Convert from normalized
            ram_req = job.plan_mem / 100 * machine.ram_gb  # Scale to machine's RAM

            if machine.can_fit(cpu_req, ram_req, job.plan_gpu):
                # Place the job
                machine.allocate(
                    job.job_id, cpu_req, ram_req, job.plan_gpu,
                    job.duration_estimate, self.current_time
                )
                job.assigned_machine = machine.machine_id
                job.start_time = self.current_time
                job.end_time = self.current_time + job.duration_estimate
                placed = True

                # Track placement
                self.placement_history.append({
                    'job_id': job.job_id,
                    'machine_id': machine.machine_id,
                    'timestamp': self.current_time,
                    'cost': machine.cost_per_hour * (job.duration_estimate / 3600),
                    'energy': (machine.energy_watt * job.duration_estimate) / 3600 / 1000
                })
            else:
                # Machine can't fit - job must wait or be rejected
                wait_time = 60.0  # Simulate 1 minute wait
        else:
            # Invalid action
            wait_time = 60.0

        # Calculate reward
        machine = self.machines[action] if 0 <= action < self.num_machines else None
        reward = self._calculate_reward(machine, job, placed, wait_time)

        # Update totals
        self.total_reward += reward
        if placed:
            self.total_cost += machine.cost_per_hour * (job.duration_estimate / 3600)
            self.total_energy += (machine.energy_watt * job.duration_estimate) / 3600 / 1000

        # Move to next job
        self.current_job_idx += 1

        # Check termination
        terminated = self.current_job_idx >= self.num_jobs
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict:
        """Return debugging info."""
        return {
            'step': self.current_step,
            'job_idx': self.current_job_idx,
            'total_jobs': self.num_jobs,
            'total_cost': self.total_cost,
            'total_energy_kwh': self.total_energy,
            'sla_violations': self.sla_violations,
            'avg_cpu_util': np.mean([m.cpu_utilization for m in self.machines]),
            'avg_ram_util': np.mean([m.ram_utilization for m in self.machines]),
            'active_jobs': sum(len(m.active_jobs) for m in self.machines)
        }

    def render(self):
        """Render current cluster state."""
        if self.render_mode == 'human':
            print(f"\n=== Step {self.current_step} | Time: {self.current_time:.1f}s ===")
            print(f"Jobs: {self.current_job_idx}/{self.num_jobs} | Cost: ${self.total_cost:.2f} | Energy: {self.total_energy:.3f} kWh")
            print("-" * 80)
            print(f"{'Machine':<15} {'CPU':>8} {'RAM':>8} {'GPU':>5} {'Util%':>8} {'Jobs':>5} {'Cost/hr':>10}")
            print("-" * 80)
            for m in self.machines:
                util = (m.cpu_utilization + m.ram_utilization) / 2 * 100
                print(f"{m.machine_id:<15} {m.used_cpu:>6.1f}/{m.cpu_cores:<1} {m.used_ram:>6.1f}/{m.ram_gb:<1} "
                      f"{m.used_gpus:>3}/{m.gpus:<1} {util:>7.1f}% {len(m.active_jobs):>5} ${m.cost_per_hour:>9.2f}")

    def get_placement_summary(self) -> pd.DataFrame:
        """Return summary of all placements."""
        return pd.DataFrame(self.placement_history)


class BaselineScheduler:
    """Simple baseline schedulers for comparison."""

    @staticmethod
    def first_fit(env: ClusterSchedulerEnv) -> int:
        """Place job on first machine that can fit it."""
        if env.current_job_idx >= env.num_jobs:
            return 0

        job = env.jobs[env.current_job_idx]
        cpu_req = job.plan_cpu / 100
        ram_req = job.plan_mem / 100 * 64  # Approximate

        for i, machine in enumerate(env.machines):
            if machine.can_fit(cpu_req, ram_req, job.plan_gpu):
                return i
        return 0  # Default to first machine if none fit

    @staticmethod
    def best_fit(env: ClusterSchedulerEnv) -> int:
        """Place job on machine with least remaining space after placement."""
        if env.current_job_idx >= env.num_jobs:
            return 0

        job = env.jobs[env.current_job_idx]
        cpu_req = job.plan_cpu / 100
        ram_req = job.plan_mem / 100 * 64

        best_idx = 0
        best_score = float('inf')

        for i, machine in enumerate(env.machines):
            if machine.can_fit(cpu_req, ram_req, job.plan_gpu):
                # Score: remaining space (lower is better = tighter packing)
                remaining = (machine.free_cpu - cpu_req) + (machine.free_ram - ram_req)
                if remaining < best_score:
                    best_score = remaining
                    best_idx = i

        return best_idx

    @staticmethod
    def least_loaded(env: ClusterSchedulerEnv) -> int:
        """Place job on machine with lowest utilization."""
        if env.current_job_idx >= env.num_jobs:
            return 0

        job = env.jobs[env.current_job_idx]
        cpu_req = job.plan_cpu / 100
        ram_req = job.plan_mem / 100 * 64

        best_idx = 0
        best_util = float('inf')

        for i, machine in enumerate(env.machines):
            if machine.can_fit(cpu_req, ram_req, job.plan_gpu):
                util = machine.cpu_utilization + machine.ram_utilization
                if util < best_util:
                    best_util = util
                    best_idx = i

        return best_idx

    @staticmethod
    def cheapest_fit(env: ClusterSchedulerEnv) -> int:
        """Place job on cheapest machine that can fit it."""
        if env.current_job_idx >= env.num_jobs:
            return 0

        job = env.jobs[env.current_job_idx]
        cpu_req = job.plan_cpu / 100
        ram_req = job.plan_mem / 100 * 64

        best_idx = 0
        best_cost = float('inf')

        for i, machine in enumerate(env.machines):
            if machine.can_fit(cpu_req, ram_req, job.plan_gpu):
                if machine.cost_per_hour < best_cost:
                    best_cost = machine.cost_per_hour
                    best_idx = i

        return best_idx
