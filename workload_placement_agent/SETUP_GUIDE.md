# Workload Placement Agent - Setup Guide

## Project Overview
Self-Optimizing Workload Placement Agent using Reinforcement Learning for AMD Hackathon.

## Directory Structure
```
workload_placement_agent/
├── simulator.py          # Core Gymnasium environment
├── data_loader.py        # Alibaba trace data loader
├── train.py             # Training script for AMD Cloud
├── evaluate.py          # Evaluation and demo script
├── requirements.txt     # Python dependencies
├── data/               # Data directory (create this)
├── models/             # Saved models (auto-created)
└── results/            # Evaluation results (auto-created)
```

## Step 1: Local Development Setup (MacBook)

### 1.1 Create Conda Environment
```bash
conda create -n workload_agent python=3.10
conda activate workload_agent
```

### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Verify Installation
```bash
python -c "import gymnasium; import stable_baselines3; print('✅ Dependencies installed successfully')"
```

## Step 2: Prepare Data

### Option A: Use Alibaba Trace (if available)
Place your Alibaba trace CSV files in the `data/` directory:
- `machine_meta.csv`
- `batch_task.csv`
- `batch_instance.csv`

### Option B: Use Synthetic Data (Default)
If you don't have the Alibaba data, the code will automatically generate synthetic data for testing.

### Preprocess Data
```bash
python -c "
from data_loader import *
from simulator import *

# This will create synthetic data if Alibaba files are not found
machine_meta = create_synthetic_machines(10)
batch_task = create_synthetic_jobs(1000)
batch_instance = create_synthetic_instances(batch_task)

data = preprocess_alibaba_data(machine_meta, batch_task, batch_instance, './data')
print(f'✅ Data prepared: {data["n_machines"]} machines, {data["n_jobs"]} jobs')
"
```

## Step 3: Quick Test (Local CPU)

### Test the Environment
```bash
python -c "
from simulator import *
from data_loader import load_processed_data

data = load_processed_data('./data')
env = ClusterSchedulerEnv(machines=data['machines'], jobs=data['jobs'])
obs, _ = env.reset()
print(f'Observation shape: {obs.shape}')
print(f'Action space: {env.action_space}')
print('✅ Environment working correctly')
"
```

### Test Baseline Schedulers
```bash
python train.py --data-dir ./data --eval-baselines --skip-training
```

## Step 4: Training on AMD Cloud

### 4.1 Access AMD Developer Cloud
1. Go to: https://cloud.amd.com/
2. Sign up with your hackathon credentials
3. Claim your $100 free credits
4. Launch an instance with ROCm pre-installed (Ubuntu + ROCm)

### 4.2 Setup AMD Cloud Instance
SSH into your instance:
```bash
ssh -i /path/to/your-key.pem ubuntu@your-instance-ip
```

Install dependencies on AMD Cloud:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create environment
conda create -n workload_agent python=3.10
conda activate workload_agent

# Install PyTorch with ROCm support
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6

# Install other dependencies
pip install gymnasium stable-baselines3 numpy pandas matplotlib seaborn scikit-learn tensorboard

# Verify ROCm
rocm-smi
```

### 4.3 Transfer Code to AMD Cloud
From your MacBook:
```bash
# Create archive
tar -czf workload_agent.tar.gz workload_placement_agent/

# Transfer to AMD Cloud
scp -i /path/to/your-key.pem workload_agent.tar.gz ubuntu@your-instance-ip:/home/ubuntu/

# SSH in and extract
ssh -i /path/to/your-key.pem ubuntu@your-instance-ip
tar -xzf workload_agent.tar.gz
cd workload_placement_agent
```

### 4.4 Train the Agent
```bash
# Basic training (CPU)
python train.py --data-dir ./data --output-dir ./models --algorithm PPO --total-timesteps 100000

# Training with AMD GPU (if available)
python train.py --data-dir ./data --output-dir ./models --algorithm PPO --total-timesteps 500000 --device cuda --n-envs 8

# With custom hyperparameters
python train.py \
    --data-dir ./data \
    --output-dir ./models \
    --algorithm PPO \
    --total-timesteps 500000 \
    --n-envs 8 \
    --learning-rate 3e-4 \
    --batch-size 128 \
    --n-steps 2048 \
    --device cuda
```

### 4.5 Monitor Training
```bash
# In a separate terminal, forward tensorboard
ssh -L 6006:localhost:6006 -i /path/to/your-key.pem ubuntu@your-instance-ip

# On AMD Cloud, run:
tensorboard --logdir ./models/logs/tensorboard --port 6006

# Open http://localhost:6006 on your MacBook
```

## Step 5: Evaluate and Compare

### Evaluate Trained Agent
```bash
python evaluate.py \
    --model-path ./models/PPO_final_model.zip \
    --algorithm PPO \
    --data-dir ./data \
    --n-episodes 20 \
    --compare-baselines \
    --generate-demo
```

### Results will be saved in:
- `./results/comparison_results.png` - Visual comparison
- `./results/comparison_table.png` - Summary table
- `./results/demo_scenario.csv` - Agent decision log

## Step 6: Download Results to MacBook
```bash
# From MacBook
scp -i /path/to/your-key.pem -r ubuntu@your-instance-ip:/home/ubuntu/workload_placement_agent/results ./results_from_cloud
scp -i /path/to/your-key.pem ubuntu@your-instance-ip:/home/ubuntu/workload_placement_agent/models/*.zip ./models_from_cloud
```

## Quick Start Commands

### Full Pipeline (Local Testing)
```bash
# 1. Setup data
python -c "from data_loader import *; from simulator import *; m=create_synthetic_machines(10); j=create_synthetic_jobs(1000); i=create_synthetic_instances(j); preprocess_alibaba_data(m,j,i,'./data')"

# 2. Evaluate baselines
python train.py --data-dir ./data --eval-baselines --skip-training

# 3. Quick train (for testing)
python train.py --data-dir ./data --output-dir ./models --total-timesteps 10000 --n-envs 2

# 4. Evaluate
python evaluate.py --model-path ./models/PPO_final_model.zip --compare-baselines --generate-demo
```

## Troubleshooting

### Issue: "No module named 'gymnasium'"
**Solution:** Make sure you're in the correct conda environment:
```bash
conda activate workload_agent
```

### Issue: "CUDA/ROCm not available"
**Solution:** Training will fall back to CPU. For AMD Cloud, verify ROCm:
```bash
rocm-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "Processed data not found"
**Solution:** Run the data preprocessing step first, or the code will auto-generate synthetic data.

### Issue: Training is very slow
**Solutions:**
- Reduce `--total-timesteps` for testing
- Increase `--n-envs` for parallel training
- Use GPU with `--device cuda`
- Reduce job count in synthetic data

## Hackathon Demo Checklist
- [ ] Environment tested locally
- [ ] Baseline results generated
- [ ] Agent trained on AMD Cloud
- [ ] Comparison visualization created
- [ ] Demo scenario generated
- [ ] Results downloaded to MacBook
- [ ] Presentation slides prepared

## Useful Resources
- Alibaba Cluster Trace: https://github.com/alibaba/clusterdata
- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/
- AMD ROCm: https://www.amd.com/en/products/software/rocm.html
- Gymnasium: https://gymnasium.farama.org/
