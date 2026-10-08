#!/bin/bash
#SBATCH -J tdvp_validate
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/tdvp_bench/%x_%j.out

# Full test suite including the CUDA-vs-CPU-oracle tier.
set -u
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
cd /n/home05/nchiu/tenpy_gpu_addon
nvidia-smi --query-gpu=name --format=csv,noheader
$HOME/.conda/envs/my_env/bin/python -m pytest tests -v
