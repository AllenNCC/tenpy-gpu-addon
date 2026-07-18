#!/bin/bash
#SBATCH -J qdm_validate
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/bench/%x_%j.out

# Converged-accuracy validation for RESULTS.md: the full addon test suite
# (test_engines, test_lanczos, test_measure) on a CUDA node so the gpu-marked
# oracle tiers run alongside the CPU tiers. Both GPU engines (run_dmrg_torch,
# TorchTwoSiteDMRGEngine) and DeviceMPS are checked against stock tenpy.

set -u
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=$HOME/.conda/envs/my_env/bin/python
cd /n/home05/nchiu/tenpy_gpu_addon

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

$PY -m pytest tests/ -v -rA --no-header 2>&1

echo "ALL DONE"
