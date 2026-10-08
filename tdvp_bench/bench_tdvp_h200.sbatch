#!/bin/bash
#SBATCH -J tdvp_bench_h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/tdvp_bench/%x_%j.out

# TDVP benchmark, H200 GPU column.
set -u
# GPU nodes have an older system libstdc++ than the conda env's numpy needs
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=$HOME/.conda/envs/my_env/bin/python
D=/n/home05/nchiu/tenpy_gpu_addon/tdvp_bench

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | grep "Model name"

# GPU column only (the CPU columns live in a100.json); grows its own state
# to chi=2048 so the two jobs can run concurrently
$PY $D/bench_tdvp.py --chis 128 256 512 1024 2048 --engines torch-cuda \
    --ckpt $D/state_L30_chi2048.npz -o $D/h200.json
echo ALL DONE
