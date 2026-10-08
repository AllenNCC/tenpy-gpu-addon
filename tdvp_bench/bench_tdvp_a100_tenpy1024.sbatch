#!/bin/bash
#SBATCH -J tdvp_bench_a100_tenpy1024
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/tdvp_bench/%x_%j.out

# TDVP benchmark on one a100 GPU + a 16-core CPU allocation (same node), so
# torch-cuda and stock tenpy are timed on the same node.
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

# stock-tenpy chi=1024 point only, from the existing state_L30.npz checkpoint
$PY $D/bench_tdvp.py --chis 1024 --engines tenpy --ckpt $D/state_L30.npz \
    -o $D/a100_tenpy1024.json
echo ALL DONE
