#!/bin/bash
#SBATCH -J tdvp_krylov_work
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/tdvp_bench/%x_%j.out

# Krylov work per sweep, TorchTDVP vs stock tenpy, from the benchmark state.
set -u
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N
D=/n/home05/nchiu/tenpy_gpu_addon/tdvp_bench
cd $D
$HOME/.conda/envs/my_env/bin/python krylov_work.py --ckpt $D/state_L30.npz \
    --chis 256 512 --out $D/krylov_work.json
echo ALL DONE
