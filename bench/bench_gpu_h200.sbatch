#!/bin/bash
#SBATCH -J qdm_bench_h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/bench/%x_%j.out

# H200 column of the QDM GPU benchmark: same workload as bench_gpu.sbatch,
# production chi range only (512/1024/2048). chi=2048 keeps
# --lanczos-max-iter 12 so the number is directly comparable to the A100
# column (same matvec count), even though the 141 GB card doesn't need the cap.

set -u
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=$HOME/.conda/envs/my_env/bin/python
BENCH=/n/home05/nchiu/tenpy_gpu_addon/bench/bench_qdm.py
OUT=/n/home05/nchiu/tenpy_gpu_addon/bench
COMMON="--Lx 6 --Ny 6 --t 1.0 --V 0.0 --v-blockade 50.0 --sweeps 6 --device cuda"

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== torch/cuda H200 chi scan (+observables) ==="
$PY $BENCH $COMMON --engine torch --chi 512 1024 --measure \
    -o $OUT/h200_torch_chiscan.json

echo "=== torch/cuda H200 chi=2048 ==="
$PY $BENCH $COMMON --engine torch --chi 2048 --lanczos-max-iter 12 \
    -o $OUT/h200_torch_chi2048.json

echo "ALL DONE"
