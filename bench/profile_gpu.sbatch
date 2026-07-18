#!/bin/bash
#SBATCH -J qdm_profile_gpu
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 01:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/bench/%x_%j.out

# torch.profiler pass at chi=512: kernel-level CUDA-time ranking of
# the torch engine on the QDM workload (GEMM vs eigh vs reshape-copies vs
# launch overhead) — fills the "Kernel profile" section of RESULTS.md.
# Table -> torchprof_chi512.txt. Timings here carry profiler overhead;
# canonical numbers stay the ones from bench_gpu.sbatch.
#
# gpu_test partition = A100 3g.20gb MIG slices (~3/7 of a full A100, 20 GB):
# absolute times are NOT comparable to the full-card bench numbers, but
# kernel *shares* are what this run is for, and chi=512 fits in 20 GB.
#
# 3 sweeps, chi=512 only: the profiler's host-side event stream is huge —
# key_averages() post-processing OOM-thrashes/times out on longer runs.
# (60G is the gpu_test per-GPU cap; enough for this event volume.) Both
# dominant cost centers scale as chi^3, so kernel *shares* are essentially
# chi-independent and carry to chi=1024.

set -u
# GPU nodes have an older system libstdc++ than the conda env's numpy needs
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=$HOME/.conda/envs/my_env/bin/python
BENCH=/n/home05/nchiu/tenpy_gpu_addon/bench/bench_qdm.py
OUT=/n/home05/nchiu/tenpy_gpu_addon/bench

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

$PY $BENCH --Lx 6 --Ny 6 --t 1.0 --V 0.0 --v-blockade 50.0 \
    --engine torch --device cuda --chi 512 --sweeps 3 \
    --torch-profile $OUT/torchprof \
    -o $OUT/gpu_torch_profiled.json

echo "ALL DONE"
