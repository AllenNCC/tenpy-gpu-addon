#!/bin/bash
#SBATCH -J qdm_bench_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/bench/%x_%j.out

# A100 column of the QDM GPU benchmark. Same fixed workload as bench_cpu.sbatch
# (SquareQDM Lx=6 x Ny=6 finite, t=1, V=0, V_blockade=50, random cold start,
# mixer, 6 sweeps). Three engines:
#   torch   run_dmrg_torch          (fully GPU-resident loop; production engine)
#   dropin  TorchTwoSiteDMRGEngine  (stock tenpy engine + GPU Lanczos; fail-safe)
# CPU baselines are in bench_cpu.sbatch / results.md and are not rerun here.

set -u
# GPU nodes have an older system libstdc++ than the conda env's numpy needs
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
# avoid allocator fragmentation at large chi (see results.md memory notes)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=$HOME/.conda/envs/my_env/bin/python
BENCH=/n/home05/nchiu/tenpy_gpu_addon/bench/bench_qdm.py
OUT=/n/home05/nchiu/tenpy_gpu_addon/bench
COMMON="--Lx 6 --Ny 6 --t 1.0 --V 0.0 --v-blockade 50.0 --sweeps 6 --device cuda"

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== torch/cuda chi scan (+observables) ==="
$PY $BENCH $COMMON --engine torch --chi 128 256 512 1024 --measure \
    -o $OUT/gpu_torch_chiscan.json

# chi=2048: cap the Krylov buffer (memory ~ max_iter x theta)
echo "=== torch/cuda chi=2048 ==="
$PY $BENCH $COMMON --engine torch --chi 2048 --lanczos-max-iter 12 \
    -o $OUT/gpu_torch_chi2048.json

# drop-in hybrid engine: architecture comparison vs the full loop
echo "=== dropin/cuda chi scan ==="
$PY $BENCH $COMMON --engine dropin --chi 128 256 512 \
    -o $OUT/gpu_dropin_chiscan.json

echo "ALL DONE"
