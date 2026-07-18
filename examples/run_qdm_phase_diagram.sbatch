#!/bin/bash
#SBATCH -J qdm_pd256
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/examples/qdm_runs/qdm_pd256_%j.out

# Same warm forward/backward QDM phase-diagram protocol as
# run_qdm_phase_diagram.sbatch, at chi_max=256, max_sweeps=30:
# 6x6 finite, V/t in [-2, 3] (21 pts), classical columnar/staggered anchors,
# min-energy estimator per point.

set -u
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=$HOME/.conda/envs/my_env/bin/python
cd /n/home05/nchiu/tenpy_gpu_addon/examples
mkdir -p qdm_runs

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY qdm_phase_diagram.py --Lx 6 --Ny 12 --chi 256 --device cuda \
    --Vmin -1.0 --Vmax 2.5 --nV 36 --max-sweeps 60 \
    --out-dir /n/home05/nchiu/tenpy_gpu_addon/examples/qdm_runs/phase_diagram_6_12

echo "ALL DONE"
