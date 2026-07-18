#!/bin/bash
#SBATCH -J qdm_pdfine512
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/examples/qdm_runs/qdm_pdfine512_%j.out

# FINE phase-diagram scan at chi_max=512: same warm forward/backward protocol
# as the chi=256 run (classical columnar/staggered anchors, min-energy
# estimator per point), 6x12 finite cylinder, V/t in [-1.0, 2.5] with step
# 0.05 (71 pts), and --save-psi so EVERY point's ground-state MPS is pickled
# (re-measurable later; chains resume from the pickles if the job times out —
# just resubmit this script).

set -u
export LD_LIBRARY_PATH=$HOME/.conda/envs/my_env/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=$HOME/.conda/envs/my_env/bin/python
cd /n/home05/nchiu/tenpy_gpu_addon/examples
mkdir -p qdm_runs

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUT=/n/home05/nchiu/tenpy_gpu_addon/examples/qdm_runs/phase_diagram_6_12

$PY qdm_phase_diagram.py --Lx 6 --Ny 12 --chi 256 --device cuda \
    --Vmin -1.0 --Vmax 2.5 --nV 36 --max-sweeps 60 --save-psi \
    --out-dir $OUT

# dimer-dimer correlation D(r)/C(r) for every saved state of both chains
# $PY qdm_dimer_corr.py --Lx 6 --Ny 12 --device cuda --from-dir $OUT

echo "ALL DONE"
