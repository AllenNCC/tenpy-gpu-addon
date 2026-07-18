#!/bin/bash
#SBATCH -J qdm_bench_cpu
#SBATCH -p test
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/home05/nchiu/tenpy_gpu_addon/bench/%x_%j.out

# CPU baseline for the QDM benchmark (stock tenpy TwoSiteDMRGEngine).
# Fixed workload: SquareQDM Lx=6 x Ny=6 finite cylinder, t=1, V=0,
# V_blockade=50, random cold start (fixed seed), mixer on, 6 forced sweeps
# at fixed chi_max. Thread scaling at chi=256, then a chi scan at 16 threads.
# See bench_qdm.py for methodology; results land in results.md.

set -u
PY=$HOME/.conda/envs/my_env/bin/python
BENCH=/n/home05/nchiu/tenpy_gpu_addon/bench/bench_qdm.py
OUT=/n/home05/nchiu/tenpy_gpu_addon/bench
COMMON="--Lx 6 --Ny 6 --t 1.0 --V 0.0 --v-blockade 50.0 --sweeps 6 --engine tenpy"

echo "node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK:-?}"
lscpu | grep -E "Model name|Socket|Core" || true

# thread scaling at fixed chi=256
for NT in 1 4 16; do
  echo "=== threads=$NT chi=256 ==="
  $PY $BENCH $COMMON --chi 256 --threads $NT -o $OUT/cpu_t${NT}_chi256.json
done

# chi scaling at full threads, with the observable suite timed per state
echo "=== threads=16 chi scan ==="
$PY $BENCH $COMMON --chi 128 256 512 --threads 16 --measure \
    -o $OUT/cpu_t16_chiscan.json

# the expensive point last, so everything above lands even on timeout
echo "=== threads=16 chi=1024 ==="
$PY $BENCH $COMMON --chi 1024 --threads 16 --measure -o $OUT/cpu_t16_chi1024.json

echo "ALL DONE"
