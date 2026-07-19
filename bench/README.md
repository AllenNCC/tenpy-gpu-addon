# Benchmark & validation results

Living document for `tenpy_gpu_addon`, measured on the public QDM example
(`examples/qdm_model.py`). All runs on Harvard FASRC. Reproduce any number with the
`bench_cpu.sbatch` / `bench_gpu.sbatch` / `bench_gpu_h200.sbatch` /
`profile_gpu.sbatch` / `validate_gpu.sbatch` scripts in this folder; each
drives `bench_qdm.py`, whose docstring documents the methodology.

## Workload

Single fixed-workload DMRG point, no physics sweep: `SquareQDM`
(square-lattice quantum dimer model, Rokhsar–Kivelson parametrisation),
**Lx=6 × Ny=6 finite cylinder** — 18 four-state A-sublattice vertex sites,
MPO bond dim 26 — kinetic `t=1`, interaction `V=0`, perfect-matching
`V_blockade=50`. The model is **charge-free** (`LegCharge.from_trivial`):
every npc tensor is a single dense block, charge conservation not implemented yet.

**Cold start from a random entangled MPS** at bond dimension `chi_max`
(fixed seed 1234), mixer on, **6 forced
sweeps** at fixed `chi_max`, `svd_min=1e-10`. Quoted `time/sweep` = mean over
the last half of the sweeps. A random entangled MPS does full chi³-depth work from sweep 1. The same seed
is used for every engine and device, so all backends time the identical
state.

Engines under test (`bench_qdm.py --engine`):

| name | class | role |
|---|---|---|
| `tenpy` | stock `TwoSiteDMRGEngine` | CPU baseline |
| `torch` | `run_dmrg_torch` | fully GPU-resident loop — **production engine** |
| `dropin` | `TorchTwoSiteDMRGEngine` | stock engine + GPU Lanczos — **fail-safe** |

E0 at a fixed sweep count varies in the low decimals between engines/configs
(convergence-path differences, not errors); the timing comparison is
unaffected, and converged-accuracy is checked separately below.

## CPU baseline — tenpy 1.1.0, OpenBLAS, 16-core allocation (job 30511056)

Intel Xeon Platinum 8480CL (Sapphire Rapids), 2×56 cores; 16-core job.

Thread scaling at chi=256 (`cpu_t{1,4,16}_chi256.json`):

| BLAS threads | 1 | 4 | 16 |
|---|---|---|---|
| t/sweep | 285.5 s | 123.7 s | 93.7 s |

Speedup 2.3× (4t) / 3.05× (16t) over single-thread. Amdahl fit: ~72–76% of
runtime is threadable BLAS, the rest serial (`npc` bookkeeping, reshapes,
transposes) → ceiling ≈ 3.6–4.2× at any core count; 16 threads is already
near it. Production CPU advice: ≤ 8–16 cores/job; several parallel low-core
jobs give better throughput per core-hour than one wide job.

chi scaling (16 threads, `cpu_t16_chiscan.json` + `cpu_t16_chi1024.json`):

| chi | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| t/sweep | 38.8 s | 95.2 s | 337 s | 1565 s |

## GPU — torch backend on A100 (job 30511204) and H200 (job 30511205)

`run_dmrg_torch`: fully GPU-resident two-site DMRG (environments, Lanczos,
truncation on device; host crossed once per run), density-matrix mixer,
batched-reorthogonalisation Lanczos. chi=2048 runs with `--lanczos-max-iter 12` on both
cards (memory on the A100; kept on the H200 for comparability, which also
trims matvec count).

| chi | CPU 16t | A100 | vs CPU | H200 | H200/A100 |
|---|---|---|---|---|---|
| 128  | 38.8 s | 0.79 s | 49× | — | — |
| 256  | 95.2 s | 2.22 s | 43× | — | — |
| 512  | 337 s | 8.73 s | 39× | 3.57 s | 2.4× |
| 1024 | 1565 s | 48.87 s | 32× | 18.85 s | 2.6× |
| 2048 | n/a | 181.7 s * | — | 69.1 s * | 2.6× |

\* `--lanczos-max-iter 12`.

Readings:
- **A100 is 32–49× the 16-core CPU**, The
  advantage stays large in the production regime: 32× (A100) at chi=1024,
  ~83× for the H200 over the same well-tuned CPU.
- **H200 is 2.4–2.6× the A100** — consistent with the H200's ~3× fp64 tensor-core peak plus higher HBM
  bandwidth. At chi=512 that is ~94× over the 16-core CPU. The 141 GB card also
  runs chi=2048 without the memory pressure that forces `--lanczos-max-iter 12`
  on the 80 GB A100 (kept on for fair comparison).
- Production regime is chi ≥ 512 (user requirement), where the GPU advantage
  is real and the cards approach their fp64 rooflines.

## Drop-in hybrid engine — architecture comparison (job 30511204)

`TorchTwoSiteDMRGEngine` keeps the stock tenpy `TwoSiteDMRGEngine` fully in
charge (mixer, chi_list, truncation, convergence all stock) and only moves
the Lanczos matvec to the GPU. It is numerically the stock engine and covers
every tenpy feature the torch loop doesn't — the **fail-safe** and the fast
oracle when validating the torch engine.
The hybrid is 14–34× slower than the full loop, and the gap widens with chi.
The full loop is the production engine.

| chi | dropin/A100 | torch/A100 | dropin ÷ torch |
|---|---|---|---|
| 128 | 10.89 s | 0.79 s | 13.8× |
| 256 | 50.98 s | 2.22 s | 23.0× |
| 512 | 298.96 s | 8.73 s | 34.2× |

## GPU observables (DeviceMPS) — vs 16-thread CPU

`measure.py::DeviceMPS` mirrors tenpy's measurement API on device, so the
unmodified observable code runs on it by duck typing. Per-state timing of the
structure-factor suite (16 correlation matrices + expectations → m1², m3²)
and the dimer density, from the `--measure` flag on the runs above:

| chi | CPU tenpy (16t) | A100 DeviceMPS | H200 DeviceMPS | speedup (A100) |
|---|---|---|---|---|
| 128  | 4.47 s | 0.49 s | — | 9× |
| 256  | 10.6 s | 0.52 s | — | 20× |
| 512  | 39.6 s | 1.15 s | 0.77 s | 34× |
| 1024 | 171 s | 5.03 s | 1.86 s | 34× |

## Converged-accuracy validation (job 30513930)

`tests/`, run on a CUDA node so the gpu-marked oracle tiers run alongside the
CPU tiers. Trust chain: stock tenpy is the reference; both GPU engines
(`run_dmrg_torch`, `TorchTwoSiteDMRGEngine`) and `DeviceMPS` are certified
against it on small systems (Lx=4×Ny=2, chi ≤ 64).

**Result: 27/27 PASSED** (A100, 55 s).

- **`test_engines.py` (5)** — GPU engines reproduce stock `TwoSiteDMRGEngine`
  ground state. The full torch loop agrees to **E rel < 1e-10, overlap deficit
  < 1e-8** (QDM and a stock TFI chain, proving model-generality); the drop-in
  hybrid — numerically the stock engine — agrees to **E rel < 1e-12, overlap
  < 1e-10**. CUDA tiers hold the same bounds (torch 1e-10, hybrid 1e-12).
- **`test_lanczos.py` (16)** — the Lanczos-stability regression (below): no
  variational undershoot across 4 near-converged systems × 4 convergence
  tolerances {0, 1e-5, 1e-7, 1e-9}.
- **`test_measure.py` (6)** — `DeviceMPS` observables vs the tenpy measurement
  oracle: expectations / correlations / entropy to **atol 1e-10**, structure
  factors to **< 1e-9** (CPU) / **< 1e-8** (a production-shaped CUDA state),
  overlaps to 1e-10, including a model-generic (non-QDM) duck-typing check.

## Engine finding: Lanczos stability on near-converged states

The batched-reorthogonalisation Lanczos can, at Krylov exhaustion on a
near-converged (near-product) state, degenerate: a single Gram–Schmidt pass
leaves a cancellation residue along a cached direction that becomes a
duplicate basis vector, later projections over-subtract, `beta` grows instead
of vanishing, and eigh of the resulting fake tridiagonal is unbounded below —
producing variationally impossible energies, always with the mixer off. No
small-`beta` guard catches an exploding `beta`.

**Fix (in `dmrg.py::_lanczos`):** a faithful port of tenpy's
`LanczosGroundState` — per-iteration Ritz-residual convergence (exit at
exhaustion *before* instability builds), tenpy's breakdown cutoff, and CGS2
("twice is enough") two-pass reorthogonalisation. Regression coverage in
`tests/test_lanczos.py`; full mechanism and diagnosis in
[docs/lanczos-stability.md](../docs/lanczos-stability.md).

## Engine finding: GPU truncation routes through rho-eigh, not SVD

On CUDA the symmetric eigensolver, not the SVD, is the right truncation
primitive: cuSOLVER's dense fp64 SVD is dramatically slower than an eigh of
the density/Gram matrix at the square sizes truncation produces (the mixer-on
path always used the density-matrix eigh; the mixer-off path historically
called SVD and paid for it). The engine now routes **all** CUDA truncation
through the rho-eigh path — the mixer just adds its perturbation when active;
exact LAPACK SVD is kept for CPU. The Gram trick's accuracy cost is the
predicted √ε Schmidt floor (~1e-8), with E/observable impact ~1e-16 — the
same floor the validated mixer path always carried.

## Engine finding: GPU memory at large chi

Peak device memory at fixed chi is dominated not by the physics state but by
the machinery: environments (∝ chi·D·chi), the Krylov basis (∝
`lanczos_max_iter` × theta), the mixer perturbation intermediate, and
allocator fragmentation under constantly-varying tensor shapes. Everything
scales as chi², so a card's ceiling is a hard chi cutoff. **Countermeasures**
(defaults in the GPU sbatch scripts / engine): cap Krylov depth
(`--lanczos-max-iter 12`, used at chi=2048); pairwise contraction order that
contracts theta with the environment *before* the MPO in the mixer
perturbation (bounds the largest intermediate); and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for defragmentation. The
141 GB H200 pushes the ceiling well past the 80 GB A100 with no code changes.

## Kernel profile — torch.profiler, A100 MIG 3g.20gb, chi=512 (job 32499055)

`profile_gpu.sbatch` (3 sweeps, `--torch-profile`); full table in
`torchprof_chi512.txt`. Run on the `gpu_test` partition — an A100 **MIG
3g.20gb slice** (~3/7 of the SMs/bandwidth of the full card): absolute
timings are not comparable to the canonical bench numbers (21.2 s/profiled
sweep vs 8.7 s canonical on the full A100 — the slice ratio plus profiler
overhead), but kernel *shares* are set by the same SM architecture and are
what this run is for. Self-CUDA total 62.9 s; breakdown by self CUDA time:

| cost center | share | detail |
|---|---|---|
| GEMM (`aten::mm` ← `tensordot`) | **74.5%** | 46.8 s, 13.3k calls, ~3.5 ms avg; cutlass `d884gemm` fp64 *tensor-op* kernels — already on the fp64 tensor-core path |
| copies (`copy_`/`clone` from `reshape`) | **15.5%** | 14.8k elementwise kernels — tensordot permutation copies |
| `linalg_eigh` (rho-eigh truncation) | **7.1%** (7.5% incl. child kernels) | 102 calls @ 46.3 ms; dominated by `sytrd` tridiagonalization (~3.2 s), bandwidth-bound |
| everything else (GEMV, norms, launch overhead) | ~3% | 69.6k kernel launches cost only 0.27 s CUDA — compute-bound at chi=512 |

Reading vs the same profile of the 5-state Rydberg workload (MPO D=14,
charged npc tensors), which split 64% GEMM / 20% eigh / 13% copies: the QDM
is even more GEMM-dominated — its MPO bond dimension D=26 scales the GEMM
flops (Lanczos matvecs, environment updates) while the truncation eigh cost
(∝ (d·chi)³, D-independent) is unchanged, and the charge-free single-block
tensors leave no bookkeeping overhead. Implications for the optimization
backlog on this workload: **MPO sparsity** (the D=26 multiplier sits in the
74.5% share) and **single-site/CBE** outrank randomized top-k eigh, whose
ceiling here is the 7% truncation share (it was ~25% on the Rydberg
workload).

chi=1024 deliberately not profiled: both dominant cost centers scale as
chi³, so kernel shares are essentially chi-independent, and the profiler's
host-side `key_averages()` aggregation is the known failure mode of longer
profiled runs (OOM-thrash/timeout) — 3 sweeps at a single chi is the
reliable recipe.

## Truncation-kernel finding: mixer-off sweeps were 3.6× slower — FIXED

Discovered by accident in the cold-start A/B (job 30226856): sweeps cost
47 s with the mixer on but 165–188 s with it off, at identical chi=1024.
Cause: mixer-on truncation diagonalizes the density matrix (cuSOLVER
`syevd`, fast); mixer-off called `torch.linalg.svd` on theta — and
cuSOLVER's dense SVD is catastrophically slow. Nothing had ever timed the
mixer-off path before: the 6-sweep benchmarks and the profiler run all sat
inside the mixer's 15-sweep window, so **all canonical t/sweep numbers are
mixer-on numbers (they stand unchanged).**

Kernel shootout (`exp_svd_kernels.py`, job 30238108; fp64, log-decaying
spectrum with cond 1e12 — a realistic Schmidt tail):

| kernel | n=2560 | n=5120 | max singular-value error |
|---|---|---|---|
| `svd` default | 2.36 s | 14.3 s | (reference) |
| `svd driver='gesvdj'` | 2.36 s | 14.3 s | 0 — **it IS the default** |
| `svd driver='gesvd'` | 0.66 s | 2.95 s | ~1e-12 |
| `svd driver='gesvda'` | fails | fails | ill-conditioned input |
| **eigh of Gram matrix** | **0.068 s** | **0.35 s** | ~8e-9 |

Lessons: (1) PyTorch's CUDA SVD default is already gesvdj — "switch to the
Jacobi driver" is a no-op, and Jacobi is the *slowest* choice at large
square fp64; (2) on GPUs the symmetric eigensolver, not the SVD, is the
right truncation primitive — 8–40× faster across drivers; (3) the Gram
trick's accuracy cost is the predicted sqrt(eps) Schmidt floor, now
*measured* at ~8e-9 — the same floor the (validated) mixer path always
had; E/observable impact ~1e-16.

**Fix (in the engine):** all CUDA truncation now routes through the
rho-eigh path; the mixer just adds its perturbation when active. Exact
LAPACK SVD kept for CPU. In-situ confirmation (A/B job 30257862): no
mixer-on/off cost cliff — ~44 s/sweep throughout at chi=1024, ~4× faster
mixer-off sweeps than before the fix.
**Re-validation: 8/8 PASSED** (job 30251883): chi=512 production point at
E rel 2.1e-12 / overlap 3.4e-9 (matching the pre-change 4.2e-12), and the
Delta=12 ordered-manifold test unaffected by the ~1e-8 Schmidt floor. The
suite itself ran 32 min vs 44 pre-fix — its converged (mixer-off) sweeps
had been paying the SVD tax too. The later `import_mps`
canonicalize='auto' change is covered by the CPU tier (4/4) and the
staged-engine GPU re-certification (job 30281056, **8/8 PASSED**).

The engine also now records per sweep: mid-chain entanglement entropy
`S_mid` (first-order-sensitive convergence signal; E error is
second-order), mean entropy `S_mean` (what the tenpy-style stop watches),
and max discarded Schmidt weight `trunc_err` (the measured accuracy floor
of the run). All cheap by-products of truncation.

## Cold-start schedule A/B: staged chi ramp vs full-chi

*(Planned follow-up run. The engine supports a tenpy-style sweep-indexed
`chi_list` ramp with the mixer re-armed at each step —
`run_dmrg_torch(..., chi_list={0: 256, 5: 512, 10: 1024},
mixer_params={'disable_after': 5})`, as used in `examples/run_qdm.py`. The
A/B will time staged vs full-chi to a fixed relative accuracy on the QDM
point. — not yet run.)*

## Pending

Optional follow-up runs not yet done: the cold-start A/B (staged
`chi_list`). A first QDM phase diagram over V/t (6×6, chi=512, job
30530715) is in `examples/` — see `examples/README.md §5`.
