# TDVP benchmark

Timing of `TorchTDVP` (two-site TDVP, `tenpy_gpu_addon/tdvp.py`) against
TeNPy's `TwoSiteTDVPEngine`, measured on the TDVP example
(`examples/run_tdvp_quench.py`). All runs on Harvard FASRC. Reproduce with the
`bench_tdvp_*.sbatch` scripts in this folder; each drives `bench_tdvp.py`,
whose docstring documents the methodology.

## Workload

The conformal-cooling quench of Fig. 2 of arXiv:1611.04591: `SpinOneBLBQChain`
(`examples/spin1_chain.py`), **L = 30 spin-1 sites**, γ = 3/4, MPO bond
dimension 14, charge-free. Bond scale λ = 0.17 on the bath, linear ramp,
λ = 1 on the system; start from exp(−0.35 H)|Néel⟩; real-time step dt = 0.1.

Real-time evolution at finite energy density saturates any bond dimension
within a few 1/J, so a production run spends essentially all its time at
`chi = chi_max`. The benchmark therefore times sweeps on a state that is
already entangled:

1. **grow** — evolve on the GPU to t = 6 with `chi_max` = the largest chi of
   the run, and checkpoint. At t = 6 the central bonds are saturated; the
   bath and edge bonds are still filling in (see the run-time estimates
   below).
2. **time** — every engine loads the *same* checkpoint, truncates it to the
   chi under test, and runs 3 sweeps (2 for TeNPy at chi = 1024). The first
   sweep at a reduced chi also performs that truncation, so it is excluded
   from the mean; the quoted `time/sweep` is the mean of the rest.

One sweep = one second-order TDVP step: 2 × 29 two-site Krylov exponentials
plus 2 × 28 one-site ones, environment updates and truncations.

| engine | class | role |
|---|---|---|
| TeNPy | `tenpy.algorithms.tdvp.TwoSiteTDVPEngine` | CPU baseline |
| GPU | `tenpy_gpu_addon.TorchTDVP(device='cuda')` | this package |

## Results

TeNPy 1.1.0 (OpenBLAS), 16 threads, and the A100 timed in the same job on
the same node (Xeon Platinum 8358, `a100.json`, jobs 51161381 + 51171298).
H200 column from its own job (`h200.json`, job 51161382), on a state grown at
chi = 2048 and truncated to each chi — the same workload up to the state's
fine details.

| chi | TeNPy, 16 CPU cores | A100 | vs CPU | H200 | vs CPU | H200/A100 |
|---|---|---|---|---|---|---|
| 128  | 14.7 s | 0.97 s | 15× | 0.69 s | 21× | 1.4× |
| 256  | 47.1 s | 2.08 s | 23× | 1.15 s | 41× | 1.8× |
| 512  | 212 s  | 7.10 s | 30× | 3.17 s | 67× | 2.2× |
| 1024 | 1009 s | 34.4 s | 29× | 14.3 s | 71× | 2.4× |
| 2048 | —      | —      | —   | 72.9 s | —   | —    |

Readings:

- **The GPU advantage grows with chi** and is ~30× (A100) / ~70× (H200) in
  the production regime chi ≥ 512 — the same picture as DMRG
  (`bench/README.md`), since a TDVP sweep is the same chi³ contractions
  (environment updates and two-site matvecs), with a Krylov exponential in
  place of the Lanczos ground-state solve.
- **The H200 gains over the A100 grow with chi** (1.4× → 2.4×): at small chi
  both cards are latency-bound, at large chi the H200's higher fp64
  throughput and memory bandwidth show.
- **chi = 2048 fits on the H200** (141 GB): the Krylov basis is
  `krylov_max + 1` two-site vectors of chi²d² complex entries (~19 GB at
  chi = 2048, d = 3).

What this means for the example's full run (t = 100/J at dt = 0.1, i.e.
1000 sweeps at chi = 1024):

| | TeNPy, 16 CPU cores | A100 | H200 |
|---|---|---|---|
| at the benchmark state's cost (lower bound) | 12 days | 9.6 h | 4.0 h |
| fully saturated bond profile (×2.2) | 26 days | 21 h | 8.8 h |

The benchmark state (t = 6) is not yet saturated: only 7 of its 29 bonds are
at chi = 1024. The bath bonds (λ = 0.17, where dynamics run ~6× slower) sit
at chi ≈ 120–190, and the bonds near the right edge are still small too. A
profile saturated wherever d^min(i, L−i) allows costs 2.2× more per sweep at
chi = 1024 (1.8× at 512, 1.5× at 256), counting two-site matvec work. A long
run fills in towards that profile, so the second row is the realistic
estimate. The speedup ratios do not depend on this, because both engines
spend their time on the same chi³ contractions. (The paper's authors report
that their t = 100/J run took several weeks of computing time.)

## Notes

- **Truncation on CUDA goes through `eigh` of the reduced density matrix**,
  not SVD — the same choice and the same ~1e-8 Schmidt-value floor as the
  DMRG loop (see the main README). With `chi_max` binding, the discarded
  weight per sweep (~1e-7–1e-6 here at chi = 1024 by t ≈ 10) is orders of
  magnitude above that floor. Below that floor `svd_min` is raised to
  1e-7 on this path, so noise directions are not kept before `chi_max`
  binds. Details: main README, § "The mixer", item 3.
- **Both engines do comparable Krylov work.** They stop the Krylov
  exponential differently (TeNPy: |c_k| < 1e-14, no reorthogonalisation;
  here: |β_k c_k| < 1e-12, CGS2). `krylov_work.py` counts effective-Hamiltonian
  matvecs over one sweep from the benchmark state (`krylov_work.json`,
  job 51378492). TorchTDVP did 973 (χ = 256) and 977 (χ = 512) matvecs per
  sweep, TeNPy 1072 and 1080, about 10% more. Against a TorchTDVP reference
  with tolerance 1e-15 (1143–1149 matvecs), the states differ by
  1 − |⟨ψ|ψ_ref⟩| = 4–6e-15 (TorchTDVP) and 4–8e-13 (TeNPy, whose SVD
  truncation path also differs from the reference). Both are far below the
  ~1e-7 per-sweep truncation error. So at most ~10% of the measured speedup
  comes from the stopping rule, and the rest is the hardware.
- **Per-sweep time keeps rising during growth** even after `max(chi)`
  reaches `chi_max`: the central bonds saturate first and the outer ones
  fill in afterwards (light cone), so work per sweep grows until the whole
  profile is saturated. Benchmarks compare engines on one fixed state for
  this reason.
- Accuracy is covered by `tests/test_tdvp.py` (agreement with TeNPy's TDVP
  step for step, on CPU and CUDA; exact dynamics; energy conservation).
