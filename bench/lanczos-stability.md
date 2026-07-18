# Lanczos stability notes: why `_lanczos` mirrors tenpy

`tenpy_gpu_addon.dmrg._lanczos` is a faithful port of tenpy's
`LanczosGroundState` (`tenpy.linalg.krylov_based`) to dense torch tensors.
This document records why: our first, home-grown Lanczos produced
**variationally impossible energies** in production, the failure mode is
subtle enough to silently corrupt physics results, and the guards that
prevent it are easy to break again by "optimizing" the code. Read this
before touching the function.

## Symptom

Isolated per-sweep energies far below the variational floor, on runs whose
truncation error and entropies look perfectly healthy:

| where | reported E | true ground energy |
|---|---|---|
| 6x6 cylinder, production run deep in the ordered phase | -322.56 | -253.28 |
| 6x12 cylinder, two production runs | -2801 ... -6184 | -542.40 |

Always on **near-product states** (S_mid ~ 1e-5, converged crystals deep in
the ordered phase) and always with the **mixer off** — the mixer's
perturbation keeps the local problem away from the trigger, so corruption
starts exactly at `disable_after`. A corrupted eigenvector is *written into
the MPS*, so it damages more than the energy record: densities, order
parameters, and (in warm chains) the seed handed to the next point. It also
rank-inflates the state — crystal anchors that should sit at chi ~ 70 get
pinned at chi_max, costing ~70x in sweep time.

Detection: energies below a crude variational floor (any per-site classical
bound times the site count) in the per-sweep energy records, or non-monotone
E0 across a coupling sweep.
A Ritz value below the true ground energy is *mathematically impossible*
with an orthonormal Krylov basis over a valid H_eff, so any such value is a
smoking gun for a broken invariant, not a convergence issue.

## Root cause (two rounds)

**Round 1 — small-beta breakdown (the textbook one).** When the start
vector is a near-exact eigenvector, the orthogonalized residual w is pure
cancellation noise: beta = ||w|| ~ eps * ||H||. Dividing by such a beta
yields a "basis vector" with O(1) overlap against the existing basis, and
eigh of the resulting non-orthonormal projection is unbounded below. An
absolute threshold (beta < 1e-13) misses this whenever ||H|| is large —
for ||H_eff|| ~ 300 the noise floor *is* ~1e-13. A scale-aware threshold
(beta < 1e-10 * max|alpha|) fixed the cases we caught. It was a near-miss.

**Round 2 — duplicate insertion at Krylov exhaustion (the one that bit).**
The original implementation orthogonalized with a *single* batched
classical Gram-Schmidt pass. At Krylov exhaustion the subtraction is
catastrophically cancelling, and the leftover points *along the cached
directions*; normalizing it re-inserts an old direction as a **duplicate
basis vector**. From then on each batched projection subtracts that
direction once per duplicate copy — over-subtracting by one copy per
iteration — so each new residual is again the same old vector and **beta
grows by |alpha| every iteration** instead of vanishing. Instrumented
reproduction (deterministic to the last digit):

```
a = [-540.0, -525.0, ..., -524.53, -524.53, -524.53, ...]   (frozen)
b = [1.4, 2.0, ..., 54, 535, 1062, 1587, 2111, 2636, ...]   (steps of |alpha|)
|V V^dag - I|_max = 1.000                                    (exact duplicate)
E = -7590   on a spectrum bounded below by -542.4
```

No small-beta guard can catch this: beta *explodes* rather than vanishing.
The isometry of the truncation tensors was verified intact throughout
which is what localized the bug to the Krylov loop.

## The fix: tenpy's guards, verbatim

1. **Per-iteration convergence via the Ritz residual** (tenpy's `P_tol`):
   `P_err = (|v0[k]| * beta_k / gap)^2 < P_tol` with
   `gap = max(E1 - E0, min_gap)`, checked from iteration 2 (`N_min = 2`).
   This is the guard that actually matters: at exhaustion the newest
   vector's weight in the Ritz ground state collapses, so the loop exits
   *before* the instability can build. tenpy never encounters the
   pathological regime because it leaves early.
2. **Breakdown cutoff** `beta < 100 * eps` (tenpy) plus the scale-aware
   `beta < 1e-10 * max|alpha|` from round 1 (belt and suspenders).
3. **CGS2 reorthogonalization** — the batched Gram-Schmidt projection is
   applied **twice** ("twice is enough", Kahan–Parlett). Two batched GEMV
   pairs per iteration; the algebraic equivalent of tenpy's sequential
   modified-GS `reortho` loop, and it removes the duplicate-insertion
   mechanism outright.

Rules of thumb if you modify this code:

- Never single-pass the orthogonalization, no matter how tempting the
  saved GEMV is.
- Never let the loop run past the P_tol exit "for safety" — extra
  iterations at exhaustion are where the damage happens.
- Any "impossible" energy (below the variational floor) is an
  orthonormality violation somewhere. Do not tune tolerances around it;
  instrument and find the broken invariant (`TENPY_GPU_ADDON_DEBUG=1` enables
  per-update isometry checks and Lanczos internals dumps).

## Verification and regression

- Reproduction case (private research repo): the exact failing production
  configuration — a 6x12 classical anchor deep in the ordered phase, mixer
  off at sweep 15. Post-fix: zero garbage, E flat at -542.3985691 through
  the mixer-off sweeps, agrees with the independent forward-chain value to
  3e-5, anchors relax to their natural chi ~ 70.
- Unit regression (this repo): `tests/test_lanczos.py` — exactly solvable
  H_eff at production energy scale, near-eigenvector starts; pre-fix it
  reproduced E = -5236 on a spectrum bounded at -324.

## Why the CPU/tenpy workflow never saw this

Stock tenpy DMRG uses this same algorithm (that is where it was ported
from); its guards exit the iteration in exactly the situations that broke
our loop. The failure was ours, introduced with the custom GPU Lanczos —
not a property of GPUs, chi, or system size (those only control how often
the trigger state occurs).
