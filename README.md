# tenpy-gpu-addon

GPU acceleration for [TeNPy](https://github.com/tenpy/tenpy) DMRG and TDVP
time evolution — **without touching TeNPy**. Your model, lattice, and
observable code stays in tenpy; this package consumes them through tenpy's
public API and moves the expensive contractions (the chi³ that dominate both
algorithms) onto a GPU with PyTorch.

```python
from tenpy_gpu_addon import run_dmrg_torch, TorchTDVP

E, psi, info = run_dmrg_torch(my_tenpy_model, chi_max=1024, device='cuda')
# psi is a regular tenpy MPS — all your downstream code works unchanged

eng = TorchTDVP(my_tenpy_model, psi0, chi_max=1024, device='cuda')
for _ in range(1000):
    eng.sweep(0.1)                      # real time; imaginary=True for exp(-dt H)
psi_t = eng.to_tenpy()
```

DMRG, measured on a 36-site cylinder model (MPO bond dimension 14): the full GPU
loop on an **A100 is 33–51× a 16-core CPU node** (growing with chi; 9.2
s/sweep at chi=512, 47 s at chi=1024), and an **H200 adds another 2.3–2.6×**
with no code changes (~118× vs CPU at chi=512). GPU measurement is ~46× per
state at chi=1024. For the full benchmark behind these numbers, see
[bench/README.md](bench/README.md).

TDVP time evolution, measured on a 30-site spin-1 quench (MPO bond dimension
14): an **A100 is 29–30× TeNPy on 16 CPU cores** at chi = 512–1024 (34 s vs
1009 s per sweep at chi = 1024), an **H200 67–71×**; see
[tdvp_bench/README.md](tdvp_bench/README.md).

## Install

Install the package either from a clone of this
repository:

```bash
git clone https://github.com/AllenNCC/tenpy-gpu-addon

cd tenpy-gpu-addon

pip install .
```

or from a local download:

```bash
pip install ./tenpy-gpu-addon
```

Everything falls back to CPU when CUDA is absent. Dependencies: `numpy`, `physics-tenpy>=1.0`, and `torch`.

## Not yet implemented / Requirements on your model

 - **Charge-conservation**: the addon currently doesn't take advantage of charge conservation yet, thus your model need to be built without charge conservation (`conserve=None` in most stock tenpy models). Every npc tensor is then a single dense block, which is what makes the dense export legal — this is
  checked loudly at export time.
- **Infinite MPS**: `run_dmrg_torch` and `DeviceMPS` (measurement) doesn't support iDMRG for now, but   `TorchTwoSiteDMRGEngine` does. It should be straightforward though. 
- **TDVP scope**: `TorchTDVP` is finite-MPS, two-site, with a time-independent
  MPO between `set_model` calls (a quench = one `set_model`; a smooth drive
  would need one per step). No single-site TDVP yet.

## The four entry points

| entry point | what it is | when to use it |
|---|---|---|
| `run_dmrg_torch(model, ...)` | fully GPU-resident two-site DMRG loop: density-matrix mixer (White 2005), sweep-indexed `chi_list` ramp with per-step mixer re-arm, tenpy-compatible convergence stopping | **production** — fastest by 6–9× over the drop-in engine |
| `TorchTwoSiteDMRGEngine` | tenpy's stock `TwoSiteDMRGEngine` with only the per-bond Lanczos on the GPU | when you need tenpy features the full loop lacks (iDMRG, charged models, other mixers) — numerically identical to stock tenpy (1e-14) |
| `DeviceMPS` | tenpy-API-compatible measurement (`expectation_value`, `correlation_function`, `overlap`, `entanglement_entropy`) on the GPU | measuring large-chi states; your observable code runs unchanged on it by duck typing |
| `TorchTDVP(model, psi, ...)` | fully GPU-resident two-site TDVP (real or imaginary time), sharing the DMRG loop's environment and effective-Hamiltonian kernels; checkpoint/resume | time evolution at large chi — quenches, transport, finite-energy-density dynamics |

### Full GPU loop

```python
from tenpy_gpu_addon import run_dmrg_torch

# production cold start: chi ramp + re-armed mixer + tenpy-style stopping
E, psi, info = run_dmrg_torch(model, chi_max=1024,
                              chi_list={0: 256, 5: 512, 10: 1024},
                              mixer_params={'disable_after': 5},
                              max_sweeps=40, device='cuda')
```

Parameters mirror the tenpy engine where they exist (`chi_list`,
`max_E_err`, `max_S_err`, `min_sweeps`/`max_sweeps`, `mixer_params`). The
mixer is a density-matrix mixer that differs from stock tenpy's in a few
deliberate ways — most visibly, its amplitude is **re-armed at each `chi_list`
step**; see [The mixer](#the-mixer). `psi` comes back as a genuine tenpy
`MPS`; `info` carries per-sweep energies, entropies, timings, truncation
errors, and a `converged` flag.

### Drop-in engine

```python
from tenpy_gpu_addon import TorchTwoSiteDMRGEngine

eng = TorchTwoSiteDMRGEngine(psi, model, {**dmrg_params,
                                          'torch_device': 'cuda'})
E, psi = eng.run()      # mixer, chi_list, convergence: all stock tenpy
```

Scope: `combine=False` and default Lanczos options only.

### GPU measurement

```python
from tenpy_gpu_addon import DeviceMPS

m = DeviceMPS.from_tenpy(psi, device='cuda')
dens = m.expectation_value('n1')                      # tenpy signatures
C = m.correlation_function('n1', 'n2')                # batched chi^3 GEMMs
# any observable function written against the tenpy MPS API works on `m`
```

Scope: finite, charge-free MPS; bosonic operators (anything needing a
Jordan–Wigner string raises — use tenpy's CPU path for those).

### Time evolution (TDVP)

```python
from tenpy_gpu_addon import TorchTDVP

eng = TorchTDVP(model, psi0, chi_max=1024, svd_min=1e-10, device='cuda')
for step in range(n_steps):
    eng.sweep(dt)                                   # exp(-i dt H), 2nd order
    h = eng.local_expectations(ops)                 # [(first_site, dense op)]
    if step % 50 == 0:
        eng.save('state.npz')                       # resume: TorchTDVP.from_checkpoint
psi = eng.to_tenpy()
```

One `sweep(dt)` is the symmetric second-order two-site TDVP integrator
(Haegeman et al., PRB 94, 165116 (2016)): every bond is evolved forward by
dt/2 with the two-site effective Hamiltonian on the way right and again on
the way back, every intermediate centre site backward with the one-site one.
`sweep(dt, imaginary=True)` applies exp(-dt H) and renormalises (ground
states, or finite-temperature state preparation). `eng.set_model(new_model)`
swaps the Hamiltonian in place — prepare with H₀, `set_model(H₁)`, evolve:
that is a quench. `eng.trunc_err` (discarded weight of the last sweep) and
`eng.max_chi` are the per-step diagnostics; `local_expectations` is exact for
the current state (it contracts left transfer matrices rather than reusing
Schmidt values from the sweep, which are one half-step stale in TDVP).
`svd_min` is relative to the largest Schmidt value; on CUDA it is effectively
`max(svd_min, 1e-7)`, the resolution of the `eigh` truncation (see
[The mixer](#the-mixer), item 3).

## Example: the square-lattice quantum dimer model

`examples/qdm_model.py` is the square-lattice QDM on a cylinder (plaquette ring-exchange via
two-site couplings, gauge constraint via a soft blockade) — and
`examples/run_qdm.py` runs the whole pipeline on it:

```bash
python examples/run_qdm.py                                    # laptop-sized
python examples/run_qdm.py --Lx 6 --Ny 6 --chi 512 --device cuda
```

The model file imports only tenpy; the GPU is attached entirely from the
outside. Use it as a template for wiring up your own model.

## Example: a conformal-cooling quench (TDVP)

`examples/spin1_chain.py` is a spin-1 bilinear-biquadratic chain (Haldane
phase) whose energy scale can vary bond by bond, `H = J Σ λ_x [S·S + (γ/3)(S·S)²]`;
`examples/run_tdvp_quench.py` reproduces the quench of Fig. 2 of Zaletel,
Kaufman, Stamper-Kurn & Yao, [arXiv:1611.04591](https://arxiv.org/abs/1611.04591):
prepare `exp(-τH)|Néel⟩` with stock tenpy (imaginary-time `ExpMPOEvolution`,
T ≈ 0.5 J), reduce λ to 0.17 on a "bath" region, and watch heat drain out of
the λ = 1 "system" region with GPU TDVP.

```bash
python examples/run_tdvp_quench.py                              # laptop: L=16, chi=64
python examples/run_tdvp_quench.py --L 30 --chi 1024 --tmax 100 --device cuda
```

The paper's authors report that their run to t = 100/J took several weeks of
computing time. At chi = 1024 that is 1000 sweeps. Measured per-sweep times
put it at 12–26 days with TeNPy on 16 CPU cores, 10–21 h on an A100 and
4–9 h on an H200; the upper ends are the realistic ones for a full run (the
range is explained in [tdvp_bench/README.md](tdvp_bench/README.md)).

## Validation

`pytest tests -v`

- `run_dmrg_torch` converged energies match stock tenpy to ~1e-10 relative
  with wavefunction overlap at ~1e-8 (TFI chain; QDM example energy to 1e-8).
- `TorchTwoSiteDMRGEngine` reproduces the stock engine to ~1e-12 (it *is*
  the stock engine except for the Lanczos arithmetic).
- `DeviceMPS` reproduces every tenpy measurement to ~1e-10 on two different
  models (spin-1/2 TFI and the 4-state QDM), including duck-typed user
  observable code.
- A regression test pins the Lanczos near-breakdown guard (see
  `docs/lanczos-stability.md` for the post-mortem that produced it).
- `TorchTDVP` matches stock tenpy's `TwoSiteTDVPEngine` step for step (state
  overlap 1 − 1e-9, local observables to 1e-9; CPU and CUDA), matches exact
  time evolution, conserves energy to ~1e-10 when nothing is truncated,
  reaches the ED ground state in imaginary time, and resumes from a
  checkpoint on the same trajectory (overlap 1 − 1e-12).

```bash
pytest tests -v -m "not gpu"    # CPU tier — any machine, no CUDA needed
pytest tests -v                 # + CUDA-vs-CPU-oracle tests on a GPU node
```

Two physics caveats when comparing engines yourself: entanglement entropy
has a reproducibility floor set by Schmidt-tail truncation paths, so
cross-engine agreement at fixed chi is bounded by the truncation error; and
in phases with a (quasi-)degenerate ground manifold, state-resolved
observables are engine/path-dependent at fixed chi — energy and the
variational bound are the robust cross-engine assertions.

## How it works

**tenpy does the physics setup at the start and gets the answer back at the
end; the GPU replaces only the middle — the DMRG engine itself.**

1. Your model compiles its Hamiltonian into `model.H_MPO` (stock tenpy, CPU).
2. `export_mpo`/`export_mps` unwrap each npc array into a dense ndarray
   (legal because the model is charge-free — asserted), and one
   `torch.as_tensor(..., device='cuda')` per tensor uploads them. This is
   the last host→device transfer until the run ends.
3. The sweep loop runs entirely in torch: Lanczos on the effective
   Hamiltonian (a faithful port of tenpy's `LanczosGroundState`, including
   its convergence and breakdown guards), truncation via the reduced density
   matrix (on CUDA, `eigh` beats every cuSOLVER SVD driver by 8–40× at these
   shapes), environment updates.
4. The site tensors come back once at the end and `import_mps` wraps them
   into a genuine tenpy `MPS`.

TDVP reuses steps 1, 2 and 4 and the same environment/matvec kernels; step 3
becomes a Krylov action of exp(a·H_eff) — Lanczos with the same batched CGS2
reorthogonalisation, converged when the weight the next Krylov vector would
receive drops below 1e-12 — followed by the same `eigh`-based truncation on
CUDA (SVD on CPU).

The drop-in engine draws the boundary differently — tenpy keeps the whole
engine and only the Lanczos matvecs cross to the GPU, paying a host↔device
round trip per bond. That boundary placement is the entire 6–9× gap between
the two: in the CPU profile ~78% of sweep time is chi³ GEMMs, and Amdahl
does the rest.

## The mixer

`run_dmrg_torch`'s mixer is a density-matrix mixer in the sense of White
(2005) — the same family as tenpy's `DensityMatrixMixer` (tenpy's default for
two-site DMRG), with the same default schedule (`amplitude=1e-5`, `decay=2.0`,
`disable_after=15`). The drop-in
`TorchTwoSiteDMRGEngine` uses tenpy's mixer verbatim.

1. **Perturbation formula.** tenpy builds the mixed density matrix as a single
   MPO-channel-weighted sum: the "identity-left" channel at weight 1 (which
   reproduces the true ρ), the "identity-right" channel zeroed, and every other
   ("operator-in-flight") channel at the amplitude `a`, un-normalized. The addon
   instead forms the true ρ = θθ† and *adds* a trace-normalized perturbation
   summed over all MPO channels:

   $$
   \rho_L \longrightarrow \theta\theta^\dagger + \frac{a}{\mathrm{Tr}}\sum_w (L W \theta)_w^{\dagger} (L W \theta)_w^\dagger
   $$

   Two consequences: (a) it perturbs along a *superset* of directions — it keeps
   the identity-right channel tenpy drops; and (b) because the perturbation is
   trace-normalized across all channels, each bond-crossing escape direction is
   scaled by `a/Tr` rather than a raw `a`. `Tr` grows with MPO bond dimension
   (~18 for the QDM example even at 4×4, larger in production), so a *single*
   mixer sweep pushes toward new Schmidt states less aggressively than tenpy's.
   The re-arm below is the intended compensation.

2. **Re-arm at every `chi_list` step.** The amplitude decays from full `a` again
   at each chi step, instead of tenpy's monotone decay-and-disable from sweep 0.
   Across a chi ramp this delivers a fresh full-amplitude kick at each new chi —
   which is what keeps the per-sweep-weaker perturbation from mattering in
   production ramps, and lets small-chi stages act as branch selectors in ordered
   phases rather than inheriting the full-chi start's manifold mixture.

3. **Truncation primitive — one engine, not two.** tenpy *switches* its
   decomposition by mixer state: it diagonalizes the (perturbed) density matrix
   with `eigh` when the mixer is on, and falls back to an SVD of `theta` when the
   mixer is off. The addon keeps the density-matrix `eigh` for **both** on CUDA,
   because cuSOLVER's dense fp64 SVD is far slower than `eigh` at the square
   shapes truncation produces. In an A100 kernel shootout at `chi·d = 5120`, the
   SVD took **14.3 s** vs **0.35 s** for the `eigh` (~40×), which had made
   mixer-off sweeps **3.6× slower** than mixer-on ones; routing all CUDA
   truncation through `eigh` removes that penalty. The cost is the Gram-matrix
   √ε Schmidt floor (~1e-8) instead of SVD's ~1e-16, with energy/observable
   impact ~1e-16 — the same floor the mixer-on path always carried. On CPU,
   LAPACK SVD is fast and exact, so mixer-off truncation there uses SVD like
   tenpy. (Benchmark: `bench/README.md` § "GPU truncation routes through
   rho-eigh".)

   **When the eigh floor matters (DMRG and TDVP alike).** `eigh` returns the
   eigenvalues $s_k^2$ of $\rho = \theta\theta^\dagger$ with absolute error
   $\sim \epsilon \lVert\rho\rVert \approx 10^{-16}$, so Schmidt values below
   $\sim 10^{-8} s_0$ are noise. Note that this does not change the state's form:
   the kept eigenvectors are orthonormal to machine precision, the Schmidt
   values are absorbed into the other factor, so canonical form is exact.
   - *Harmless:* whenever `chi_max` binds — the production regime. Discarded
     weights there are $10^{-7}$–$10^{-6}$ per TDVP sweep at chi = 1024
     (well above $10^{-16}$), so `trunc_err` is meaningful and energies and
     local observables are unaffected; entanglement entropy picks up
     $O(10^{-16})$.
   - *Not resolved:* the deep entanglement-spectrum tail ($s_k < 10^{-8}
     s_0$). Use the CPU path (SVD) if you need it.
   - *On the `eigh` path, `svd_min` is floored at $10^{-7}$ (relative to
     $s_0$).* The null space of $\rho$ comes back as $\pm$ noise of
     $s/s_0 \approx 2$–$3.4\times10^{-8}$ (growing slowly with the matrix
     size, measured for $n$ = 200–3000). A cut below that, such as the
     default `svd_min**2` = $10^{-20}$, would keep the positive half of the
     noise. Then, while `chi_max` does not bind (early times in a quench,
     small DMRG stages), chi would grow several times beyond what the state
     needs: for a rank-50 $600\times600$ test matrix it kept 322 directions.
     The floor (`_EIGH_SCHMIDT_FLOOR` in `dmrg.py`, ~3× above the noise at
     $n$ = 3000) drops only directions of weight $< 10^{-14}$. Once
     `chi_max` binds it never matters.

## Layout

- `tenpy_gpu_addon/dense.py` — tenpy MPO/MPS ↔ dense tensors (public API only)
- `tenpy_gpu_addon/dmrg.py` — the fully GPU-resident two-site DMRG loop
- `tenpy_gpu_addon/engine.py` — GPU Lanczos inside the stock tenpy engine
- `tenpy_gpu_addon/measure.py` — `DeviceMPS` GPU measurement
- `tenpy_gpu_addon/tdvp.py` — `TorchTDVP` two-site TDVP time evolution
- `examples/` — the QDM model + DMRG demo; the spin-1 chain + TDVP quench demo
- `tests/` — the validation suite described above
- `bench/` — DMRG timing benchmarks
- `tdvp_bench/` — TDVP timing benchmarks

## License

MIT
