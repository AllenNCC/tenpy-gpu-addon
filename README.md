# tenpy-gpu-addon

GPU acceleration for [TeNPy](https://github.com/tenpy/tenpy) DMRG — **without
touching TeNPy**. Your model, lattice, and observable code stays in tenpy;
this package consumes them through tenpy's public API and moves the expensive
contractions (the chi³ that dominate DMRG) onto a GPU with PyTorch.

```python
from tenpy_gpu_addon import run_dmrg_torch

E, psi, info = run_dmrg_torch(my_tenpy_model, chi_max=1024, device='cuda')
# psi is a regular tenpy MPS — all your downstream code works unchanged
```

Measured on a 36-site cylinder model (MPO bond dimension 14): the full GPU
loop on an **A100 is 33–51× a 16-core CPU node** (growing with chi; 9.2
s/sweep at chi=512, 47 s at chi=1024), and an **H200 adds another 2.3–2.6×**
with no code changes (~118× vs CPU at chi=512). GPU measurement is ~46× per
state at chi=1024. For the full benchmark behind these numbers, see
[bench/RESULTS.md](bench/RESULTS.md).

## Install

Install the package either from a clone of this
repository:

```bash
git clone https://github.com/<user>/tenpy-gpu-addon

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

## The three entry points

| entry point | what it is | when to use it |
|---|---|---|
| `run_dmrg_torch(model, ...)` | fully GPU-resident two-site DMRG loop: density-matrix mixer (White 2005), sweep-indexed `chi_list` ramp with per-step mixer re-arm, tenpy-compatible convergence stopping | **production** — fastest by 6–9× over the drop-in engine |
| `TorchTwoSiteDMRGEngine` | tenpy's stock `TwoSiteDMRGEngine` with only the per-bond Lanczos on the GPU | when you need tenpy features the full loop lacks (iDMRG, charged models, other mixers) — numerically identical to stock tenpy (1e-14) |
| `DeviceMPS` | tenpy-API-compatible measurement (`expectation_value`, `correlation_function`, `overlap`, `entanglement_entropy`) on the GPU | measuring large-chi states; your observable code runs unchanged on it by duck typing |

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

## Example: the square-lattice quantum dimer model

`examples/qdm_model.py` is a complete, realistic model — the 4-state
vertex-site square-lattice QDM on a cylinder (plaquette ring-exchange via
two-site couplings, perfect-matching constraint via a soft blockade) — and
`examples/run_qdm.py` runs the whole pipeline on it:

```bash
python examples/run_qdm.py                                    # laptop-sized
python examples/run_qdm.py --Lx 6 --Ny 6 --chi 512 --device cuda
```

The model file imports only tenpy; the GPU is attached entirely from the
outside. Use it as a template for wiring up your own model.

## Validation

`pytest tests -v` — the trust chain is: **stock CPU tenpy is the oracle.**

- `run_dmrg_torch` converged energies match stock tenpy to ~1e-10 relative
  with wavefunction overlap at ~1e-8 (TFI chain; QDM example energy to 1e-8).
- `TorchTwoSiteDMRGEngine` reproduces the stock engine to ~1e-12 (it *is*
  the stock engine except for the Lanczos arithmetic).
- `DeviceMPS` reproduces every tenpy measurement to ~1e-10 on two different
  models (spin-1/2 TFI and the 4-state QDM), including duck-typed user
  observable code.
- A regression test pins the Lanczos near-breakdown guard (see
  `docs/lanczos-stability.md` for the post-mortem that produced it).

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

The drop-in engine draws the boundary differently — tenpy keeps the whole
engine and only the Lanczos matvecs cross to the GPU, paying a host↔device
round trip per bond. That boundary placement is the entire 6–9× gap between
the two: in the CPU profile ~78% of sweep time is chi³ GEMMs, and Amdahl
does the rest.

## The mixer

`run_dmrg_torch`'s mixer is a density-matrix mixer in the sense of White
(2005) — the same family as tenpy's `DensityMatrixMixer` (tenpy's default for
two-site DMRG), with the same default schedule (`amplitude=1e-5`, `decay=2.0`,
`disable_after=15`). It differs from stock tenpy in three ways; only the drop-in
`TorchTwoSiteDMRGEngine` uses tenpy's mixer verbatim.

1. **Perturbation formula.** tenpy builds the mixed density matrix as a single
   MPO-channel-weighted sum: the "identity-left" channel at weight 1 (which
   reproduces the true ρ), the "identity-right" channel zeroed, and every other
   ("operator-in-flight") channel at the amplitude `a`, un-normalized. The addon
   instead forms the true ρ = θθ† and *adds* a trace-normalized perturbation
   summed over **all** MPO channels:

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
   tenpy. (Benchmark: `bench/RESULTS.md` § "GPU truncation routes through
   rho-eigh".)

**Cold-start caveat (shared with tenpy, not specific to this mixer).** A
density-matrix mixer perturbs ρ by `h·θ` terms, so if the initial product state
is a *dark state* of the off-diagonal Hamiltonian (`h·θ = 0`) there is nothing to
inject and *no* such mixer can escape it. The QDM's natural product states are
dark states of the plaquette kinetic term, so a cold start stalls at E=0 with chi
pinned at 1 — in stock tenpy exactly as here. Seed with a warm start (a couple of
stock tenpy sweeps, or a random-bond-dimension MPS) as `examples/run_qdm.py`
does. This is a property of the model, not the engine.

## Layout

- `tenpy_gpu_addon/dense.py` — tenpy MPO/MPS ↔ dense tensors (public API only)
- `tenpy_gpu_addon/dmrg.py` — the fully GPU-resident two-site DMRG loop
- `tenpy_gpu_addon/engine.py` — GPU Lanczos inside the stock tenpy engine
- `tenpy_gpu_addon/measure.py` — `DeviceMPS` GPU measurement
- `examples/` — the QDM model + runnable demo
- `tests/` — the validation suite described above
- `docs/lanczos-stability.md` — why the inner Lanczos mirrors tenpy's guards
  exactly (read before touching `_lanczos`)

## License

MIT
