"""Converged-accuracy trust chain: stock CPU tenpy is the oracle.

Both GPU-facing engines must reproduce the stock `TwoSiteDMRGEngine` ground
state on charge-free models:

- `run_dmrg_torch` (the fully torch-resident loop) — converged energy to
  ~1e-10 relative, wavefunction overlap to ~1e-8, on a model with a unique
  gapped ground state (TFI chain, paramagnetic side).
- `TorchTwoSiteDMRGEngine` (GPU Lanczos inside the stock engine) —
  numerically the stock engine; energies agree to ~1e-12.

The QDM example model is exercised too (energy only: its ordered phase has
a quasi-degenerate manifold, so state-resolved quantities are path-dependent
at fixed chi — see the README).

The CPU tier runs the same code paths with device='cpu'; gpu-marked tests
repeat on CUDA.
"""
import numpy as np
import pytest

pytest.importorskip('torch')

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain

from tenpy_gpu_addon import run_dmrg_torch, TorchTwoSiteDMRGEngine
from tenpy_gpu_addon.dense import ucw_kwargs

from qdm_model import SquareQDM


def _tfi(L=12, g=1.5):
    return TFIChain({'L': L, 'J': 1.0, 'g': g, 'bc_MPS': 'finite',
                     'conserve': None})


def _product_psi(model):
    sites = model.lat.mps_sites()
    return MPS.from_product_state(sites, [0] * len(sites), bc='finite',
                                  **ucw_kwargs(model.lat))


def _reference(model, chi=32, sweeps=12):
    psi = _product_psi(model)
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True, 'min_sweeps': sweeps, 'max_sweeps': sweeps,
        'combine': False,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10}})
    E, psi = eng.run()
    psi.canonical_form()
    return E, psi


def _run_torch(model, chi=32, device='cpu'):
    return run_dmrg_torch(model, chi_max=chi,
                          mixer_params={'disable_after': 4},
                          max_sweeps=30, device=device)


def test_run_dmrg_torch_vs_tenpy_tfi():
    model = _tfi()
    E_ref, psi_ref = _reference(model)
    E, psi, info = _run_torch(model)
    assert info['converged']
    assert abs(E - E_ref) / abs(E_ref) < 1e-10
    assert abs(abs(psi.overlap(psi_ref)) - 1.0) < 1e-8


def test_run_dmrg_torch_vs_tenpy_qdm():
    """Warm-seeded on purpose: the all-'1' product state is a zero-energy
    dark state of the plaquette kinetic term (Lanczos exits immediately on an
    exact eigenstate), so a cold start stalls at E=0 — a property of the
    model, not the engine; production QDM sweeps warm-start for the same
    reason. Two stock sweeps break the symmetry, then the torch loop must
    converge to the oracle."""
    model = SquareQDM(dict(Lx=4, Ny=2, bc_MPS='finite',
                           t=1.0, V=0.0, V_blockade=50.0))
    E_ref, _ = _reference(model, chi=64, sweeps=14)
    psi_seed = _product_psi(model)
    eng = dmrg.TwoSiteDMRGEngine(psi_seed, model, {
        'mixer': True, 'min_sweeps': 2, 'max_sweeps': 2, 'combine': False,
        'trunc_params': {'chi_max': 16, 'svd_min': 1e-10}})
    _, psi_seed = eng.run()
    psi_seed.canonical_form()
    E, _, info = run_dmrg_torch(model, psi0=psi_seed, chi_max=64,
                                mixer_params={'disable_after': 6},
                                max_sweeps=40, device='cpu')
    assert info['converged']
    assert abs(E - E_ref) / abs(E_ref) < 1e-8


def test_drop_in_engine_vs_tenpy():
    """TorchTwoSiteDMRGEngine is numerically the stock engine."""
    model = _tfi()
    params = {'mixer': True, 'min_sweeps': 10, 'max_sweeps': 10,
              'combine': False,
              'trunc_params': {'chi_max': 32, 'svd_min': 1e-10}}
    E_ref, psi_ref = _reference(model, chi=32, sweeps=10)
    psi = _product_psi(model)
    eng = TorchTwoSiteDMRGEngine(psi, model,
                                 {**params, 'torch_device': 'cpu'})
    E, psi = eng.run()
    psi.canonical_form()
    assert abs(E - E_ref) / abs(E_ref) < 1e-12
    assert abs(abs(psi.overlap(psi_ref)) - 1.0) < 1e-10


@pytest.mark.gpu
def test_run_dmrg_torch_cuda():
    model = _tfi()
    E_ref, psi_ref = _reference(model)
    E, psi, info = _run_torch(model, device='cuda')
    assert info['converged']
    assert abs(E - E_ref) / abs(E_ref) < 1e-10
    assert abs(abs(psi.overlap(psi_ref)) - 1.0) < 1e-8


@pytest.mark.gpu
def test_drop_in_engine_cuda():
    model = _tfi()
    E_ref, _ = _reference(model, chi=32, sweeps=10)
    psi = _product_psi(model)
    eng = TorchTwoSiteDMRGEngine(psi, model, {
        'mixer': True, 'min_sweeps': 10, 'max_sweeps': 10, 'combine': False,
        'trunc_params': {'chi_max': 32, 'svd_min': 1e-10},
        'torch_device': 'cuda'})
    E, _ = eng.run()
    assert abs(E - E_ref) / abs(E_ref) < 1e-12
