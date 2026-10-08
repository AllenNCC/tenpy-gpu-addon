"""TorchTDVP against two oracles: stock tenpy TwoSiteTDVPEngine and exact
time evolution.

On a chain small enough that chi_max never truncates, two-site TDVP is exact
up to its O(dt^3) splitting error, and the torch engine must agree with the
stock engine step for step (same integrator) to ~1e-10. The gpu-marked test
repeats the comparison on CUDA (eigh-based split instead of SVD).
"""
import numpy as np
import pytest

pytest.importorskip('torch')

from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

from tenpy_gpu_addon import TorchTDVP
from tenpy_gpu_addon.dense import ucw_kwargs


def _model(L=8):
    # Neel start in a transverse field: non-trivial, entangling dynamics
    return TFIChain({'L': L, 'J': 1.0, 'g': 1.2, 'bc_MPS': 'finite',
                     'conserve': None})


def _neel(model):
    sites = model.lat.mps_sites()
    return MPS.from_product_state(sites, ['up', 'down'] * (len(sites) // 2),
                                  bc='finite', **ucw_kwargs(model.lat))


def _sz(psi):
    return np.real(psi.expectation_value('Sigmaz'))


def _run_torch(model, psi, n, dt, device):
    eng = TorchTDVP(model, psi, chi_max=64, device=device)
    for _ in range(n):
        eng.sweep(dt)
    return eng


def _check_against_stock(device):
    model = _model()
    psi = _neel(model)
    n, dt = 10, 0.05
    stock = TwoSiteTDVPEngine(psi.copy(), model, {
        'trunc_params': {'chi_max': 64, 'svd_min': 1e-12},
        'dt': dt, 'N_steps': n})
    stock.run()
    eng = _run_torch(model, psi, n, dt, device)
    out = eng.to_tenpy()
    assert abs(abs(out.overlap(stock.psi)) - 1) < 1e-9
    assert np.allclose(_sz(out), _sz(stock.psi), atol=1e-9)


def test_matches_stock_tdvp_cpu():
    _check_against_stock('cpu')


@pytest.mark.gpu
def test_matches_stock_tdvp_gpu():
    _check_against_stock('cuda')


def test_exact_dynamics_and_energy_conservation():
    model = _model()
    psi = _neel(model)
    T, dt = 1.0, 0.02
    eng = _run_torch(model, psi, int(round(T / dt)), dt, 'cpu')
    # exact: ED in the full Hilbert space
    ed = ExactDiag.from_H_mpo(model.H_MPO)
    ed.build_full_H_from_mpo()
    ed.full_diagonalization()
    vt = ed.exp_H(T).to_ndarray() @ ed.mps_to_full(psi).to_ndarray()
    vtdvp = ed.mps_to_full(eng.to_tenpy()).to_ndarray()
    assert abs(abs(np.vdot(vt, vtdvp)) - 1) < 1e-6
    E0 = model.H_MPO.expectation_value(psi)
    assert abs(model.H_MPO.expectation_value(eng.to_tenpy()) - E0) < 1e-10


def test_imaginary_time_reaches_ground_state():
    model = TFIChain({'L': 10, 'J': 1.0, 'g': 1.5, 'bc_MPS': 'finite',
                      'conserve': None})
    # all-up: same Z2 (prod Sigmaz) sector as the ground state -- a Neel start
    # on 10 sites is parity-odd and correctly converges to E_1 instead
    sites = model.lat.mps_sites()
    psi = MPS.from_product_state(sites, ['up'] * len(sites), bc='finite',
                                 **ucw_kwargs(model.lat))
    eng = TorchTDVP(model, psi, chi_max=32, device='cpu')
    for _ in range(40):
        eng.sweep(0.25, imaginary=True)
    ed = ExactDiag.from_H_mpo(model.H_MPO)
    ed.build_full_H_from_mpo()
    ed.full_diagonalization()
    assert abs(model.H_MPO.expectation_value(eng.to_tenpy()) - ed.E.min()) < 1e-8


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    eng = _run_torch(model, _neel(model), 5, 0.05, 'cpu')
    eng.save(tmp_path / 'ck.npz')
    eng2 = TorchTDVP.from_checkpoint(model, tmp_path / 'ck.npz', chi_max=64,
                                     device='cpu')
    assert eng2.t == pytest.approx(eng.t)
    for _ in range(3):
        eng.sweep(0.05)
        eng2.sweep(0.05)
    assert abs(abs(eng.to_tenpy().overlap(eng2.to_tenpy())) - 1) < 1e-12
