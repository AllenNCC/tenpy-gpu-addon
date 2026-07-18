"""DeviceMPS (measure.py) vs the tenpy measurement oracle.

Trust chain: tenpy's own measurement methods are the reference; DeviceMPS
must reproduce them to contraction-order noise (~1e-10) on the same states.
The CPU tier runs everything with device='cpu' (same code path, no GPU
needed); the gpu-marked test repeats the comparison on CUDA.

Two models are used so the layer is provably not specialized to one site:
a stock tenpy transverse-field Ising chain (spin-1/2, charge conservation
off) and the 4-state QDM example model.
"""
import numpy as np
import pytest

pytest.importorskip('torch')

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain

from tenpy_gpu_addon.dense import ucw_kwargs
from tenpy_gpu_addon.measure import DeviceMPS

from qdm_model import SquareQDM, structure_factors, dimer_density

ATOL = 1e-10


def _ground_state(model, chi, sweeps=8):
    psi = MPS.from_product_state(model.lat.mps_sites(),
                                 [0] * len(model.lat.mps_sites()), bc='finite',
                                 **ucw_kwargs(model.lat))
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True, 'min_sweeps': sweeps, 'max_sweeps': sweeps,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10}})
    eng.run()
    psi.canonical_form()
    return psi


@pytest.fixture(scope='module')
def tfi_state():
    model = TFIChain({'L': 8, 'J': 1.0, 'g': 1.2, 'bc_MPS': 'finite',
                      'conserve': None})
    return model, _ground_state(model, chi=16, sweeps=6)


@pytest.fixture(scope='module')
def qdm_state():
    model = SquareQDM(dict(Lx=4, Ny=2, bc_MPS='finite',
                           t=1.0, V=0.0, V_blockade=50.0))
    return model, _ground_state(model, chi=32)


def test_expectation_value(tfi_state, qdm_state):
    _, psi = tfi_state
    m = DeviceMPS.from_tenpy(psi, device='cpu')
    for op in ('Sx', 'Sz'):
        assert np.allclose(m.expectation_value(op),
                           psi.expectation_value(op), atol=ATOL)
    _, psi = qdm_state
    m = DeviceMPS.from_tenpy(psi, device='cpu')
    for k in range(1, 5):
        assert np.allclose(m.expectation_value(f'n{k}'),
                           psi.expectation_value(f'n{k}'), atol=ATOL)


def test_correlation_function(tfi_state, qdm_state):
    _, psi = tfi_state
    m = DeviceMPS.from_tenpy(psi, device='cpu')
    L = psi.L
    ref = psi.correlation_function('Sz', 'Sz', sites1=range(L), sites2=range(L))
    got = m.correlation_function('Sz', 'Sz')
    assert np.allclose(got, ref, atol=ATOL)

    _, psi = qdm_state
    m = DeviceMPS.from_tenpy(psi, device='cpu')
    L = psi.L
    for ka, kb in [(1, 3), (2, 2), (4, 1)]:
        ref = psi.correlation_function(f'n{ka}', f'n{kb}',
                                       sites1=range(L), sites2=range(L))
        got = m.correlation_function(f'n{ka}', f'n{kb}')
        assert np.allclose(got, ref, atol=ATOL)


def test_correlation_function_subsets(qdm_state):
    """Non-contiguous, asymmetric site lists."""
    _, psi = qdm_state
    m = DeviceMPS.from_tenpy(psi, device='cpu')
    s1, s2 = [0, 2], [1, 2, 3]        # nS = 4 on the 4x2 cylinder
    ref = psi.correlation_function('n1', 'n2', sites1=s1, sites2=s2)
    got = m.correlation_function('n1', 'n2', sites1=s1, sites2=s2)
    assert np.allclose(got, ref, atol=ATOL)


def test_entropy_and_overlap(qdm_state):
    model, psi = qdm_state
    m = DeviceMPS.from_tenpy(psi, device='cpu')
    assert np.allclose(m.entanglement_entropy(), psi.entanglement_entropy(),
                       atol=ATOL)
    model2 = SquareQDM(dict(Lx=4, Ny=2, bc_MPS='finite',
                            t=2.0, V=0.0, V_blockade=50.0))
    psi2 = _ground_state(model2, chi=32)
    ref = complex(psi.overlap(psi2))
    got = m.overlap(psi2)                     # accepts a tenpy MPS directly
    assert abs(got - ref) < 1e-9
    assert abs(m.overlap(m) - 1.0) < 1e-10


def test_observables_ducktype(qdm_state):
    """User observable code works UNCHANGED on a DeviceMPS."""
    model, psi = qdm_state
    m = DeviceMPS.from_tenpy(psi, device='cpu')
    chi1_ref, chi3_ref = structure_factors(psi, model.geo)
    chi1, chi3 = structure_factors(m, model.geo)
    assert abs(chi1 - chi1_ref) < 1e-9 and abs(chi3 - chi3_ref) < 1e-9
    assert abs(dimer_density(m, model.geo)
               - dimer_density(psi, model.geo)) < 1e-10


@pytest.mark.gpu
def test_gpu_matches_oracle(qdm_state):
    """Same comparisons on CUDA."""
    model, psi = qdm_state
    m = DeviceMPS.from_tenpy(psi, device='cuda')
    chi1_ref, chi3_ref = structure_factors(psi, model.geo)     # CPU tenpy
    chi1, chi3 = structure_factors(m, model.geo)               # GPU
    assert abs(chi1 - chi1_ref) < 1e-8 and abs(chi3 - chi3_ref) < 1e-8
    assert abs(m.overlap(psi) - 1.0) < 1e-9
