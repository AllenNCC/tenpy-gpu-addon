"""Regression: Lanczos near-breakdown must not undershoot the ground state.

Observed in production (see docs/lanczos-stability.md): on a near-product
state (S_mid ~ 3e-3) the final-sweep Lanczos returned E = -322.56 against a
true ground state of -253.28 -- variationally impossible. Mechanism: with
||H_eff|| ~ 300 the cancellation noise floor of the orthogonalised residual
is eps * ||H|| ~ 1e-13, so a beta just above an ABSOLUTE breakdown
threshold (1e-13) already yields a Krylov vector with O(1) orthogonality
error; eigh of the resulting non-orthonormal projection is unbounded below.

The test builds an exactly-diagonalisable H_eff at that energy scale and
starts from (near-)exact eigenvectors, forcing Krylov exhaustion well before
max_iter. Any undershoot beyond rounding is the bug.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from tenpy_gpu_addon.dmrg import _lanczos


def _diag_problem(d=3, scale=18.0, dtype=None):
    """H_eff theta[p,q] = d1[p] * d2[q] * theta[p,q] via trivial (D_w = 1)
    environments: exactly solvable, ||H_eff|| ~ scale^2 ~ the production
    energy scale where the failure occurred."""
    dtype = dtype or torch.float64
    d1 = torch.linspace(-scale, scale - 1.0, d, dtype=dtype)
    d2 = torch.linspace(-scale + 0.5, scale, d, dtype=dtype)
    L = torch.ones(1, 1, 1, dtype=dtype)
    R = torch.ones(1, 1, 1, dtype=dtype)
    W1 = torch.zeros(1, 1, d, d, dtype=dtype)
    W1[0, 0] = torch.diag(d1)
    W2 = torch.zeros(1, 1, d, d, dtype=dtype)
    W2[0, 0] = torch.diag(d2)
    prod = d1[:, None] * d2[None, :]
    pmin, qmin = divmod(int(torch.argmin(prod)), d)
    E_true = float(prod[pmin, qmin])
    v = torch.zeros(1, d, d, 1, dtype=dtype)
    v[0, pmin, qmin, 0] = 1.0
    return L, W1, W2, R, v, E_true


@pytest.mark.parametrize('eta', [0.0, 1e-5, 1e-7, 1e-9])
@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_no_ground_state_undershoot(eta, seed):
    L, W1, W2, R, v, E_true = _diag_problem()
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(*v.shape, generator=g, dtype=v.dtype)
    theta0 = v + eta * noise
    E, theta = _lanczos(L, W1, W2, R, theta0, torch, max_iter=20)
    scale = abs(E_true)
    assert E >= E_true - 1e-9 * scale, \
        f"Ritz value undershoots the true ground state: {E} < {E_true}"
    assert abs(E - E_true) < 1e-9 * scale
    overlap = abs(float((theta.reshape(-1).conj() @ v.reshape(-1)).real))
    assert overlap > 1.0 - 1e-8
