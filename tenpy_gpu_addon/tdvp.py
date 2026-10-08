"""Two-site TDVP time evolution in pure PyTorch (dense tensors, CPU or CUDA).

Same contract as `run_dmrg_torch`: consumes a stock tenpy `MPOModel` + finite,
charge-free `MPS`, runs on `device`, and hands back a regular tenpy MPS via
`to_tenpy()`. The expensive pieces -- environment updates and the two-site
effective-Hamiltonian matvec -- are shared with the DMRG loop; the Lanczos
ground-state solve is replaced by a Krylov action of exp(a H_eff).

One `sweep(dt)` is the symmetric second-order integrator of Haegeman et al.,
PRB 94, 165116 (2016): a left->right and a right->left half-sweep, each
evolving every bond forward by dt/2 with the two-site H_eff and every
intermediate centre site backward by dt/2 with the one-site H_eff. Real time
by default; `imaginary=True` evolves with exp(-dt H) and renormalises.
"""
import time

import numpy as np

from .dense import export_mpo, export_mps, import_mps, ucw_kwargs
from .dmrg import _eigh_keep, _matvec, _safe_eigh, _torch, _update_L, _update_R


def _expm_krylov(apply, v0, a, torch, max_iter=30, tol=1e-12):
    """exp(a H) v0 for Hermitian H (given as `apply`) and complex scalar a.

    Lanczos with two batched classical Gram-Schmidt passes per step (CGS2,
    as in the DMRG Lanczos -- a single pass loses orthogonality at Krylov
    exhaustion). Converged when the weight the next Krylov vector would
    receive, |beta_k [exp(a T) e0]_k|, drops below `tol` (relative to |v0|).
    Returns (vector, number of matvecs).
    """
    shape = v0.shape
    nrm = torch.linalg.vector_norm(v0)
    V = torch.empty((max_iter + 1, v0.numel()), dtype=v0.dtype, device=v0.device)
    V[0] = (v0 / nrm).reshape(-1)
    alpha = np.zeros(max_iter)
    beta = np.zeros(max_iter)
    cutoff = 100 * np.finfo(np.float64).eps
    c = np.ones(1, dtype=complex)
    N = 1
    for k in range(max_iter):
        w = apply(V[k].reshape(shape)).reshape(-1)
        alpha[k] = float(torch.real(V[k].conj() @ w))
        for _ in range(2):
            w = w - (V[:k + 1].conj() @ w) @ V[:k + 1]
        beta[k] = float(torch.linalg.vector_norm(w))
        T = np.diag(alpha[:k + 1]) + np.diag(beta[:k], 1) + np.diag(beta[:k], -1)
        e, Q = np.linalg.eigh(T)
        c = Q @ (np.exp(a * e) * Q[0, :])            # exp(aT) e0
        N = k + 1
        if beta[k] < max(cutoff, 1e-14 * np.abs(alpha[:k + 1]).max()):
            break                                    # invariant subspace: exact
        if abs(beta[k] * c[k]) < tol:
            break
        V[k + 1] = w / beta[k]
    coef = torch.as_tensor(c[:N], dtype=v0.dtype, device=v0.device)
    return (nrm * (coef @ V[:N])).reshape(shape), N


class TorchTDVP:
    """Two-site TDVP on dense torch tensors.

    The state is kept as `Ms` with the orthogonality centre on site 0 between
    sweeps (Ms[0] = centre, Ms[1:] right-canonical B tensors, legs [vL,p,vR]).

    Parameters
    ----------
    model : tenpy MPOModel (finite, charge-free)
    psi : tenpy MPS (finite, charge-free; not modified), or a list of dense
        [vL,p,vR] arrays in centre-on-0 form, e.g. from a `save()` checkpoint
    chi_max, svd_min : truncation of each two-site split (svd_min relative
        to the largest Schmidt value; on CUDA floored at 1e-7, see
        dmrg._EIGH_SCHMIDT_FLOOR)
    device : 'cuda' | 'cpu' | None (cuda if available)
    krylov_max, krylov_tol : Krylov exponential controls
    """

    def __init__(self, model, psi, chi_max=256, svd_min=1e-10, device=None,
                 dtype=None, krylov_max=30, krylov_tol=1e-12):
        torch = self.torch = _torch()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.dtype = torch.complex128 if dtype is None else dtype
        self.chi_max, self.svd_min = chi_max, svd_min
        self.krylov_max, self.krylov_tol = krylov_max, krylov_tol
        self.sites = model.lat.mps_sites()
        self.ucw = ucw_kwargs(model.lat)
        arrays = psi if isinstance(psi, (list, tuple)) else export_mps(psi)
        self.Ms = [torch.as_tensor(np.asarray(B), dtype=self.dtype, device=device)
                   for B in arrays]
        self.L = len(self.Ms)
        self.t = 0.0
        self.trunc_err = 0.0
        self.stats = dict(matvecs=0, t_sweep=[])
        self.set_model(model)

    # -- setup ----------------------------------------------------------------

    def set_model(self, model):
        """(Re)load the Hamiltonian, e.g. to quench after state preparation."""
        torch = self.torch
        Ws_np, IdL, IdR = export_mpo(model.H_MPO)
        if len(Ws_np) != self.L:
            raise ValueError("model and state have different lengths")
        self.Ws = [torch.as_tensor(W, dtype=self.dtype, device=self.device)
                   for W in Ws_np]
        self.LE = [None] * self.L
        self.RE = [None] * self.L
        L0 = torch.zeros((1, self.Ws[0].shape[0], 1), dtype=self.dtype,
                         device=self.device)
        L0[0, IdL, 0] = 1.0
        R0 = torch.zeros((1, self.Ws[-1].shape[1], 1), dtype=self.dtype,
                         device=self.device)
        R0[0, IdR, 0] = 1.0
        self.LE[0], self.RE[-1] = L0, R0
        for i in range(self.L - 1, 0, -1):     # centre is on site 0
            self.RE[i - 1] = _update_R(self.RE[i], self.Ws[i], self.Ms[i], torch)

    # -- effective Hamiltonians ---------------------------------------------

    def _H2(self, i):
        torch, L, W1, W2, R = self.torch, self.LE[i], self.Ws[i], self.Ws[i + 1], self.RE[i + 1]

        def apply(theta):
            x = _matvec(L, W1, W2, theta, torch)               # (i, m, p, u, q)
            return torch.tensordot(x, R, dims=([1, 3], [2, 1]))  # (i, p, q, k)
        return apply

    def _H1(self, i):
        torch, L, W, R = self.torch, self.LE[i], self.Ws[i], self.RE[i]

        def apply(C):
            x = torch.tensordot(L, C, dims=([2], [0]))          # (i, w, p', m)
            x = torch.tensordot(x, W, dims=([1, 2], [0, 3]))    # (i, m, v, p)
            x = torch.tensordot(x, R, dims=([1, 2], [2, 1]))    # (i, p, k)
            return x
        return apply

    def _expm(self, apply, v, a):
        out, n = _expm_krylov(apply, v, a, self.torch, self.krylov_max,
                              self.krylov_tol)
        self.stats['matvecs'] += n
        return out

    # -- truncated split ------------------------------------------------------

    def _split(self, theta, to_right):
        """theta (l,p,q,r) -> (left, right) with the singular values absorbed
        on the side the centre moves to. Returns discarded weight too."""
        torch = self.torch
        l, p, q, r = theta.shape
        M = theta.reshape(l * p, q * r)
        nrm2 = float(torch.real(torch.vdot(M.reshape(-1), M.reshape(-1))))
        if M.is_cuda:
            # cuSOLVER SVD is ~40x slower than eigh at these sizes (see dmrg.py)
            if to_right:
                evals, V = _safe_eigh(M @ M.conj().t(), torch)
            else:
                evals, V = _safe_eigh(M.conj().t() @ M, torch)
            evals, V = evals.flip(0), V.flip(1)
            keep = _eigh_keep(evals, self.svd_min ** 2 * nrm2, self.chi_max)
            kept = float(torch.clamp(evals[:keep], min=0.0).sum())
            V = V[:, :keep]
            if to_right:
                left = V.reshape(l, p, keep)
                right = (V.conj().t() @ M).reshape(keep, q, r)
            else:
                right = V.conj().t().reshape(keep, q, r)
                left = (M @ V).reshape(l, p, keep)
        else:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            keep = max(1, min(self.chi_max,
                              int((S > self.svd_min * S[0]).sum().item())))
            kept = float((S[:keep] ** 2).sum())
            U, S, Vh = U[:, :keep], S[:keep].to(self.dtype), Vh[:keep]
            if to_right:
                left = U.reshape(l, p, keep)
                right = (S[:, None] * Vh).reshape(keep, q, r)
            else:
                left = (U * S[None, :]).reshape(l, p, keep)
                right = Vh.reshape(keep, q, r)
        disc = max(0.0, 1.0 - kept / nrm2)
        # renormalise the centre (truncation and imaginary time change the norm)
        scale = 1.0 / np.sqrt(kept)
        if to_right:
            right = right * scale
        else:
            left = left * scale
        return left, right, keep, disc

    # -- time step ------------------------------------------------------------

    def sweep(self, dt, imaginary=False):
        """Advance by dt (one symmetric second-order TDVP sweep)."""
        torch = self.torch
        t0 = time.time()
        a_fwd = -dt / 2 if imaginary else -0.5j * dt
        a_bwd = -a_fwd
        L, Ms, Ws = self.L, self.Ms, self.Ws
        disc_tot, chi_max = 0.0, 1
        # left -> right
        for i in range(L - 1):
            theta = torch.tensordot(Ms[i], Ms[i + 1], dims=([2], [0]))
            theta = self._expm(self._H2(i), theta, a_fwd)
            Ms[i], Ms[i + 1], keep, disc = self._split(theta, to_right=True)
            disc_tot += disc
            chi_max = max(chi_max, keep)
            self.LE[i + 1] = _update_L(self.LE[i], Ws[i], Ms[i], torch)
            if i < L - 2:
                Ms[i + 1] = self._expm(self._H1(i + 1), Ms[i + 1], a_bwd)
        # right -> left
        for i in range(L - 2, -1, -1):
            theta = torch.tensordot(Ms[i], Ms[i + 1], dims=([2], [0]))
            theta = self._expm(self._H2(i), theta, a_fwd)
            Ms[i], Ms[i + 1], keep, disc = self._split(theta, to_right=False)
            disc_tot += disc
            chi_max = max(chi_max, keep)
            self.RE[i] = _update_R(self.RE[i + 1], Ws[i + 1], Ms[i + 1], torch)
            if i > 0:
                Ms[i] = self._expm(self._H1(i), Ms[i], a_bwd)
        Ms[0] = Ms[0] / torch.linalg.vector_norm(Ms[0])
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
        self.t += dt
        self.trunc_err = disc_tot
        self.max_chi = chi_max
        self.stats['t_sweep'].append(time.time() - t0)
        return disc_tot

    # -- measurement ----------------------------------------------------------

    def _left_envs(self, upto):
        """X[i] = (M_0..M_{i-1})^dag (M_0..M_{i-1}) for i <= upto."""
        torch = self.torch
        X = [torch.ones((1, 1), dtype=self.dtype, device=self.device)]
        for i in range(upto):
            y = torch.tensordot(X[-1], self.Ms[i], dims=([1], [0]))       # (a, p, b)
            X.append(torch.tensordot(self.Ms[i].conj(), y, dims=([0, 1], [0, 1])))
        return X

    def local_expectations(self, terms):
        """<op> for each (first_site, dense k-site op) term, on the current state.

        Uses the left transfer matrices of the centre-on-0 form, so it is exact
        (no stale Schmidt values); costs one chi^3 contraction per site.
        """
        torch = self.torch
        X = self._left_envs(self.L - 1)
        out = np.empty(len(terms))
        for n, (i0, op) in enumerate(terms):
            d = self.Ms[i0].shape[1]
            k = int(round(np.log(op.shape[0]) / np.log(d)))
            th = self.Ms[i0]
            for j in range(1, k):
                th = torch.tensordot(th, self.Ms[i0 + j], dims=([th.dim() - 1], [0]))
            l, r = th.shape[0], th.shape[-1]
            th = th.reshape(l, -1, r)
            O = torch.as_tensor(op, dtype=self.dtype, device=self.device)
            Oth = torch.tensordot(O, th, dims=([1], [1]))                  # (p, a, b)
            y = torch.tensordot(X[i0], Oth, dims=([1], [1]))               # (a', p, b)
            out[n] = float(torch.real(torch.tensordot(th.conj(), y,
                                                     dims=([0, 1, 2], [0, 1, 2]))))
        return out

    def norm(self):
        return float(torch_real_scalar(self._left_envs(self.L)[-1]))

    @property
    def chi(self):
        return [M.shape[2] for M in self.Ms[:-1]]

    def save(self, path, **extra):
        """Checkpoint the state (centre-on-0 tensors) and time to an .npz."""
        arrs = {f"M{i}": M.cpu().numpy() for i, M in enumerate(self.Ms)}
        np.savez(path, t=self.t, L=self.L, **arrs, **extra)

    @classmethod
    def from_checkpoint(cls, model, path, **kwargs):
        f = np.load(path)
        eng = cls(model, [f[f"M{i}"] for i in range(int(f["L"]))], **kwargs)
        eng.t = float(f["t"])
        return eng

    def to_tenpy(self):
        """Current state as a canonical tenpy MPS."""
        Bs = [M.cpu().numpy() for M in self.Ms]
        SVs = [np.ones(1)] + [np.ones(M.shape[2]) / np.sqrt(M.shape[2])
                              for M in self.Ms[:-1]] + [np.ones(1)]
        return import_mps(self.sites, Bs, SVs, canonicalize=True,
                          unit_cell_width=self.ucw.get('unit_cell_width'))


def torch_real_scalar(x):
    return x.reshape(-1)[0].real
