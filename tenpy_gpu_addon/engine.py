"""GPU-resident effective Hamiltonian + Lanczos inside the stock TeNPy engine.

STATUS: fail-safe/validation reference — use `tenpy_gpu_addon.dmrg`
(the `torch` backend) for production; it measured 6–9× faster. This module
stays because it is numerically the stock tenpy engine (agrees to 1e-14),
covers tenpy features the torch loop doesn't (iDMRG, charged models, other
mixers), and is the fast oracle for validating torch-engine changes.

Torch port of an earlier CuPy prototype (``gpu_dmrg.py``): the TeNPy
`TwoSiteDMRGEngine` stays fully in charge — mixer, chi_list, convergence,
truncation all work unchanged — and only the per-bond ground-state search
(the Lanczos matvecs, the dominant cost) runs on the GPU, via TeNPy's public
extension points (`Sweep.EffectiveH`, `DMRGEngine.diag`).

Trade-off vs `tenpy_gpu_addon.dmrg.run_dmrg_torch` (the fully GPU-resident loop):
this keeps every TeNPy feature but still does environment updates and SVD on
the CPU as `npc` operations, and re-uploads LP/RP once per bond. Which wins
is an empirical question — benchmark both.

Scope (as in the original): `combine=False` only; Lanczos with default
options only (`reortho=False`, `E_shift=None`, `N_cache=N_max`).
"""
import warnings

import numpy as np

import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mps_common import TwoSiteH
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig


def _torch():
    import torch
    return torch


def _default_device():
    torch = _torch()
    return 'cuda' if torch.cuda.is_available() else 'cpu'


TRANSFER_COUNT = {'to_gpu': 0, 'from_gpu': 0}   # npc.Array <-> on-device crossings


def reset_transfer_count():
    TRANSFER_COUNT['to_gpu'] = 0
    TRANSFER_COUNT['from_gpu'] = 0


def _theta_to_device(theta, device):
    """npc.Array -> (tensor on device, labels, legs, dtype). One transfer."""
    torch = _torch()
    TRANSFER_COUNT['to_gpu'] += 1
    labels = list(theta.get_leg_labels())
    data = torch.as_tensor(theta.to_ndarray(), device=device)
    return data, labels, theta.legs, theta.dtype


def _theta_from_device(data, labels, legs, dtype):
    """tensor on device -> npc.Array. One transfer."""
    TRANSFER_COUNT['from_gpu'] += 1
    return npc.Array.from_ndarray(data.cpu().numpy(), legs, dtype=dtype, labels=labels)


def _normalize_axes(axes):
    a, b = axes
    if isinstance(a, str):
        a, b = [a], [b]
    return list(a), list(b)


def _tensordot_labeled(a, a_labels, b, b_labels, axes):
    """torch.tensordot for labelled legs, tracking labels like npc.tensordot."""
    torch = _torch()
    ax_a, ax_b = _normalize_axes(axes)
    pos_a = [a_labels.index(l) for l in ax_a]
    pos_b = [b_labels.index(l) for l in ax_b]
    result = torch.tensordot(a, b, dims=(pos_a, pos_b))
    rem_a = [l for i, l in enumerate(a_labels) if i not in pos_a]
    rem_b = [l for i, l in enumerate(b_labels) if i not in pos_b]
    return result, rem_a + rem_b


class TorchTwoSiteH(TwoSiteH):
    """`TwoSiteH` whose environments live on the GPU as dense tensors.

    `matvec_dense` is the on-device core (used by `TorchLanczosGroundState`);
    `matvec` is the npc-facing wrapper so stock solvers still work (one
    transfer per call).
    """
    device = None            # class default; set by TorchTwoSiteDMRGEngine

    def __init__(self, env, i0, combine=False, move_right=True):
        if combine:
            raise NotImplementedError("TorchTwoSiteH only supports combine=False")
        super().__init__(env, i0, combine=False, move_right=move_right)
        torch = _torch()
        dev = self.device or _default_device()
        self._LP_labels = list(self.LP.get_leg_labels())
        self._W0_labels = list(self.W0.get_leg_labels())
        self._W1_labels = list(self.W1.get_leg_labels())
        self._RP_labels = list(self.RP.get_leg_labels())
        self._LP_g = torch.as_tensor(self.LP.to_ndarray(), device=dev)
        self._W0_g = torch.as_tensor(self.W0.to_ndarray(), device=dev)
        self._W1_g = torch.as_tensor(self.W1.to_ndarray(), device=dev)
        self._RP_g = torch.as_tensor(self.RP.to_ndarray(), device=dev)

    def matvec_dense(self, data, labels):
        """Apply H_eff to an on-device tensor; returns same leg order as input."""
        out, labs = _tensordot_labeled(self._LP_g, self._LP_labels, data, labels,
                                       ('vR', 'vL'))
        out, labs = _tensordot_labeled(self._W0_g, self._W0_labels, out, labs,
                                       (['wL', 'p0*'], ['wR', 'p0']))
        out, labs = _tensordot_labeled(out, labs, self._W1_g, self._W1_labels,
                                       (['wR', 'p1'], ['wL', 'p1*']))
        out, labs = _tensordot_labeled(out, labs, self._RP_g, self._RP_labels,
                                       (['wR', 'vR'], ['wL', 'vL']))
        labs = ['vL' if l == 'vR*' else ('vR' if l == 'vL*' else l) for l in labs]
        perm = [labs.index(l) for l in labels]
        return out.permute(perm)

    def matvec(self, theta):
        dev = self.device or _default_device()
        data, labels, legs, dtype = _theta_to_device(theta, dev)
        out = self.matvec_dense(data, labels)
        return _theta_from_device(out, labels, legs, dtype)


class TorchLanczosGroundState:
    """Lanczos with the whole Krylov basis on-device.

    Faithful port of `tenpy.linalg.krylov_based.LanczosGroundState` for its
    default options; only scalars (tridiagonal entries) sync to host per
    iteration.
    """

    def __init__(self, eff_H, data0, labels, options):
        self.eff_H = eff_H
        self.labels = labels
        self.data0 = data0
        options = asConfig(options, 'TorchLanczosGroundState')
        for opt, default in [('reortho', False), ('E_shift', None)]:
            if options.get(opt, default) != default:
                raise NotImplementedError(
                    f"TorchLanczosGroundState doesn't support lanczos_params[{opt!r}]")
        self.N_min = options.get('N_min', 2, int)
        self.N_max = options.get('N_max', 20, int)
        if options.get('N_cache', self.N_max, int) < self.N_max:
            raise NotImplementedError("N_cache < N_max not supported")
        self.P_tol = options.get('P_tol', 1.0e-14, 'real')
        self.min_gap = options.get('min_gap', 1.0e-12, 'real')
        self.E_tol = options.get('E_tol', np.inf, 'real')
        self._cutoff = options.get('cutoff', np.finfo(np.float64).eps * 100, 'real')
        if self.N_min < 2:
            raise ValueError('Should perform at least 2 steps.')

    @staticmethod
    def _inner_re(a, b):
        torch = _torch()
        return float(torch.real(torch.sum(torch.conj(a) * b)))

    def _norm(self, v):
        return self._inner_re(v, v) ** 0.5

    def _converged(self, k):
        E = self.Es[k, :]
        RitzRes = abs(self._result_krylov[k]) * self.h[k, k + 1]
        gap = max(E[1] - E[0], self.min_gap)
        P_err = (RitzRes / gap) ** 2
        Delta_E0 = self.Es[k - 1, 0] - E[0]
        return P_err < self.P_tol and Delta_E0 < self.E_tol

    def run(self):
        self.h = np.zeros((self.N_max + 1, self.N_max + 1))
        self.Es = np.zeros((self.N_max, self.N_max))
        cache = []
        w = self.data0
        beta = self._norm(w)
        if beta < self._cutoff:
            raise ValueError(f'Norm of initial guess too small: {beta}')
        N = 0
        for k in range(self.N_max):
            w = w / beta
            cache.append(w)
            w = self.eff_H.matvec_dense(w, self.labels)
            alpha = self._inner_re(w, cache[-1])
            self.h[k, k] = alpha
            E_kr, v_kr = np.linalg.eigh(self.h[:k + 1, :k + 1])
            self.Es[k, :k + 1] = E_kr
            self._result_krylov = v_kr[:, 0]
            w = w - alpha * cache[-1]
            if k > 0:
                w = w - beta * cache[-2]
            beta = self._norm(w)
            self.h[k, k + 1] = self.h[k + 1, k] = beta
            N = k + 1
            if beta < self._cutoff or (N >= self.N_min and self._converged(k)):
                break

        E0 = self.Es[N - 1, 0]
        if N == 1:
            return E0, self.data0, N
        vf = self._result_krylov
        psif = vf[0] * cache[0]
        for j in range(1, N):
            psif = psif + vf[j] * cache[j]
        psif_norm = self._norm(psif)
        if abs(1.0 - psif_norm) > 1.0e-5:
            warnings.warn(f'poorly conditioned TorchLanczosGroundState! |psi_0| = {psif_norm:f}')
        psif = psif / psif_norm
        return E0, psif, N


class TorchTwoSiteDMRGEngine(dmrg.TwoSiteDMRGEngine):
    """`TwoSiteDMRGEngine` whose Lanczos search is GPU-resident per bond.

    Everything else (mixer, chi_list, truncation, convergence) is stock
    TeNPy. Set `device` via engine options: `dmrg_params['torch_device']`.
    """

    EffectiveH = TorchTwoSiteH

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        dev = self.options.get('torch_device', None, str)
        TorchTwoSiteH.device = dev or _default_device()

    def diag(self, theta_guess):
        uses_lanczos = self.diag_method == 'lanczos' or (
            self.diag_method == 'default'
            and self.eff_H.N >= self.options.get('max_N_for_ED', 400, int)
        )
        if not uses_lanczos:
            return super().diag(theta_guess)

        dev = TorchTwoSiteH.device or _default_device()
        data0, labels, legs, dtype = _theta_to_device(theta_guess, dev)
        solver = TorchLanczosGroundState(self.eff_H, data0, labels, self.lanczos_params)
        E0, data_f, N = solver.run()
        theta = _theta_from_device(data_f, labels, legs, dtype)
        ov_change = 1.0 - abs(npc.inner(theta_guess, theta, 'labels', do_conj=True))
        return E0, theta, N, ov_change
