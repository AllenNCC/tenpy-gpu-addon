"""GPU-accelerated MPS measurement: a tenpy-API-compatible wrapper.

`DeviceMPS` mirrors the measurement methods of `tenpy.networks.mps.MPS`
(`expectation_value`, `correlation_function`, `overlap`,
`entanglement_entropy`) with the contractions done in torch on `device`.
Because the signatures and return types match, any code written against
tenpy's API — e.g. an observables module written for tenpy states —
works unchanged when handed a `DeviceMPS` instead of the tenpy MPS, by
duck typing. Model-specific conventions (operators, phases) stay in one
place; only the contractions move to the GPU.

Measured motivation: the 16 correlation matrices behind the structure
factors cost ~17 min/state in CPU tenpy at chi=1024 —
they are chi^3 transfer-matrix GEMMs, i.e. exactly what the GPU is for.

Scope (v1, enforced loudly): finite MPS; charge-free (dense) tensors;
bosonic operators only (ops needing a Jordan-Wigner string raise).
The state is taken in right-canonical B form with Schmidt values, which
both tenpy MPS and the torch engine's output provide; measurements
exploit canonicality (contractions only touch the operator span).
"""
import numpy as np

from .dense import export_mps


def _torch():
    import torch
    return torch


class DeviceMPS:
    """Finite MPS held as dense torch B-tensors for fast measurement.

    B[i] has legs [vL, p, vR] (right-canonical); SVs[i] are the Schmidt
    values on the bond left of site i (SVs[0] = [1.] at the boundary).
    """

    def __init__(self, sites, Bs, SVs, device=None, dtype=None):
        torch = _torch()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if dtype is None:
            cplx = any(np.iscomplexobj(b) for b in Bs)
            dtype = torch.complex128 if cplx else torch.float64
        self.sites = list(sites)
        self.device, self.dtype = device, dtype
        self._Bs = [torch.as_tensor(np.asarray(b), dtype=dtype, device=device)
                    for b in Bs]
        self._SVs = [np.asarray(s) for s in SVs]

    @classmethod
    def from_tenpy(cls, psi, device=None, dtype=None):
        """Wrap a tenpy finite MPS (canonical form assumed, as produced by
        DMRG / `import_mps`)."""
        Bs = export_mps(psi)                    # checks finite + charge-free
        SVs = [np.asarray(psi.get_SL(i)) for i in range(psi.L)]
        SVs.append(np.asarray(psi.get_SR(psi.L - 1)))
        return cls(psi.sites, Bs, SVs, device=device, dtype=dtype)

    @property
    def L(self):
        return len(self._Bs)

    # -- operator resolution -------------------------------------------------

    def _op(self, op, i):
        """Site-i operator as a torch matrix; `op` is a name or a matrix."""
        torch = _torch()
        if isinstance(op, str):
            site = self.sites[i]
            if hasattr(site, 'op_needs_JW') and site.op_needs_JW(op):
                raise NotImplementedError(
                    "DeviceMPS v1 handles bosonic operators only; "
                    f"'{op}' needs a Jordan-Wigner string — use tenpy's CPU "
                    "correlation_function for fermionic operators.")
            op = site.get_op(op).to_ndarray()
        arr = np.asarray(op)
        if np.iscomplexobj(arr) and not self.dtype.is_complex:
            raise ValueError("complex operator on a real-dtype DeviceMPS; "
                             "construct with dtype=torch.complex128")
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device)

    def _weights(self, i):
        torch = _torch()
        w = torch.as_tensor(self._SVs[i] ** 2, dtype=self.dtype,
                            device=self.device)
        return w

    # -- tenpy-compatible measurements ---------------------------------------

    def expectation_value(self, op, sites=None):
        """<op_i> for each i (tenpy signature; `op` = name or matrix)."""
        torch = _torch()
        if sites is None:
            sites = range(self.L)
        out = []
        for i in sites:
            O, B, w = self._op(op, i), self._Bs[i], self._weights(i)
            val = torch.einsum('a,apr,pq,aqr->', w, B.conj(), O, B)
            out.append(complex(val))
        return np.real_if_close(np.array(out))

    def entanglement_entropy(self):
        """von Neumann entropy at each internal bond (length L-1)."""
        out = []
        for b in range(1, self.L):
            s2 = self._SVs[b] ** 2
            s2 = s2[s2 > 1e-300]
            out.append(float(-(s2 * np.log(s2)).sum()))
        return np.array(out)

    def overlap(self, other):
        """<self|other> for two finite MPS over the same sites."""
        torch = _torch()
        if not isinstance(other, DeviceMPS):
            other = DeviceMPS.from_tenpy(other, device=self.device,
                                         dtype=self.dtype)
        T = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        for k, b in zip(other._Bs, self._Bs):
            T = torch.einsum('rs,rpx,spy->xy', T, k, b.conj())
        return complex(T.reshape(()))

    def correlation_function(self, ops1, ops2, sites1=None, sites2=None):
        """C[i, j] = <ops1_i ops2_j> (raw, tenpy conventions), shape
        (len(sites1), len(sites2)). Bosonic operators only.

        One left-to-right sweep: environments started at every needed site
        are stacked and propagated together, so each bond is crossed once
        per direction — batched chi^3 GEMMs on the device.
        """
        torch = _torch()
        sites1 = list(range(self.L)) if sites1 is None else list(sites1)
        sites2 = list(range(self.L)) if sites2 is None else list(sites2)
        i_of = {s: n for n, s in enumerate(sites1)}
        j_of = {s: n for n, s in enumerate(sites2)}
        C = np.zeros((len(sites1), len(sites2)), dtype=complex)

        # diagonal: onsite product, <(ops1 @ ops2)_i>
        both = sorted(set(sites1) & set(sites2))
        for s in both:
            O = self._op(ops1, s) @ self._op(ops2, s)
            C[i_of[s], j_of[s]] = complex(torch.einsum(
                'a,apr,pq,aqr->', self._weights(s), self._Bs[s].conj(), O,
                self._Bs[s]))

        # off-diagonal: envs[c] carries (start_site, kind); kind 'A' = env
        # started with ops1 at start_site (terminates with ops2 at j > start,
        # giving C[start, j]); kind 'B' started with ops2 (gives C[i, start]).
        envs, tags = None, []
        for m in range(self.L):
            B = self._Bs[m]
            if envs is not None:
                # terminate at m: all 'A' envs if m is a sites2 member, etc.
                for kind, op_t in (('A', ops2), ('B', ops1)):
                    if (m not in j_of) if kind == 'A' else (m not in i_of):
                        continue
                    hits = [c for c, (_, k) in enumerate(tags) if k == kind]
                    if not hits:
                        continue
                    O = self._op(op_t, m)
                    vals = torch.einsum('crs,rpx,qp,sqx->c',
                                        envs[hits], B, O, B.conj())
                    for c, v in zip(hits, vals):
                        s0 = tags[c][0]
                        if kind == 'A':
                            C[i_of[s0], j_of[m]] = complex(v)
                        else:
                            C[i_of[m], j_of[s0]] = complex(v)
                envs = torch.einsum('crs,rpx,spy->cxy', envs, B, B.conj())
            # start new environments at m (only if a partner site lies ahead)
            new = []
            if m in i_of and any(s > m for s in sites2):
                new.append((self._op(ops1, m), (m, 'A')))
            if m in j_of and any(s > m for s in sites1):
                new.append((self._op(ops2, m), (m, 'B')))
            if new:
                w = self._weights(m)
                stack = torch.stack([
                    torch.einsum('a,apx,qp,aqy->xy', w, B, O, B.conj())
                    for O, _ in new])
                envs = stack if envs is None else torch.cat([envs, stack])
                tags += [t for _, t in new]
        return np.real_if_close(C)
