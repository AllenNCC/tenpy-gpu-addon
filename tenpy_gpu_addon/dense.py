"""Export tenpy MPO/MPS to dense tensors and back.

Route B: TeNPy stays untouched upstream — we pull tensors out through its
public API (`to_ndarray`), run DMRG in torch, and hand the state back as a
regular `tenpy.networks.mps.MPS` so all observable code works unchanged.

Only valid for models WITHOUT charge conservation (single dense block);
`export_mpo` checks this.
"""
import numpy as np


def ucw_kwargs(lat):
    """`unit_cell_width` kwarg for MPS constructors, read off the lattice.

    tenpy >= 1.1 warns when it is omitted: the default (len(sites)) is only
    correct for a Chain, while our cylinders need Ls[0] = Lx (what tenpy's
    own `from_lat_product_state` passes). Empty dict on older tenpy without
    the attribute, so the kwarg is never sent to a signature lacking it.
    """
    w = getattr(lat, 'mps_unit_cell_width', None)
    return {} if w is None else {'unit_cell_width': w}


def _assert_trivial_charges(npc_array, what):
    if npc_array.chinfo.qnumber != 0:
        raise ValueError(
            f"{what} carries non-trivial charges; the dense GPU backend "
            "requires a charge-free model (LegCharge.from_trivial).")


def export_mpo(H_MPO):
    """MPO -> (list of ndarray W[wL, wR, p, p*], IdL, IdR)."""
    L = H_MPO.L
    Ws = []
    for i in range(L):
        W = H_MPO.get_W(i)
        _assert_trivial_charges(W, f"MPO tensor {i}")
        Ws.append(W.transpose(['wL', 'wR', 'p', 'p*']).to_ndarray())
    IdL, IdR = H_MPO.get_IdL(0), H_MPO.get_IdR(L - 1)
    if IdL is None or IdR is None:
        raise ValueError("MPO lacks IdL/IdR boundary indices")
    return Ws, IdL, IdR


def export_mps(psi):
    """finite MPS -> list of ndarray B[vL, p, vR] (right-canonical form)."""
    if psi.bc != 'finite':
        raise NotImplementedError("dense GPU backend: finite MPS only (for now)")
    Bs = []
    for i in range(psi.L):
        B = psi.get_B(i, form='B')
        _assert_trivial_charges(B, f"MPS tensor {i}")
        Bs.append(B.transpose(['vL', 'p', 'vR']).to_ndarray())
    return Bs


def import_mps(sites, Bs, SVs, canonicalize='auto', unit_cell_width=None):
    """dense B[vL, p, vR] list + singular-value list -> tenpy MPS.

    The torch engine already returns right-canonical tensors (every bond
    split leaves an exact isometry) with Schmidt values from the final
    half-sweep, so the full canonical_form() re-sweep — minutes of CPU
    SVDs at chi=1024 — is redundant up to SV staleness ~ the convergence
    level. 'auto' runs it only when norm_test disagrees at > 1e-8 (e.g.
    an unconverged intermediate state); True forces it; False trusts.
    """
    from tenpy.networks.mps import MPS
    Bflat = [np.transpose(B, (1, 0, 2)) for B in Bs]   # from_Bflat wants ['p','vL','vR']
    kw = {} if unit_cell_width is None else {'unit_cell_width': unit_cell_width}
    psi = MPS.from_Bflat(sites, Bflat, SVs=SVs, bc='finite', form='B', **kw)
    if canonicalize == 'auto':
        canonicalize = float(np.max(psi.norm_test())) > 1e-8
    if canonicalize:
        psi.canonical_form()
    return psi
