"""Spin-1 bilinear-biquadratic chain with a position-dependent energy scale.

    H = J sum_x lam_x [ S_x.S_{x+1} + (gamma/3) (S_x.S_{x+1})^2 ]

gamma = 0 is the Heisenberg point, gamma = 1 the AKLT point; for 0 <= gamma
<= 1 the chain is in the gapped Haldane phase with spin-1/2 edge states.
The per-bond scale lam_x (default 1) is what a "conformal cooling" quench
modulates (Zaletel, Kaufman, Stamper-Kurn, Yao, arXiv:1611.04591): a bath
region with lam < 1 is colder at equal entropy density and draws heat out of
the lam = 1 system region.

Stock tenpy only -- no GPU code here. The model is charge-free
(`conserve=None`), which is what the dense GPU backend requires.
"""
import numpy as np

from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinSite

# S.S = sum_ab g_ab S^a S^b over the operator basis (Sz, Sp, Sm)
_SS = [("Sz", "Sz", 1.0), ("Sp", "Sm", 0.5), ("Sm", "Sp", 0.5)]


class SpinOneBLBQChain(CouplingMPOModel):
    """Finite spin-1 BLBQ chain; model params: L, J, gamma, lam.

    `lam` is a scalar or an array of length L-1 (one value per bond).
    """

    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        return SpinSite(S=1.0, conserve=None)

    def init_terms(self, model_params):
        L = model_params.get('L', 10)
        J = model_params.get('J', 1.0)
        gamma = model_params.get('gamma', 0.75)
        lam = np.broadcast_to(np.asarray(model_params.get('lam', 1.0), float),
                              (L - 1,))
        strength = J * lam
        # bilinear part
        for a, b, g in _SS:
            self.add_coupling(g * strength, 0, a, 0, b, 1)
        # biquadratic part: (S.S)^2 = sum g_ab g_cd (S^a S^c)_i (S^b S^d)_j
        for a, b, g1 in _SS:
            for c, d, g2 in _SS:
                self.add_coupling(gamma / 3 * g1 * g2 * strength,
                                  0, f"{a} {c}", 0, f"{b} {d}", 1)


def bond_energy_ops(model):
    """Dense 9x9 bond operators lam_x h_x (sum = H), for local energy maps."""
    site = model.lat.unit_cell[0]
    S = {n: site.get_op(n).to_ndarray() for n in ("Sz", "Sp", "Sm")}
    SS = sum(g * np.kron(S[a], S[b]) for a, b, g in _SS)
    p = model.options
    L, J, gamma = p['L'], p.get('J', 1.0), p.get('gamma', 0.75)
    lam = np.broadcast_to(np.asarray(p.get('lam', 1.0), float), (L - 1,))
    h = J * (SS + gamma / 3 * SS @ SS)
    return [(x, l * h) for x, l in enumerate(lam)]


def ramp_profile(L, bath_end, ramp_end, lam0):
    """Per-bond lam: lam0 in the bath (bond centre <= bath_end, 1-indexed
    sites), 1 in the system (>= ramp_end), linear in between -- Fig. 2 of
    arXiv:1611.04591 with L=30, bath_end=13, ramp_end=20, lam0=0.17."""
    centres = np.arange(L - 1) + 1.5
    frac = np.clip((centres - bath_end) / (ramp_end - bath_end), 0.0, 1.0)
    return lam0 + (1.0 - lam0) * frac
