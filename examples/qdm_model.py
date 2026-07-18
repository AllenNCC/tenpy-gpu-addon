"""Example model: the 4-state vertex-site square-lattice quantum dimer model.

A realistic, non-trivial tenpy model to demonstrate the GPU backends on: the
square-lattice QDM on a finite (or infinite) cylinder in the standard
Rokhsar-Kivelson parametrisation — kinetic amplitude `t` for the plaquette
ring-exchange and interaction `V` for parallel dimer pairs. Each
A-sublattice vertex carries a 4-state site ("which of my four legs holds the
dimer"), the perfect-matching constraint on B-vertices is enforced by a soft
blockade penalty, and plaquette ring-exchange enters as a two-site coupling
between the two A-corners.

Crucially for the GPU backends: the site uses `LegCharge.from_trivial`, i.e.
NO charge conservation — every npc tensor is a single dense block, which is
exactly what `tenpy_gpu_addon` requires. If you adapt this to your own model,
keep it charge-free (`conserve=None` in most stock tenpy models).

Only tenpy is imported here; nothing in this file knows about torch or the
addon. That is the point: the model is plain tenpy, the GPU is bolted on
from the outside (see `run_qdm.py`).
"""
from itertools import combinations

import numpy as np

from tenpy.networks.site import Site
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel, MPOModel


# ---------------------------------------------------------------------------
# 1. Cylinder geometry (finite or infinite)
# ---------------------------------------------------------------------------
def is_A(x, y):
    """A/B checkerboard: A-vertices carry the 4-state sites."""
    return (x + y) % 2 == 0


def star_bonds(x, y, Lx):
    """The four candidate bonds meeting vertex (x, y) as (type, bx, by)."""
    return [('H', x % Lx, y),          # right
            ('H', (x - 1) % Lx, y),    # left
            ('V', x % Lx, y),          # up
            ('V', x % Lx, y - 1)]      # down


def bond_owner(t, x, y, Lx):
    """Which A-vertex owns a bond, and as which leg -> (ax, ay, leg)."""
    if t == 'H':                       # (x,y)-(x+1,y)
        if is_A(x, y):
            return (x % Lx, y, 0)              # right leg of (x,y)
        return ((x + 1) % Lx, y, 1)           # left  leg of (x+1,y)
    else:                              # 'V': (x,y)-(x,y+1)
        if is_A(x, y):
            return (x % Lx, y, 2)              # up   leg of (x,y)
        return (x % Lx, y + 1, 3)             # down leg of (x,y+1)


def build_cylinder(Lx, Ny, bc_MPS='finite'):
    """Cylinder geometry + helpers, switchable finite/infinite."""
    assert Lx % 2 == 0, "even circumference required for the A/B 2-colouring"
    infinite = (bc_MPS == 'infinite')
    if infinite:
        assert Ny % 2 == 0, "even #rings so the A/B colouring is periodic in y"

    A_cell = [(x, y) for y in range(Ny) for x in range(Lx) if is_A(x, y)]
    sidx = {v: i for i, v in enumerate(A_cell)}
    nS = len(A_cell)
    n_bonds = Lx * Ny + (Lx * Ny if infinite else Lx * (Ny - 1))  # H + V bonds

    def bond_exists(t, x, y):
        if t == 'H':
            return True
        return infinite or (0 <= y <= Ny - 2)

    def mps_index(ax, ay):
        if infinite:
            cell, yy = divmod(ay, Ny)
            return sidx[(ax % Lx, yy)], cell
        if 0 <= ay < Ny:
            return sidx[(ax % Lx, ay)], 0
        return None

    return dict(Lx=Lx, Ny=Ny, bc_MPS=bc_MPS, infinite=infinite, A_cell=A_cell,
                sidx=sidx, nS=nS, n_bonds=n_bonds,
                bond_exists=bond_exists, mps_index=mps_index)


def plaquette_corners(x, y, Lx):
    """For the plaquette with lower-left corner (x, y), return its four edges
    and the two A-corner vertices with the leg each uses for its horizontal and
    vertical edge:  (edges, {(ax,ay): {'H': hleg, 'V': vleg}, ...})."""
    edges = [('H', x % Lx, y),          # bottom  (x,y)-(x+1,y)
             ('H', x % Lx, y + 1),      # top     (x,y+1)-(x+1,y+1)
             ('V', x % Lx, y),          # left    (x,y)-(x,y+1)
             ('V', (x + 1) % Lx, y)]    # right   (x+1,y)-(x+1,y+1)
    owners = {}
    for t, bx, by in edges:
        ax, ay, leg = bond_owner(t, bx, by, Lx)
        owners.setdefault((ax, ay), {})[t] = leg
    return edges, owners


# ---------------------------------------------------------------------------
# 2. The 4-state vertex site and the SquareQDM model class
# ---------------------------------------------------------------------------
def make_four_state_site():
    """4-state vertex site: basis ['1','2','3','4'] == 'leg k holds the dimer'
    (no empty state -> exactly one dimer per A-vertex).  Operators: n1..n4 and
    T{a}{b} = |a><b| (the plaquette ring-exchange ladder operators)."""
    leg = npc.LegCharge.from_trivial(4)
    ops = {}
    for k in range(1, 5):
        nk = np.zeros((4, 4)); nk[k - 1, k - 1] = 1.0; ops[f'n{k}'] = nk
    for a in range(1, 5):
        for b in range(1, 5):
            T = np.zeros((4, 4)); T[a - 1, b - 1] = 1.0; ops[f'T{a}{b}'] = T
    return Site(leg, ['1', '2', '3', '4'], **ops)


class SquareQDM(CouplingModel, MPOModel):
    """Square-lattice QDM on a finite/infinite cylinder in Rokhsar-Kivelson
    parametrisation:  H = -t (plaquette flips) + V (parallel dimer pairs)
    + V_blockade (soft perfect-matching constraint on B-vertices).

    `model_params` (defaults shown):
        Lx=6, Ny=6, bc_MPS='finite',
        t=1.0, V=0.0, V_blockade=50.0"""

    default_params = dict(Lx=6, Ny=6, bc_MPS='finite',
                          t=1.0, V=0.0, V_blockade=50.0)

    def __init__(self, model_params):
        p = {**self.default_params, **model_params}
        self.geo = geo = build_cylinder(p['Lx'], p['Ny'], p['bc_MPS'])
        self.infinite, self.nS = geo['infinite'], geo['nS']
        self.t = p['t']
        self.V = p['V']

        site = make_four_state_site()
        bc = 'periodic' if geo['infinite'] else 'open'
        lat = Lattice([geo['nS']], [site], bc=bc, bc_MPS=p['bc_MPS'])
        CouplingModel.__init__(self, lat)

        self._add_terms(self.t, self.V, p['V_blockade'])
        MPOModel.__init__(self, lat, self.calc_H_MPO())

    def _add_terms(self, t, V, V_blockade):
        geo, nS = self.geo, self.nS
        Lx, Ny = geo['Lx'], geo['Ny']
        mps_index, bond_exists = geo['mps_index'], geo['bond_exists']
        INERT = 1e6            # forbid a dimer on a non-existent (boundary) leg

        def add2(strength, ax, ay, op_a, bx, by, op_b):
            """A two-A-site term `strength * op_a(A) op_b(B)`, honouring the
            infinite unit-cell convention.  Skips off-chain endpoints and
            same-site pairs."""
            ma, mb = mps_index(ax, ay), mps_index(bx, by)
            if ma is None or mb is None:
                return
            (ba, ca), (bb, cb) = ma, mb
            cmin = min(ca, cb)
            ia, ib = ba + (ca - cmin) * nS, bb + (cb - cmin) * nS
            if ia == ib:
                return
            if ia < ib:
                self.add_coupling_term(strength, ia, ib, op_a, op_b)
            else:
                self.add_coupling_term(strength, ib, ia, op_b, op_a)

        # (a) forbid occupying a leg whose bond does not exist (finite ends)
        for (x, y), i in geo['sidx'].items():
            for bt, bx, by in star_bonds(x, y, Lx):
                if bond_exists(bt, bx, by):
                    continue
                _, _, k = bond_owner(bt, bx, by, Lx)
                self.add_onsite_term(INERT, i, f'n{k + 1}')

        # (b) plaquette kinetic (-t) and diagonal (V) terms
        for y in range(Ny):
            for x in range(Lx):
                edges, owners = plaquette_corners(x, y, Lx)
                if not all(bond_exists(*e) for e in edges):
                    continue                 # plaquette straddles an open boundary
                if len(owners) != 2:
                    continue                 # degenerate wrap (Lx too small)
                (P, lp), (Q, lq) = owners.items()
                hP, vP = lp['H'], lp['V']
                hQ, vQ = lq['H'], lq['V']
                # kinetic:  -t ( |v><h|_P |v><h|_Q  +  h.c. )
                add2(-t, *P, f'T{vP + 1}{hP + 1}', *Q, f'T{vQ + 1}{hQ + 1}')
                add2(-t, *P, f'T{hP + 1}{vP + 1}', *Q, f'T{hQ + 1}{vQ + 1}')
                # diagonal:  V ( n_hP n_hQ + n_vP n_vQ )
                add2(V, *P, f'n{hP + 1}', *Q, f'n{hQ + 1}')
                add2(V, *P, f'n{vP + 1}', *Q, f'n{vQ + 1}')

        # (c) soft B-vertex blockade == perfect-matching constraint on B
        def add_nn(strength, bond_a, bond_b):
            if not (bond_exists(*bond_a) and bond_exists(*bond_b)):
                return
            ax, ay, ka = bond_owner(*bond_a, Lx)
            bx, by, kb = bond_owner(*bond_b, Lx)
            add2(strength, ax, ay, f'n{ka + 1}', bx, by, f'n{kb + 1}')

        for y in range(Ny):
            for x in range(Lx):
                if is_A(x, y):
                    continue
                for ba, bb in combinations(star_bonds(x, y, Lx), 2):
                    add_nn(V_blockade, ba, bb)


# ---------------------------------------------------------------------------
# 3. Observables — written for a tenpy MPS, run unchanged on a DeviceMPS
# ---------------------------------------------------------------------------
def phase_theta(t, x, y):
    """columnar chi^1 phase on a bond's labelling vertex."""
    return (-1.0) ** x if t == 'H' else -1j * (-1.0) ** y


def phase_zeta(t, x, y):
    """staggered chi^3 phase."""
    return (-1.0) ** (x + y) if t == 'H' else -1j * (-1.0) ** (x + y)


def structure_factors(psi, geo, n_cells=6):
    """Intensive columnar (chi1) and staggered (chi3) order parameters
    m^2 = <O^dag O> / (N_vert/2)^2 with O = sum_bond phi_bond n_bond.

    `psi` may be a tenpy MPS or a `tenpy_gpu_addon.DeviceMPS` — the code is
    identical (duck typing on `expectation_value`/`correlation_function`)."""
    Lx, Ny, nS = geo['Lx'], geo['Ny'], geo['nS']
    mps_index, bond_exists = geo['mps_index'], geo['bond_exists']
    W = n_cells if geo['infinite'] else 1

    bonds = []
    for cell in range(W):
        for y in range(Ny):
            for x in range(Lx):
                for t in ('H', 'V'):
                    if not bond_exists(t, x, y):
                        continue
                    yy = y + cell * Ny
                    ax, ay, k = bond_owner(t, x, yy, Lx)
                    base, c = mps_index(ax, ay)
                    site = base + c * nS
                    bonds.append((site, k + 1,
                                  phase_theta(t, x, yy), phase_zeta(t, x, yy)))

    maxsite = max(s for s, *_ in bonds) + 1
    en = {k: psi.expectation_value(f'n{k}') for k in range(1, 5)}
    C = {(ka, kb): psi.correlation_function(f'n{ka}', f'n{kb}',
                                            sites1=range(maxsite), sites2=range(maxsite))
         for ka in range(1, 5) for kb in range(1, 5)}

    Nb = len(bonds)
    theta = np.array([b[2] for b in bonds])
    zeta = np.array([b[3] for b in bonds])
    M = np.zeros((Nb, Nb))
    for a in range(Nb):
        sa, ka = bonds[a][0], bonds[a][1]
        for b in range(Nb):
            sb, kb = bonds[b][0], bonds[b][1]
            if a == b:
                M[a, b] = en[ka][sa % nS]
            elif sa == sb:
                M[a, b] = 0.0
            else:
                M[a, b] = C[(ka, kb)][sa, sb]
    Nvert = Lx * Ny * W
    denom = (Nvert / 2) ** 2
    chi1 = float(np.real(np.conj(theta) @ M @ theta)) / denom
    chi3 = float(np.real(np.conj(zeta) @ M @ zeta)) / denom
    return chi1, chi3


def dimer_density(psi, geo):
    """<total dimer occupation> / N_bonds -- fixed for a valid covering."""
    tot = sum(psi.expectation_value(f'n{k}').sum() for k in range(1, 5))
    return float(tot) / geo['n_bonds']
