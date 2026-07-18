"""Phase diagram of the square-lattice QDM over the coupling V (at fixed t=1),
built on the modular sweep driver in `run_gpu_sweep.py`.

Protocol (see run_gpu_sweep.py for the schedule details):
  - a warm FORWARD chain (seeded low->high V, anchored in the columnar crystal)
  - a warm BACKWARD chain (seeded high->low V, anchored in the staggered crystal)
  - (optional, --cold) an independent cold chain per point.
Deep in an ordered phase the fixed-chi result is branch- and history-dependent,
so the physical estimator at each V is the **pointwise minimum energy** over
the chains (variational principle) — `combine_min_energy`. The forward and
backward chains bracket the transition where their energies cross.

NOTE: QDM product states are dark states of the plaquette kinetic term, so a
cold chain seeded from one stalls at E=0, chi=1 (see examples/README.md). The
classically-anchored warm chains escape this because the columnar seed carries
flippable plaquettes the kinetic term acts on; --cold is off by default.

This file is the model-specific half; everything reusable is in
`run_gpu_sweep.py`. To do the same for another charge-free tenpy model, copy
this file and replace `build_qdm` / `qdm_measure` / `qdm_classical_seed`.

Usage:
    python qdm_phase_diagram.py --Lx 6 --Ny 6 --chi 512 --device cuda \
        --Vmin -2 --Vmax 3 --nV 21 --out-dir pd_6x6
    python qdm_phase_diagram.py            # tiny CPU default (smoke test)
"""
import argparse
import os

import numpy as np

from qdm_model import (SquareQDM, structure_factors, dimer_density,
                       star_bonds, is_A)
from tenpy_gpu_addon.dense import ucw_kwargs

from run_gpu_sweep import run_gpu_sweep, forward_backward, combine_min_energy
from plot_qdm_pd import plot as plot_pd


# --------------------------------------------------------------------------
# Model + observables (the interface run_gpu_sweep expects)
# --------------------------------------------------------------------------
def make_build_qdm(Lx, Ny, t, V_blockade):
    """Return a `build_model(V)` closure fixing everything but the coupling V."""
    def build_qdm(V):
        return SquareQDM(dict(Lx=Lx, Ny=Ny, bc_MPS='finite', t=t, V=V,
                              V_blockade=V_blockade))
    return build_qdm


def qdm_measure(state, geo):
    """Order parameters + density; `state` is a DeviceMPS or a tenpy MPS."""
    m1sq, m3sq = structure_factors(state, geo)
    return dict(m1sq=float(m1sq), m3sq=float(m3sq),
                density=float(dimer_density(state, geo)))


# --------------------------------------------------------------------------
# Classical (t=0) anchor seeds
# --------------------------------------------------------------------------
# At t=0 the QDM is diagonal in the covering basis: E = V * (# flippable
# plaquettes). The two relevant crystals are exact product coverings:
#   columnar  (m1^2 = 1): even-x A-vertices point right, odd-x point left
#   staggered (m3^2 = 1): every A-vertex points right (zero flippable)
# For V < 0 columnar wins (most flippable), for V > 0 staggered wins.
_COLUMNAR = lambda x, y, Lx: 0 if x % 2 == 0 else 1
_STAGGERED = lambda x, y, Lx: 0


def _covering_states(geo, rule):
    """Per-A-site 4-state labels (index == leg) for a covering rule."""
    Lx = geo['Lx']
    states = [0] * geo['nS']
    for (x, y), i in geo['sidx'].items():
        states[i] = rule(x, y, Lx)
    return states


def _is_valid_covering(geo, states):
    """Each B-vertex touched by exactly one occupied bond (a perfect matching)."""
    Lx, Ny, be = geo['Lx'], geo['Ny'], geo['bond_exists']
    occ = set()
    for (x, y), i in geo['sidx'].items():
        t, bx, by = star_bonds(x, y, Lx)[states[i]]
        if not be(t, bx, by):
            return False
        occ.add((t, bx % Lx, by))
    for y in range(Ny):
        for x in range(Lx):
            if is_A(x, y):
                continue
            n = sum(1 for t, bx, by in star_bonds(x, y, Lx)
                    if be(t, bx, by) and (t, bx % Lx, by) in occ)
            if n != 1:
                return False
    return True


def _product_mps(model, states):
    from tenpy.networks.mps import MPS
    return MPS.from_product_state(model.lat.mps_sites(), states, bc='finite',
                                  **ucw_kwargs(model.lat))


def make_random_seed(chi_max, base_seed=1234):
    """Return a `cold_seed(model, V)` making a random entangled MPS at chi_max
    (reproducible: seed derived from V). Escapes the QDM dark state — a generic
    entangled state is not the product eigenstate — so a cold-from-random scan
    does real work at every independent point."""
    from tenpy.networks.mps import MPS

    def cold_seed(model, V):
        np.random.seed((base_seed + int(round(V * 1000))) & 0x7fffffff)
        psi = MPS.from_desired_bond_dimension(model.lat.mps_sites(), chi_max,
                                              bc='finite',
                                              **ucw_kwargs(model.lat))
        psi.canonical_form()
        return psi
    return cold_seed


def qdm_classical_seed(model, V):
    """Deterministic branch-pure anchor: the lower-energy of the columnar and
    staggered product coverings at this V. Returns (psi0, E_classical)."""
    best = None
    for rule in (_COLUMNAR, _STAGGERED):
        states = _covering_states(model.geo, rule)
        assert _is_valid_covering(model.geo, states), "invalid covering rule"
        psi = _product_mps(model, states)
        E = float(model.H_MPO.expectation_value(psi))   # t-term = 0 on a product
        if best is None or E < best[1]:
            best = (psi, E)
    return best


# --------------------------------------------------------------------------
# Driver + plot
# --------------------------------------------------------------------------
def plot_phase_diagram(combined, path, size=None, Vc=1.0):
    """Render the pointwise-lowest-energy phase diagram in the shared
    two-panel style (see `plot_qdm_pd.plot`): order parameters on the left,
    mid-chain entropy of the winning state with fwd/bwd winner markers on the
    right, the RK point V/t=1 marked. Delegates to `plot_qdm_pd.plot` so the
    in-pipeline figure and the standalone `plot_qdm_pd.py` stay identical."""
    run = dict(V=combined['param'], m1=combined['m1sq'], m3=combined['m3sq'],
               S=combined['S_mid'], winner=combined['winner_name'],
               E=combined['E0'])
    plot_pd(run, path, Vc=Vc, size=size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--Lx', type=int, default=4)
    ap.add_argument('--Ny', type=int, default=4)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--v-blockade', type=float, default=50.0)
    ap.add_argument('--Vmin', type=float, default=-2.0)
    ap.add_argument('--Vmax', type=float, default=3.0)
    ap.add_argument('--nV', type=int, default=11)
    ap.add_argument('--chi', type=int, default=64)
    ap.add_argument('--device', default=None, help="'cuda'/'cpu' (default auto)")
    ap.add_argument('--max-sweeps', type=int, default=30)
    ap.add_argument('--cold', action='store_true',
                    help='also run a cold (product-seeded) chain. OFF by '
                         'default: QDM product states are dark states of the '
                         'kinetic term, so a cold chain stalls at E=0, chi=1 '
                         '(see examples/README.md). The classically-anchored '
                         'warm fwd/bwd pair is the protocol.')
    ap.add_argument('--cold-random', action='store_true',
                    help='INSTEAD of the warm fwd/bwd protocol, run one '
                         'independent cold-from-random point per V (staged '
                         'ramp + convergence stop). A history-free, '
                         'branch-unbiased cross-check; escapes the dark state.')
    ap.add_argument('--save-psi', action='store_true',
                    help='pickle the ground-state MPS of every point '
                         '({tag}_{i:02d}_psi.pkl) so observables can be '
                         're-measured later without new DMRG; also makes '
                         'warm chains resumable after a timeout')
    ap.add_argument('--out-dir', default='qdm_pd',
                    help='per-point .npz + summary + plot land here')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    V_grid = np.linspace(args.Vmin, args.Vmax, args.nV)
    build = make_build_qdm(args.Lx, args.Ny, args.t, args.v_blockade)

    print(f"QDM phase diagram: {args.Lx}x{args.Ny}, chi={args.chi}, "
          f"V in [{args.Vmin}, {args.Vmax}] ({args.nV} pts), "
          f"device={args.device or 'auto'}", flush=True)

    if args.cold_random:
        print("mode: cold-from-random (independent point per V)", flush=True)
        runs = {'coldrand': run_gpu_sweep(
            build, V_grid, measure=qdm_measure, warm_start=False,
            cold_seed=make_random_seed(args.chi), tag='coldrand',
            chi_max=args.chi, device=args.device, out_dir=args.out_dir,
            max_sweeps=args.max_sweeps, save_psi=args.save_psi)}
    else:
        runs = forward_backward(
            build, V_grid, measure=qdm_measure,
            classical_seed=qdm_classical_seed, cold=args.cold,
            chi_max=args.chi, device=args.device, out_dir=args.out_dir,
            max_sweeps=args.max_sweeps, save_psi=args.save_psi)
    combined = combine_min_energy(runs)

    # save the combined phase diagram
    summ = os.path.join(args.out_dir, 'phase_diagram.npz')
    np.savez(summ, **{f"combined_{k}": v for k, v in combined.items()},
             **{f"{name}_{k}": v for name, r in runs.items() for k, v in r.items()})
    print(f"wrote {summ}", flush=True)

    print("\n V/t      E(min)   winner     m1^2     m3^2")
    for i, V in enumerate(combined['param']):
        print(f"{V:6.2f}  {combined['E0'][i]:10.4f}  {str(combined['winner_name'][i]):8s}  "
              f"{combined['m1sq'][i]:.4f}  {combined['m3sq'][i]:.4f}")

    try:
        plot_phase_diagram(combined,
                           os.path.join(args.out_dir, 'phase_diagram.png'),
                           size=(args.Lx, args.Ny))
    except ImportError:
        print("matplotlib not available — skipping plot "
              "(data saved in phase_diagram.npz)", flush=True)


if __name__ == '__main__':
    main()
