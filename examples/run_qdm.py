"""Demo: GPU DMRG + GPU measurement on the square-lattice QDM.

The three-line story:
  1. build a plain tenpy model (`SquareQDM` — nothing GPU-aware in it),
  2. hand it to `run_dmrg_torch` (full GPU loop; falls back to CPU torch),
  3. measure with `DeviceMPS` using the SAME observable code you would run
     on a tenpy MPS.

Usage:
    python run_qdm.py                      # small default (laptop-friendly)
    python run_qdm.py --Lx 6 --Ny 6 --chi 512 --device cuda   # production-ish
"""
import argparse
import logging
import time

logging.getLogger('tenpy').setLevel(logging.WARNING)

from qdm_model import SquareQDM, structure_factors, dimer_density
from tenpy_gpu_addon import run_dmrg_torch, DeviceMPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Lx', type=int, default=4, help='circumference (even)')
    ap.add_argument('--Ny', type=int, default=4, help='number of rings')
    ap.add_argument('--t', type=float, default=1.0,
                    help='plaquette kinetic amplitude')
    ap.add_argument('--V', type=float, default=0.0,
                    help='parallel-dimer interaction')
    ap.add_argument('--chi', type=int, default=128)
    ap.add_argument('--device', default=None,
                    help="'cuda' or 'cpu' (default: cuda if available)")
    args = ap.parse_args()

    model = SquareQDM(dict(Lx=args.Lx, Ny=args.Ny, bc_MPS='finite',
                           t=args.t, V=args.V, V_blockade=50.0))
    print(f"SquareQDM {args.Lx}x{args.Ny}: t={model.t:.5g}, V={model.V:.5g}, "
          f"{model.nS} sites, MPO bond dim "
          f"{max(model.H_MPO.get_W(i).shape[1] for i in range(model.H_MPO.L))}")

    # Warm seed: the natural product states are dark states of the plaquette
    # kinetic term (a cold start stalls at E=0 — a property of the QDM, not
    # the engine), so two stock tenpy sweeps break the symmetry.
    # Any tenpy MPS can be handed to run_dmrg_torch as psi0 like this.
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy_gpu_addon.dense import ucw_kwargs
    sites = model.lat.mps_sites()
    psi_seed = MPS.from_product_state(sites, [0] * len(sites), bc='finite',
                                      **ucw_kwargs(model.lat))
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi_seed, model, {
        'mixer': True, 'min_sweeps': 2, 'max_sweeps': 2, 'combine': False,
        'max_trunc_err': None,     # the seed is deliberately crude
        'trunc_params': {'chi_max': 16, 'svd_min': 1e-10}})
    eng.run()
    psi_seed.canonical_form()

    t0 = time.time()
    E, psi, info = run_dmrg_torch(
        model, psi0=psi_seed, chi_max=args.chi,
        chi_list={0: min(64, args.chi), 4: args.chi},
        mixer_params={'disable_after': 6},
        max_sweeps=40, device=args.device, verbose=True)
    print(f"E0 = {E:.10f}   converged={info['converged']}   "
          f"[{time.time() - t0:.1f}s total]")

    # psi is a regular tenpy MPS — anything downstream works. For big chi,
    # wrap it in DeviceMPS so the measurement contractions run on the GPU too:
    m = DeviceMPS.from_tenpy(psi, device=args.device)
    chi1, chi3 = structure_factors(m, model.geo)   # unchanged observable code
    print(f"dimer density = {dimer_density(m, model.geo):.6f}   "
          f"m1^2 = {chi1:.6f}   m3^2 = {chi3:.6f}")


if __name__ == '__main__':
    main()
