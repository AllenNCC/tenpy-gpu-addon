"""Canonical TDVP example: a conformal-cooling quench on the GPU.

Reproduces the setup of Fig. 2 of Zaletel, Kaufman, Stamper-Kurn & Yao,
arXiv:1611.04591: a 30-site spin-1 chain (gamma = 3/4, Haldane phase) is
prepared at finite energy density, |psi> ~ exp(-tau H)|Neel> (tau = 0.35/J,
T ~ 0.5 J), then quenched into H[lam_x] whose energy scale is reduced to
lam0 = 0.17 on a "bath" region. Heat flows from the system (right) into the
colder bath (left); the map of local energy densities lam_x h_x(t) - lam_x h_x(0)
is the figure.

Pipeline -- tenpy at both ends, the GPU only in the middle:
  1. stock tenpy: model + MPO (examples/spin1_chain.py), state preparation
     with ExpMPOEvolution in imaginary time;
  2. tenpy_gpu_addon.TorchTDVP: real-time two-site TDVP on the GPU,
     measuring bond energies on device every step, checkpointing;
  3. eng.to_tenpy() hands back a regular tenpy MPS at the end.

    python examples/run_tdvp_quench.py                      # laptop: L=16, chi=64
    python examples/run_tdvp_quench.py --L 30 --chi 512 --tmax 100 --device cuda
"""
import argparse
import os
import time

import numpy as np

from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from tenpy.networks.mps import MPS

from tenpy_gpu_addon import TorchTDVP

from spin1_chain import SpinOneBLBQChain, bond_energy_ops, ramp_profile


def prepare(L, gamma, tau, dtau=0.01, chi=256):
    """exp(-tau H)|Neel>, normalised, with the uniform (lam = 1) Hamiltonian."""
    model = SpinOneBLBQChain({'L': L, 'gamma': gamma, 'bc_MPS': 'finite'})
    psi = MPS.from_product_state(model.lat.mps_sites(), ['up', 'down'] * (L // 2),
                                 bc='finite',
                                 unit_cell_width=model.lat.mps_unit_cell_width)
    n = max(1, int(round(tau / dtau)))
    ExpMPOEvolution(psi, model, {
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-12},
        'compression_method': 'zip_up', 'm_temp': 2,
    }).run_evolution(n, -1j * tau / n)          # imaginary time: exp(-tau H)
    psi.canonical_form()
    return psi


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--L', type=int, default=16)
    p.add_argument('--gamma', type=float, default=0.75)
    p.add_argument('--lam0', type=float, default=0.17)
    p.add_argument('--bath-end', type=float, default=None,
                   help='default: L * 13/30 (the paper geometry)')
    p.add_argument('--ramp-end', type=float, default=None,
                   help='default: L * 20/30')
    p.add_argument('--tau', type=float, default=0.35)
    p.add_argument('--chi', type=int, default=64)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--tmax', type=float, default=10.0)
    p.add_argument('--device', default=None, help="'cuda' | 'cpu' (default: auto)")
    p.add_argument('--out', default='tdvp_quench.npz')
    p.add_argument('--resume', action='store_true',
                   help='continue from <out>_state.npz if present')
    args = p.parse_args()

    L = args.L
    bath_end = args.bath_end or L * 13 / 30
    ramp_end = args.ramp_end or L * 20 / 30
    lam = ramp_profile(L, bath_end, ramp_end, args.lam0)
    model = SpinOneBLBQChain({'L': L, 'gamma': args.gamma, 'lam': lam,
                              'bc_MPS': 'finite'})
    terms = bond_energy_ops(model)          # [(first site, 9x9 op)], sum = H
    ckpt = args.out.replace('.npz', '_state.npz')

    if args.resume and os.path.exists(ckpt):
        eng = TorchTDVP.from_checkpoint(model, ckpt, chi_max=args.chi,
                                        device=args.device)
        rec = {k: list(v) for k, v in np.load(args.out).items()
               if k in ('t', 'h', 'chi', 'wall')}
        print(f'resumed at t = {eng.t:.2f}')
    else:
        t0 = time.time()
        psi = prepare(L, args.gamma, args.tau)
        print(f'prepared exp(-{args.tau} H)|Neel>: chi = {max(psi.chi)} '
              f'[{time.time() - t0:.1f} s]')
        eng = TorchTDVP(model, psi, chi_max=args.chi, device=args.device)
        rec = dict(t=[0.0], h=[eng.local_expectations(terms)],
                   chi=[max(psi.chi)], wall=[0.0])
    print(f'device = {eng.device}, L = {L}, chi_max = {args.chi}, dt = {args.dt}')

    n_steps = int(round(args.tmax / args.dt))
    wall0 = time.time() - rec['wall'][-1]
    for n in range(int(round(eng.t / args.dt)) + 1, n_steps + 1):
        eng.sweep(args.dt)
        h = eng.local_expectations(terms)
        rec['t'].append(eng.t)
        rec['h'].append(h)
        rec['chi'].append(eng.max_chi)
        rec['wall'].append(time.time() - wall0)
        print(f't = {eng.t:6.2f}  E = {h.sum():+.10f}  chi = {eng.max_chi:4d}  '
              f'trunc = {eng.trunc_err:.1e}  [{eng.stats["t_sweep"][-1]:.2f} s/sweep]',
              flush=True)
        if n % 20 == 0 or n == n_steps:      # checkpoint
            np.savez(args.out, lam=lam, **{k: np.array(v) for k, v in rec.items()})
            eng.save(ckpt)

    psi = eng.to_tenpy()                     # a regular tenpy MPS again
    print('final <Sz>:', np.round(psi.expectation_value('Sz'), 3))
    try:
        plot(np.array(rec['t']), np.array(rec['h']), lam,
             args.out.replace('.npz', '.png'))
    except ImportError:
        pass


def plot(t, h, lam, path):
    import matplotlib.pyplot as plt
    x = np.arange(len(lam)) + 1.5           # bond centres, 1-indexed sites
    dh = h - h[0]
    v = np.abs(dh).max()
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(5, 6), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
    im = a0.pcolormesh(x, t, dh, cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest')
    fig.colorbar(im, ax=[a0, a1], label=r'$\lambda_x h_x(t)-\lambda_x h_x(0)$')
    a0.set_ylabel('$t$ [1/J]')
    a1.plot(x, lam)
    a1.set_ylim(0, 1.1)
    a1.set_ylabel(r'$\lambda_x$')
    a1.set_xlabel('bond position $x$')
    fig.savefig(path, dpi=150)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
