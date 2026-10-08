"""Do TorchTDVP and TeNPy's TwoSiteTDVPEngine do comparable Krylov work?

The two engines use different stopping rules for the Krylov exponential:
  TeNPy LanczosEvolution : stop when |c_k| < P_tol = 1e-14, N_max = 20, no
                           reorthogonalisation
  TorchTDVP             : stop when |beta_k c_k| < 1e-12, max 30, CGS2
If ours stopped much earlier, part of the speedup would be a looser
tolerance rather than the hardware. From the benchmark checkpoint, for each
chi this runs one warm-up sweep (absorbs the truncation to chi) and then one
counted sweep, and reports per engine
  - effective-Hamiltonian matvecs in the counted sweep,
  - 1 - |<psi|psi_ref>| against TorchTDVP with a much tighter Krylov
    tolerance (1e-15, up to 60 vectors) run the same way.

    python krylov_work.py --ckpt state_L30.npz --chis 256 512
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
from spin1_chain import SpinOneBLBQChain, ramp_profile  # noqa: E402

from tenpy_gpu_addon import TorchTDVP  # noqa: E402


def count_tenpy_matvecs():
    """Wrap every effective-Hamiltonian matvec in tenpy with a counter."""
    import tenpy.algorithms.mps_common as mc
    counter = {'n': 0}
    for name in dir(mc):
        cls = getattr(mc, name)
        if isinstance(cls, type) and 'matvec' in cls.__dict__:
            def wrap(f):
                def matvec(self, theta):
                    counter['n'] += 1
                    return f(self, theta)
                return matvec
            cls.matvec = wrap(cls.__dict__['matvec'])
    return counter


def torch_run(model, ckpt, chi, dt, device, **kw):
    eng = TorchTDVP.from_checkpoint(model, ckpt, chi_max=chi, device=device, **kw)
    eng.sweep(dt)
    n0 = eng.stats['matvecs']
    eng.sweep(dt)
    return eng.to_tenpy(), eng.stats['matvecs'] - n0, eng.stats['t_sweep'][-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='state_L30.npz')
    p.add_argument('--chis', type=int, nargs='+', default=[256, 512])
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='krylov_work.json')
    args = p.parse_args()

    L = 30
    lam = ramp_profile(L, 13, 20, 0.17)
    model = SpinOneBLBQChain({'L': L, 'gamma': 0.75, 'lam': lam, 'bc_MPS': 'finite'})
    counter = count_tenpy_matvecs()
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    import time

    out = []
    for chi in args.chis:
        ref, n_ref, _ = torch_run(model, args.ckpt, chi, args.dt, args.device,
                                  krylov_tol=1e-15, krylov_max=60)
        ours, n_ours, t_ours = torch_run(model, args.ckpt, chi, args.dt, args.device)

        psi = TorchTDVP.from_checkpoint(model, args.ckpt, chi_max=chi,
                                        device='cpu').to_tenpy()
        eng = TwoSiteTDVPEngine(psi, model, {
            'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
            'dt': args.dt, 'N_steps': 1})
        eng.run()
        counter['n'] = 0
        t0 = time.time()
        eng.run()
        t_tenpy = time.time() - t0
        n_tenpy = counter['n']

        rec = dict(chi=chi, matvecs_torch=n_ours, matvecs_tenpy=n_tenpy,
                   matvecs_ref=n_ref,
                   err_torch=1 - abs(ref.overlap(ours)),
                   err_tenpy=1 - abs(ref.overlap(eng.psi)),
                   t_torch=t_ours, t_tenpy=t_tenpy)
        out.append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
