"""TDVP benchmark: one two-site TDVP sweep, three engines, identical state.

Workload: the conformal-cooling quench of examples/run_tdvp_quench.py --
L=30 spin-1 BLBQ chain (gamma=3/4, MPO bond dim 14), bath lam0=0.17,
exp(-0.35 H)|Neel> start, dt=0.1. Real-time evolution at finite energy
density saturates any chi within a few 1/J, so this is the chi^3-bound
regime a production run spends all its time in.

Two phases:
  grow  -- evolve on the GPU with chi_max = max(--chis) up to --t-grow and
           checkpoint (skipped if the checkpoint exists);
  time  -- for every chi and engine, load the SAME checkpoint and time
           --sweeps sweeps. The first sweep at chi < max also truncates the
           state down to chi, so it is reported but excluded from the mean.

Engines:
  torch-cuda  tenpy_gpu_addon.TorchTDVP on the GPU
  tenpy       stock tenpy TwoSiteTDVPEngine (numpy / BLAS threads), the baseline
  torch-cpu   (optional) TorchTDVP on CPU threads

    python tdvp_bench/bench_tdvp.py --chis 128 256 512 1024 -o tdvp_bench/a100.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'examples'))

from spin1_chain import SpinOneBLBQChain, ramp_profile          # noqa: E402
from run_tdvp_quench import prepare                              # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--L', type=int, default=30)
    p.add_argument('--chis', type=int, nargs='+', default=[128, 256, 512, 1024])
    p.add_argument('--engines', nargs='+',
                   default=['torch-cuda', 'tenpy'])
    p.add_argument('--sweeps', type=int, default=3)
    p.add_argument('--cpu-max-chi', type=int, default=1024)
    p.add_argument('--cpu-sweeps-large', type=int, default=2,
                   help='sweeps for CPU engines at chi >= 1024 (they are slow)')
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--t-grow', type=float, default=6.0)
    p.add_argument('--ckpt', default=os.path.join(HERE, 'state_L30.npz'))
    p.add_argument('-o', '--out', default=os.path.join(HERE, 'results.json'))
    args = p.parse_args()

    import torch
    from tenpy_gpu_addon import TorchTDVP

    L = args.L
    lam = ramp_profile(L, L * 13 / 30, L * 20 / 30, 0.17)
    model = SpinOneBLBQChain({'L': L, 'gamma': 0.75, 'lam': lam, 'bc_MPS': 'finite'})
    chi_top = max(args.chis)
    info = dict(L=L, dt=args.dt, mpo_chi=max(model.H_MPO.chi),
                threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)

    if not os.path.exists(args.ckpt):
        eng = TorchTDVP(model, prepare(L, 0.75, 0.35), chi_max=chi_top)
        while eng.t < args.t_grow - 1e-9:
            eng.sweep(args.dt)
            print(f'grow t={eng.t:5.2f} chi={eng.max_chi:5d} '
                  f'{eng.stats["t_sweep"][-1]:.2f}s', flush=True)
        eng.save(args.ckpt)
        info['grow'] = dict(t=eng.t, chi=eng.chi, t_sweep=eng.stats['t_sweep'])

    results = []
    for chi in args.chis:
        for engine in args.engines:
            on_cpu = engine != 'torch-cuda'
            if on_cpu and chi > args.cpu_max_chi:
                continue
            if engine == 'torch-cuda' and not torch.cuda.is_available():
                continue
            n = args.cpu_sweeps_large if (on_cpu and chi >= 1024) else args.sweeps
            ts = []
            if engine.startswith('torch'):
                dev = 'cuda' if engine == 'torch-cuda' else 'cpu'
                eng = TorchTDVP.from_checkpoint(model, args.ckpt, chi_max=chi, device=dev)
                for _ in range(n):
                    eng.sweep(args.dt)
                    ts.append(eng.stats['t_sweep'][-1])
                reached = eng.max_chi
            else:
                from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
                psi = TorchTDVP.from_checkpoint(model, args.ckpt, chi_max=chi,
                                                device='cpu').to_tenpy()
                eng = TwoSiteTDVPEngine(psi, model, {
                    'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
                    'dt': args.dt, 'N_steps': 1})
                for _ in range(n):
                    t0 = time.time()
                    eng.run()
                    ts.append(time.time() - t0)
                reached = max(eng.psi.chi)
            timed = ts[1:] if (len(ts) > 1 and chi < chi_top) else ts
            rec = dict(engine=engine, chi=chi, chi_reached=reached, t_sweep=ts,
                       mean_s=float(np.mean(timed)))
            results.append(rec)
            print(json.dumps(rec), flush=True)
            with open(args.out, 'w') as f:
                json.dump(dict(info=info, results=results), f, indent=1)


if __name__ == '__main__':
    main()
