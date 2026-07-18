"""Fixed-workload DMRG benchmark on the QDM example model.

Workload (matching the methodology in results.md): SquareQDM on a finite
Lx x Ny cylinder, t=1, V=0, V_blockade=50, cold-started from a RANDOM MPS
at bond dimension chi_max (`MPS.from_desired_bond_dimension`, fixed seed),
then a fixed number of forced sweeps at fixed chi_max, mixer on. Quoted
t/sweep = mean over the last half of the sweeps.

Why a random init and not a product state: the QDM's product states are
dark states of the plaquette kinetic term (a cold start from one stalls at
E=0 with chi pinned at 1 — see examples/run_qdm.py), so they do no real
work. A random entangled MPS is generic, so every timed sweep does full
chi^3-depth work from sweep 1 with no warm-up. The same fixed seed is used
for every engine and device, so all backends time the identical state.
`--init warm` selects the alternative 2-stock-sweep symmetry-breaking seed.

Engines:
    tenpy   stock TwoSiteDMRGEngine        (the CPU baseline)
    torch   run_dmrg_torch                 (the fully GPU-resident loop)
    dropin  TorchTwoSiteDMRGEngine         (stock engine + GPU Lanczos)

`--measure` additionally times the structure-factor observable suite
(16 correlation matrices + expectations, examples/qdm_model.py) on the
final state — with plain tenpy for engine=tenpy, with DeviceMPS otherwise.

Examples:
    python bench_qdm.py --engine tenpy --threads 16 --chi 128 256 -o cpu.json
    python bench_qdm.py --engine torch --device cuda --chi 512 1024 -o gpu.json
"""
import argparse
import json
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--Lx', type=int, default=6)
    ap.add_argument('--Ny', type=int, default=6)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--V', type=float, default=0.0)
    ap.add_argument('--v-blockade', type=float, default=50.0)
    ap.add_argument('--chi', type=int, nargs='+', default=[128, 256, 512])
    ap.add_argument('--sweeps', type=int, default=6)
    ap.add_argument('--engine', choices=['tenpy', 'torch', 'dropin'],
                    default='torch')
    ap.add_argument('--device', default=None,
                    help="'cuda'/'cpu' for torch engines (default: auto)")
    ap.add_argument('--threads', type=int, default=None,
                    help='BLAS/torch CPU threads (set before imports)')
    ap.add_argument('--lanczos-max-iter', type=int, default=20)
    ap.add_argument('--init', choices=['random', 'warm'], default='random',
                    help="cold start from a random chi_max MPS (default) or "
                         "a 2-sweep symmetry-breaking warm seed")
    ap.add_argument('--seed', type=int, default=1234,
                    help='RNG seed for the random cold start (reproducible)')
    ap.add_argument('--measure', action='store_true',
                    help='also time the structure-factor suite per state')
    ap.add_argument('--torch-profile', default=None, metavar='PREFIX',
                    help='run each point under torch.profiler; key-averages '
                         'table -> PREFIX_chi{chi}.txt (timings carry '
                         'profiler overhead; canonical numbers stay the '
                         'un-profiled ones)')
    ap.add_argument('-o', '--out', default=None, help='JSON output path')
    return ap.parse_args()


args = parse_args()
if args.threads is not None:            # must precede numpy/torch import
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = str(args.threads)

import logging
logging.getLogger('tenpy').setLevel(logging.WARNING)

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg as tenpy_dmrg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'examples'))
from qdm_model import SquareQDM, structure_factors, dimer_density
from tenpy_gpu_addon.dense import ucw_kwargs


def random_state(model, chi, seed):
    """Random entangled MPS at bond dimension `chi` (cold start). Seeded so
    every engine/device times the identical state."""
    np.random.seed(seed)
    psi = MPS.from_desired_bond_dimension(model.lat.mps_sites(), chi,
                                          bc='finite', **ucw_kwargs(model.lat))
    psi.canonical_form()
    return psi


def warm_seed(model, sweeps=2, chi=16):
    """Symmetry-breaking warm seed, identical for every engine/chi."""
    sites = model.lat.mps_sites()
    psi = MPS.from_product_state(sites, [0] * len(sites), bc='finite',
                                 **ucw_kwargs(model.lat))
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True, 'min_sweeps': sweeps, 'max_sweeps': sweeps,
        'combine': False, 'max_trunc_err': None,   # crude seed on purpose
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10}})
    eng.run()
    psi.canonical_form()
    return psi


def run_stock(model, psi, chi, n_sweeps, torch_device=None):
    """Stock tenpy engine (torch_device=None) or the drop-in GPU engine."""
    params = {'mixer': True, 'min_sweeps': n_sweeps, 'max_sweeps': n_sweeps,
              'combine': False, 'N_sweeps_check': 1, 'max_trunc_err': None,
              'trunc_params': {'chi_max': chi, 'svd_min': 1e-10}}
    if torch_device is None:
        eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, params)
    else:
        from tenpy_gpu_addon import TorchTwoSiteDMRGEngine
        eng = TorchTwoSiteDMRGEngine(psi, model,
                                     {**params, 'torch_device': torch_device})
    E, psi = eng.run()
    cum = eng.sweep_stats['time']
    t_sweep = list(np.diff([0.0] + list(cum)))
    return E, psi, dict(t_sweep=t_sweep,
                        chi=[int(c) for c in eng.sweep_stats['max_chi']],
                        E=[float(e) for e in eng.sweep_stats['E']])


def run_torch(model, psi, chi, n_sweeps, device, lanczos_max_iter):
    from tenpy_gpu_addon import run_dmrg_torch
    E, psi, info = run_dmrg_torch(model, psi0=psi, chi_max=chi,
                                  n_sweeps=n_sweeps, device=device,
                                  lanczos_max_iter=lanczos_max_iter,
                                  verbose=True)
    return E, psi, {k: info[k] for k in ('t_sweep', 'chi', 'E')}


def torch_profiled(fn, prefix, chi, device):
    """Run one bench point under torch.profiler; save the key-averages table.

    Timings collected this way carry profiler overhead — read the kernel
    *shares*, not the totals; canonical t/sweep numbers are the un-profiled
    runs. The host-side event stream is large: keep the sweep count low
    (~3) and the host RAM high, or key_averages() itself becomes the
    bottleneck.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity
    on_gpu = torch.cuda.is_available() and str(device) != 'cpu'
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_gpu else [])
    with profile(activities=acts) as prof:
        out = fn()
    key = 'cuda_time_total' if on_gpu else 'cpu_time_total'
    table = prof.key_averages().table(sort_by=key, row_limit=40)
    pfile = f"{prefix}_chi{chi}.txt"
    with open(pfile, 'w') as f:
        f.write(table)
    print(f"# torch profile (chi={chi}) -> {pfile}; top by {key}:",
          flush=True)
    print("\n".join(table.splitlines()[:20]), flush=True)
    return out


def time_measurement(psi, geo, engine, device):
    rec = {}
    if engine == 'tenpy':
        m, rec['backend'] = psi, 'tenpy-cpu'
    else:
        from tenpy_gpu_addon import DeviceMPS
        t0 = time.time()
        m = DeviceMPS.from_tenpy(psi, device=device)
        rec['backend'] = f'DeviceMPS-{m.device}'
        rec['t_upload'] = time.time() - t0
    t0 = time.time()
    chi1, chi3 = structure_factors(m, geo)
    rec['t_structure_factors'] = time.time() - t0
    t0 = time.time()
    rec['dimer_density'] = dimer_density(m, geo)
    rec['t_density'] = time.time() - t0
    rec['m1_sq'], rec['m3_sq'] = chi1, chi3
    return rec


def main():
    import platform
    env = dict(host=platform.node(), engine=args.engine,
               threads=args.threads, sweeps=args.sweeps,
               workload=dict(Lx=args.Lx, Ny=args.Ny, t=args.t, V=args.V,
                             V_blockade=args.v_blockade))
    try:
        import torch
        env['torch'] = torch.__version__
        if args.engine != 'tenpy':
            dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            if str(dev).startswith('cuda'):
                env['gpu'] = torch.cuda.get_device_name(0)
        if args.threads is not None:
            torch.set_num_threads(args.threads)
    except ImportError:
        pass
    import tenpy
    env['tenpy'] = tenpy.__version__

    model = SquareQDM(dict(Lx=args.Lx, Ny=args.Ny, bc_MPS='finite',
                           t=args.t, V=args.V, V_blockade=args.v_blockade))
    D = max(model.H_MPO.get_W(i).shape[1] for i in range(model.H_MPO.L))
    env['workload'].update(n_sites=model.nS, mpo_D=int(D))
    print(f"{env}", flush=True)

    env['init'] = args.init
    warm = warm_seed(model) if args.init == 'warm' else None
    if warm is not None:
        print("init: 2-sweep symmetry-breaking warm seed at chi=16",
              flush=True)

    records = []
    for chi in args.chi:
        print(f"=== engine={args.engine} chi={chi} "
              f"sweeps={args.sweeps} init={args.init} ===", flush=True)
        if args.init == 'random':
            psi = random_state(model, chi, args.seed)
            print(f"  random cold start, initial chi={psi.chi}", flush=True)
        else:
            psi = warm.copy()
        if args.engine == 'torch':
            dev = args.device or 'cuda'
            run = lambda: run_torch(model, psi, chi, args.sweeps,
                                    args.device, args.lanczos_max_iter)
        else:
            dev = None if args.engine == 'tenpy' else (args.device or 'cuda')
            run = lambda: run_stock(model, psi, chi, args.sweeps,
                                    torch_device=dev)
        t0 = time.time()
        if args.torch_profile:
            E, psi, info = torch_profiled(run, args.torch_profile, chi,
                                          dev or 'cpu')
        else:
            E, psi, info = run()
        total = time.time() - t0
        n = len(info['t_sweep'])
        mean_last = float(np.mean(info['t_sweep'][n // 2:]))
        rec = dict(chi=chi, E=float(E), total=total,
                   t_sweep_mean_last_half=mean_last,
                   t_sweep=info['t_sweep'], chi_per_sweep=info['chi'],
                   E_per_sweep=info['E'])
        for i in range(n):
            print(f"  sweep {i + 1}: {info['t_sweep'][i]:7.2f}s  "
                  f"chi={info['chi'][i]}  E={info['E'][i]:.8f}", flush=True)
        print(f"  t/sweep (last half) = {mean_last:.2f}s   "
              f"chi reached = {max(info['chi'])}   [{total:.1f}s total]",
              flush=True)
        if args.measure:
            rec['measure'] = time_measurement(psi, model.geo, args.engine,
                                              args.device)
            print(f"  measurement ({rec['measure']['backend']}): "
                  f"structure factors {rec['measure']['t_structure_factors']:.2f}s"
                  f"  density {rec['measure']['t_density']:.2f}s", flush=True)
        records.append(rec)
        if args.out:                    # write incrementally: timeouts keep
            with open(args.out, 'w') as f:  # everything finished so far
                json.dump(dict(env=env, records=records), f, indent=1)

    print("\nchi      t/sweep   chi_reached")
    for r in records:
        print(f"{r['chi']:5d}  {r['t_sweep_mean_last_half']:8.2f}s  "
              f"{max(r['chi_per_sweep'])}")


if __name__ == '__main__':
    main()
