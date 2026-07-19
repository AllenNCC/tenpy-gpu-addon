"""Modular parameter-sweep driver for phase diagrams on the GPU DMRG backend.

This module is **model-agnostic** — it knows nothing about the QDM. You give
it three callbacks and it runs cold / warm-forward / warm-backward chains of
`tenpy_gpu_addon.run_dmrg_torch` over a list of parameter values, saving a
self-contained record per point and returning aligned result arrays.

Callbacks (the interface a collaborator implements for their own model):

    build_model(value)          -> a tenpy model for this parameter value.
                                   Must expose `.lat` and `.geo` and work with
                                   `run_dmrg_torch` (i.e. charge-free / dense).
    measure(state, geo)         -> dict of scalar observables for this point.
                                   `state` is a `DeviceMPS` (GPU) or a tenpy
                                   MPS (CPU); write the measurement so it runs
                                   on either by duck typing (see qdm_model's
                                   structure_factors / dimer_density).
    classical_seed(model, value) -> (psi0, E_classical)  [optional]
                                   A product-state MPS at an exact classical
                                   (kinetic-free) ground state, used as the
                                   deterministic anchor of a warm chain. None
                                   (or omitted) falls back to a cold anchor.

Schedule per point (the A/B-validated staged cold start from the engine's
development, carried over verbatim):
  - cold point: chi ramp {0: chi/4, 5: chi/2, 10: chi} with the mixer
    re-armed at each step and disabled after sweep 5, then tenpy-style
    stopping (max_E_err AND max_S_err) with `max_sweeps` as the budget.
  - warm chain link: start from the previous ground state at FULL chi (no
    ramp), mixer decaying to zero after sweep 15 — enough fluctuation to
    relax the seeded branch locally without a low-chi squeeze that would
    re-select it.
  - classical anchor: the seed goes through the staged ramp like a cold
    point. An A/B on the Rydberg staggered anchor (6x6, Delta=15, chi=256)
    found this bit-identical to the full-chi schedule and ~4x faster; if a
    classically seeded anchor ever lands off-branch, revisit (full chi is
    the one-word change at the classical_seed call below).
  - cold anchor of a warm chain (no classical seed): the staged cold start
    with the larger `max_sweeps_first` budget (default 3x max_sweeps); an
    unconverged anchor would contaminate the whole chain.

See `qdm_phase_diagram.py` for a complete worked example.
"""
import json
import os
import pickle
import time

import numpy as np

from tenpy_gpu_addon import run_dmrg_torch


def _measurer(psi, device, gpu_measure):
    """Wrap `psi` in a DeviceMPS on CUDA when available, else return it."""
    if not gpu_measure:
        return psi
    try:
        import torch
        if torch.cuda.is_available() and (device is None
                                          or str(device).startswith('cuda')):
            from tenpy_gpu_addon import DeviceMPS
            return DeviceMPS.from_tenpy(psi, device=device or 'cuda')
    except ImportError:
        pass
    return psi


def run_gpu_sweep(build_model, param_values, *, measure,
                  chi_max=512, warm_start=False, classical_seed=None,
                  cold_seed=None,
                  device=None, out_dir=None, tag='sweep',
                  max_sweeps=30, max_sweeps_first=None,
                  max_E_err=1e-8, max_S_err=1e-5, lanczos_max_iter=20,
                  gpu_measure=True, sweep_verbose=False, verbose=True,
                  save_psi=False):
    """Run one DMRG chain over `param_values` (in the given order).

    warm_start=False: every point is an independent cold start (via the staged
    chi ramp). `cold_seed(model, value) -> psi0` supplies the initial MPS —
    pass a random-state factory for a cold-from-random scan; omit it to start
    from the product state (the engine's default). With `out_dir` set,
    completed points are SKIPPED on restart (resume after timeout/preemption).
    warm_start=True: point 0 is the anchor (classical seed if `classical_seed`
    is given, else a cold ramp with the larger budget); every later point is
    seeded with the previous ground state. Iterating `param_values` ascending
    gives a forward chain, descending gives a backward chain. A warm chain can
    also resume: a completed point whose `save_psi` pickle exists on disk is
    skipped, with the chain's seed reloaded from the pickle — so run long
    chains with `save_psi=True` to make them restartable.

    Returns a dict of numpy arrays aligned to `param_values`: `param`, `E0`,
    `chi`, `converged`, `n_sweeps`, `t_point`, plus one array per key returned
    by `measure`.

    save_psi=True additionally pickles the ground-state MPS of every point to
    `{tag}_{i:02d}_psi.pkl` next to the .npz record (requires `out_dir`), so
    observables can be re-measured later without re-running DMRG. `save_psi`
    may also be a callable `value -> bool` to save only selected points.
    """
    chi_list = {0: max(chi_max // 4, 8), 5: max(chi_max // 2, 8), 10: chi_max}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    obs_cols, base_cols = {}, {k: [] for k in
                             ('param', 'E0', 'S_mid', 'chi', 'converged',
                              'n_sweeps', 't_point')}
    psi_prev = None
    t0 = time.time()

    for i, value in enumerate(param_values):
        fpath = os.path.join(out_dir, f"{tag}_{i:02d}.npz") if out_dir else None
        ppath = fpath[:-4] + '_psi.pkl' if fpath else None
        if fpath and os.path.exists(fpath):
            d = np.load(fpath, allow_pickle=True)
            resumable = (not warm_start) or os.path.exists(ppath)
            if bool(d.get('complete', False)) and resumable:
                if warm_start:      # reload the chain's seed for the next point
                    with open(ppath, 'rb') as f:
                        psi_prev = pickle.load(f)
                base_cols['param'].append(float(d['param']))
                for k in ('E0', 'S_mid', 'chi', 'converged',
                          'n_sweeps', 't_point'):
                    base_cols[k].append(d[k].item())
                for k in d['obs_keys']:
                    obs_cols.setdefault(str(k), []).append(d[f"obs_{k}"].item())
                if verbose:
                    print(f"  {tag} param={value:8.4f}  [skip: complete]",
                          flush=True)
                continue

        model = build_model(value)
        t_pt = time.time()
        warm = warm_start and psi_prev is not None
        anchor = warm_start and psi_prev is None

        # full_chi = start from a state already near the answer (warm link or
        # classical anchor), so run at full chi with a long mixer decay.
        # Otherwise it is a cold point (product or random seed): use the staged
        # chi ramp, whose low-chi stage acts as a branch selector.
        psi0, E_cl, full_chi = None, None, False
        if warm:
            psi0, full_chi = psi_prev, True
        elif anchor and classical_seed is not None:
            psi0, E_cl, full_chi = (*classical_seed(model, value), False)
            if verbose:
                print(f"  [anchor: classical seed, E_cl={E_cl:.4f}]", flush=True)
        elif not warm_start and cold_seed is not None:
            psi0 = cold_seed(model, value)         # cold random start; use ramp

        # cold anchor with no explicit seed gets the larger budget
        max_sw = max_sweeps
        if anchor and psi0 is None:
            max_sw = max_sweeps_first or 3 * max_sweeps

        E0, psi, info = run_dmrg_torch(
            model, psi0=psi0, chi_max=chi_max,
            chi_list=None if full_chi else chi_list,
            mixer_params={'disable_after': 15 if full_chi else 5},
            max_sweeps=max_sw, max_E_err=max_E_err, max_S_err=max_S_err,
            lanczos_max_iter=lanczos_max_iter, device=device,
            verbose=sweep_verbose)
        if warm_start:
            psi_prev = psi

        obs = measure(_measurer(psi, device, gpu_measure), model.geo)
        rec = dict(param=float(value), E0=float(E0),
                   S_mid=float(info['S_mid'][-1]) if info['S_mid'] else 0.0,
                   chi=int(max(psi.chi)), converged=bool(info['converged']),
                   n_sweeps=int(len(info['E'])), t_point=time.time() - t_pt)

        want_psi = save_psi(value) if callable(save_psi) else save_psi
        if fpath and want_psi:
            with open(ppath, 'wb') as f:
                pickle.dump(psi, f)

        if fpath:
            np.savez(fpath, complete=True,
                     params_json=json.dumps({'param': float(value),
                                             'chi_max': chi_max,
                                             'warm_start': warm_start}),
                     obs_keys=np.array(list(obs.keys())),
                     sweep_E=np.array(info['E']),
                     sweep_S_mid=np.array(info['S_mid']),
                     sweep_S_mean=np.array(info['S_mean']),
                     sweep_trunc=np.array(info['trunc_err']),
                     sweep_t=np.array(info['t_sweep']),
                     **rec, **{f"obs_{k}": v for k, v in obs.items()})

        for k in base_cols:
            base_cols[k].append(rec[k])
        for k, v in obs.items():
            obs_cols.setdefault(k, []).append(float(v))

        if verbose:
            obs_str = "  ".join(f"{k}={v:.4f}" for k, v in obs.items())
            print(f"  {tag} param={value:8.4f}  E0={E0:11.4f}  {obs_str}  "
                  f"chi={rec['chi']:4d}  sweeps={rec['n_sweeps']:2d}"
                  f"{'' if rec['converged'] else ' (cap)'}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    out = {k: np.array(v) for k, v in base_cols.items()}
    out.update({k: np.array(v) for k, v in obs_cols.items()})
    return out


def forward_backward(build_model, param_values, *, measure, classical_seed=None,
                     cold=True, **kw):
    """Run cold, warm-forward and warm-backward chains over `param_values`.

    Returns {'cold':..., 'forward':..., 'backward':...} (cold omitted if
    `cold=False`), each the dict from `run_gpu_sweep`, all aligned to
    ascending `param_values`. The forward chain seeds low->high, the backward
    chain high->low; each is anchored by `classical_seed` at its starting end
    (columnar-like at the low end, staggered-like at the high end for a QDM).
    """
    values = list(param_values)
    runs = {}
    if cold:
        runs['cold'] = run_gpu_sweep(build_model, values, measure=measure,
                                     warm_start=False, tag='cold', **kw)
    runs['forward'] = run_gpu_sweep(build_model, values, measure=measure,
                                    warm_start=True, classical_seed=classical_seed,
                                    tag='fwd', **kw)
    bwd = run_gpu_sweep(build_model, values[::-1], measure=measure,
                        warm_start=True, classical_seed=classical_seed,
                        tag='bwd', **kw)
    # re-sort the backward chain back to ascending param order
    order = np.argsort(bwd['param'])
    runs['backward'] = {k: v[order] for k, v in bwd.items()}
    return runs


def combine_min_energy(runs):
    """Pointwise phase-diagram estimator: at each parameter value take the
    result from whichever run found the LOWEST energy (variational principle).

    `runs` is a dict of run-dicts (as from `forward_backward`), each aligned to
    the same ascending `param`. Returns a single dict with the same keys plus
    `winner` (index of the chosen run per point) and `param`.
    """
    names = list(runs)
    ref = runs[names[0]]['param']
    for name in names:
        assert np.allclose(runs[name]['param'], ref), \
            f"run '{name}' param grid misaligned"
    E = np.stack([runs[name]['E0'] for name in names])   # (n_runs, n_points)
    win = np.argmin(E, axis=0)
    out = {'param': ref, 'winner': win,
           'winner_name': np.array([names[w] for w in win])}
    for key in runs[names[0]]:
        if key == 'param':
            continue
        stacked = np.stack([runs[name][key] for name in names])
        out[key] = stacked[win, np.arange(len(ref))]
    return out
