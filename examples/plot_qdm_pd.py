"""Plot a QDM phase-diagram run (from qdm_phase_diagram.py) in the two-panel
style: order parameters vs V/t on the left, mid-chain ground-state
entanglement entropy of the winning state (with fwd/bwd winner markers) on
the right, and the RK point V/t = 1 marked as the theory transition.

    python plot_qdm_pd.py --dir /path/to/run_out_dir

Style mirrors the Rydberg phase-diagram figures: a parametrised title that
names the system size and states what the dotted line is, open markers, and a
compact critical-value label sitting on the dotted line.

For later reuse: qdm_corr_map.py is now a general CLI that works on any --save-psi run directory:


python qdm_corr_map.py --dir qdm_runs/phase_diagram_6_12 --Lx 6 --Ny 12 \
    --Vs -1.0,0.0,0.9,1.0 --overlay -1.0,0.0,0.9,1.0
It scans the *_XX.npz records, selects winners by energy, and regenerates only 
missing figures (--force to redo). The same command will work unchanged on the 8×16 run 
once its states are saved (--dir qdm_runs/dimer_corr_8_16 --Lx 8 --Ny 16) — that wider 
cylinder is exactly where the 1/r² window should stretch toward a full decade.
"""
import argparse
import glob
import os
import re

import numpy as np


def _final_S_mid_by_param(run_dir, tag):
    """{param_value: last-sweep S_mid} over all per-point npz files `tag_*`."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, f"{tag}_*.npz")):
        d = np.load(f, allow_pickle=True)
        if 'sweep_S_mid' in d and len(d['sweep_S_mid']):
            out[round(float(d['param']), 6)] = float(d['sweep_S_mid'][-1])
    return out


def load_run(run_dir):
    d = np.load(os.path.join(run_dir, 'phase_diagram.npz'), allow_pickle=True)
    V = d['combined_param']
    m1, m3 = d['combined_m1sq'], d['combined_m3sq']
    winner = np.array([str(w) for w in d['combined_winner_name']])
    E = d['combined_E0']

    # S_mid of the winning state: prefer the combined column (new runs),
    # else reconstruct from the per-point npz files (older runs).
    if 'combined_S_mid' in d.files:
        S = d['combined_S_mid']
    else:
        by_dir = {'forward': _final_S_mid_by_param(run_dir, 'fwd'),
                  'backward': _final_S_mid_by_param(run_dir, 'bwd'),
                  'cold': _final_S_mid_by_param(run_dir, 'cold')}
        S = np.array([by_dir.get(winner[i], {}).get(round(float(V[i]), 6),
                                                    np.nan)
                      for i in range(len(V))])
    return dict(V=V, m1=m1, m3=m3, S=S, winner=winner, E=E)


def _size_from_dir(run_dir):
    """Pull the system size out of a dir name, accepting either the 'NxM'
    form ('pd_6x6_chi256') or the underscore form ('phase_diagram_6_12');
    None if neither is present."""
    m = re.search(r'(\d+)[x_](\d+)', os.path.basename(os.path.normpath(run_dir)))
    return (int(m.group(1)), int(m.group(2))) if m else None


def plot(run, path, Vc=1.0, size=None, title=None,
         crit_label=None, xlabel=r'$V/t$', ylabelL='order parameter',
         ylabelR='bipartite entanglement entropy', density=None):
    """`density` (optional): occupation curve overlaid on the left panel.
    Label/axis kwargs let model-specific callers (e.g. the Rydberg sweep,
    which is in Delta/Omega) reuse the same figure; defaults are the QDM."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    V, m1, m3, S, winner = (run[k] for k in ('V', 'm1', 'm3', 'S', 'winner'))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))

    # --- title: "<size>: phase diagram from pointwise lowest energy; dotted = ..."
    if title is None:
        size_str = f"{size[0]} × {size[1]}, " if size else ""
        title = (f"DMRG {size_str} cylinder phase diagram; "
                 r"RK point $V/t = 1$")
    fig.suptitle(title)
    if crit_label is None:
        crit_label = rf"$V_c={Vc:g}$"

    # --- left: order parameters (open markers, line through) ---
    axL.plot(V, m1, marker='o', ls='-', color='C0', mfc='white', mec='C0',
             ms=6, lw=1.5, label=r'$m_1^2$ columnar')
    axL.plot(V, m3, marker='s', ls='-', color='C3', mfc='white', mec='C3',
             ms=6, lw=1.5, label=r'$m_3^2$ staggered')
    if density is not None:
        axL.plot(V, density, ls=':', color='0.5', lw=1.5, label='density')
    ymax = float(np.nanmax([m1.max(), m3.max()] +
                           ([density.max()] if density is not None else [])))
    if Vc is not None:
        axL.axvline(Vc, ls=':', color='0.5', lw=1.2)
        # axL.text(Vc, 0.55 * ymax, crit_label + '  ', rotation=90, va='center',
                #  ha='right', color='0.45', fontsize=10)
    axL.set_xlabel(xlabel)
    axL.set_ylabel(ylabelL)
    axL.set_ylim(-0.03 * ymax, 1.05 * ymax)
    axL.tick_params(which='both', direction='in')
    axL.legend(loc='upper left')

    # --- right: mid-chain entropy of the winning state + winner markers ---
    axR.plot(V, S, marker='D', ls='-', color='C2', mfc='white', mec='C2',
             ms=6, lw=1.5)
    if Vc is not None:
        axR.axvline(Vc, ls=':', color='0.5', lw=1.2)
    Smin, Smax = np.nanmin(S), np.nanmax(S)
    span = (Smax - Smin) or 1.0
    y0 = Smin - 0.14 * span                      # baseline for winner markers
    fwd, bwd = winner == 'forward', winner == 'backward'
    axR.scatter(V[fwd], np.full(fwd.sum(), y0), marker='o', s=42,
                facecolors='none', edgecolors='0.4', label='fwd')
    axR.scatter(V[bwd], np.full(bwd.sum(), y0), marker='s', s=42,
                facecolors='none', edgecolors='0.4', label='bwd')
    axR.set_xlabel(xlabel)
    axR.set_ylabel(ylabelR)
    axR.set_ylim(y0 - 0.06 * span, Smax + 0.10 * span)
    axR.tick_params(which='both', direction='in')
    axR.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(path, dpi=512)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='phase-diagram run out_dir')
    ap.add_argument('--Vc', type=float, default=1.0,
                    help='reference transition V/t to mark (default RK point 1)')
    ap.add_argument('--Lx', type=int, default=None,
                    help='system circumference for the title (else parsed from --dir)')
    ap.add_argument('--Ny', type=int, default=None,
                    help='system length for the title (else parsed from --dir)')
    ap.add_argument('--title', default=None, help='override the full figure title')
    ap.add_argument('--crit-label', default=None,
                    help='label drawn on the dotted line (default "$V_c=<Vc>$")')
    ap.add_argument('--out', default=None, help='output PNG (default in --dir)')
    args = ap.parse_args()

    size = None
    if args.Lx and args.Ny:
        size = (args.Lx, args.Ny)
    else:
        size = _size_from_dir(args.dir)

    run = load_run(args.dir)
    plot(run, args.out or os.path.join(args.dir, 'phase_diagram_orderparams.png'),
         Vc=args.Vc, size=size, title=args.title, crit_label=args.crit_label)


if __name__ == '__main__':
    main()
