"""Two-site DMRG in pure PyTorch (dense tensors, CPU or CUDA).

Consumes a stock tenpy `MPOModel` + `MPS` (finite, charge-free), runs the
sweep loop on `device`, returns `(E, psi)` with `psi` a regular tenpy MPS.

Includes a density-matrix mixer (White 2005, like tenpy's DensityMatrixMixer):
the reduced density matrix is perturbed by amplitude-scaled environment terms
before diagonalisation, with the same amplitude/decay/disable_after schedule
as tenpy. Set `mixer=False` for plain SVD truncation.
"""
import os
import time

import numpy as np

from .dense import export_mpo, export_mps, import_mps, ucw_kwargs

# TENPY_GPU_ADDON_DEBUG=1: per-update isometry checks + Lanczos internals dump on
# insane Ritz values. Costs one small GEMM per bond update -- debug only.
_DEBUG = os.environ.get('TENPY_GPU_ADDON_DEBUG', '') == '1'


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("the GPU backend needs torch: pip install torch") from e
    return torch


def _matvec(L, W1, W2, theta, torch):
    """H_eff @ theta.  L[i,w,j], W[w,v,p,p*], R[k,u,m], theta[j,p',q',m]."""
    x = torch.tensordot(L, theta, dims=([2], [0]))          # (i, w, p', q', m)
    x = torch.tensordot(x, W1, dims=([1, 2], [0, 3]))       # (i, q', m, v, p)
    x = torch.tensordot(x, W2, dims=([3, 1], [0, 3]))       # (i, m, p, u, q)
    return x


def _lanczos(L, W1, W2, R, theta0, torch, max_iter=20, E_tol=1e-14, P_tol=1e-14):
    """Ground state of H_eff -- a faithful port of tenpy's LanczosGroundState
    (`tenpy.linalg.krylov_based`), Krylov basis as rows of one preallocated
    device matrix.

    tenpy's guards, mirrored:
      - per-iteration convergence: the Ritz residual estimate
        P_err = (|v0[k]| * beta_k / gap)^2 < P_tol AND Delta_E0 < E_tol
        (from the 2nd iteration on, tenpy N_min=2) -- at Krylov exhaustion
        the newest vector's weight |v0[k]| collapses and the loop exits
        BEFORE numerical trouble can build;
      - breakdown: beta below a machine-eps-scale cutoff -> invariant
        subspace reached, happy exit.
    Full reorthogonalisation is TWO batched classical Gram-Schmidt passes
    (CGS2, "twice is enough"), replacing tenpy's sequential modified-GS
    loop. A SINGLE batched pass is unstable exactly at exhaustion: the
    cancellation residue points back along the cached directions, gets
    normalised into a duplicate basis vector, and every later projection
    over-subtracts once per duplicate -- betas then grow by |alpha| per
    iteration and eigh(T) is unbounded below (observed in production: E=-7590 on a
    spectrum bounded at -542; regression test in tests/test_lanczos.py,
    post-mortem in docs/lanczos-stability.md).
    """
    def apply_H(t):
        x = _matvec(L, W1, W2, t, torch)                    # (i, m, p, u, q)
        x = torch.tensordot(x, R, dims=([1, 3], [2, 1]))    # (i, p, q, k)
        return x

    shape = theta0.shape
    n = theta0.numel()
    V = torch.empty((max_iter + 1, n), dtype=theta0.dtype, device=theta0.device)
    V[0] = (theta0 / torch.linalg.vector_norm(theta0)).reshape(-1)
    a = np.zeros(max_iter)
    b = np.zeros(max_iter)
    cutoff = 100 * np.finfo(np.float64).eps        # tenpy's breakdown cutoff
    E = E_prev = None
    gs = np.ones(1)
    N = 1
    for k in range(max_iter):
        w = apply_H(V[k].reshape(shape)).reshape(-1)
        a[k] = float(torch.real(V[k].conj() @ w))
        for _ in range(2):                          # CGS2 reorthogonalisation
            w = w - (V[:k + 1].conj() @ w) @ V[:k + 1]
        b[k] = float(torch.linalg.vector_norm(w))
        T = np.diag(a[:k + 1]) + np.diag(b[:k], 1) + np.diag(b[:k], -1)
        evals, evecs = np.linalg.eigh(T)
        E, gs = float(evals[0]), evecs[:, 0]
        N = k + 1
        if b[k] < max(cutoff, 1e-10 * np.abs(a[:k + 1]).max()):
            break                                   # invariant subspace
        if k >= 1:                                  # tenpy N_min = 2
            gap = max(float(evals[1] - evals[0]), 1e-12)    # tenpy min_gap
            P_err = (abs(gs[k]) * b[k] / gap) ** 2
            if P_err < P_tol and E_prev - E < E_tol * max(1.0, abs(E)):
                break
        E_prev = E
        V[k + 1] = w / b[k]
    if _DEBUG and E < -1.5 * abs(a[0]) - 10.0:      # variationally impossible
        G = (V[:N].conj() @ V[:N].t()).cpu().numpy()
        with np.printoptions(precision=3, linewidth=200):
            print(f"    [LANCZOS-INSANE] E={E:.4f} a0={a[0]:.4f} N={N}\n"
                  f"      a={a[:N]}\n      b={b[:N]}\n"
                  f"      |VV^-I|max={np.abs(G - np.eye(N)).max():.3e}",
                  flush=True)
    if N == 1:
        return E, (V[0] / torch.linalg.vector_norm(V[0])).reshape(shape)
    gs = torch.as_tensor(gs[:N], dtype=theta0.dtype, device=theta0.device)
    theta = (gs @ V[:N]).reshape(shape)
    return E, theta / torch.linalg.vector_norm(theta)


def _safe_eigh(rho, torch):
    """Hermitian eigendecomposition robust to cuSOLVER `syevd` convergence
    failures on (near-)degenerate density matrices — e.g. a product-state
    anchor at small chi, whose rho is nearly rank-deficient. Symmetrise, then
    on failure retry with a negligible diagonal jitter, then fall back to a
    CPU (LAPACK) eigh. All fallbacks are numerically inert (jitter ~1e-12 of
    the spectrum; CPU eigh is the same decomposition)."""
    rho = 0.5 * (rho + rho.conj().t())
    try:
        return torch.linalg.eigh(rho)
    except (torch._C._LinAlgError, RuntimeError):
        n = rho.shape[-1]
        scale = float(torch.diagonal(rho).abs().max()) or 1.0
        eye = torch.eye(n, dtype=rho.dtype, device=rho.device)
        try:
            return torch.linalg.eigh(rho + (1e-12 * scale) * eye)
        except (torch._C._LinAlgError, RuntimeError):
            evals, evecs = torch.linalg.eigh(rho.cpu())    # LAPACK, robust
            return evals.to(rho.device), evecs.to(rho.device)


def _mix_rho_L(L, W, theta, amplitude, torch):
    """White-style perturbation of the left reduced density matrix:
    amplitude * sum_w (L W theta)(L W theta)^dag, trace-normalised."""
    Y = torch.tensordot(L, theta, dims=([2], [0]))       # (i, w, p', q, r)
    Y = torch.tensordot(Y, W, dims=([1, 2], [0, 3]))     # (i, q, r, v, p)
    Y = Y.permute(0, 4, 3, 1, 2)                         # (i, p, v, q, r)
    l, p, v, q, r = Y.shape
    M = Y.reshape(l * p, v * q * r)                      # copy (non-contiguous view)
    del Y                                                # free before the big matmul
    mix = M @ M.conj().t()
    tr = torch.real(torch.diagonal(mix).sum())
    return mix * (amplitude / tr) if tr > 0 else mix * 0


def _mix_rho_R(R, W, theta, amplitude, torch):
    """Mirror of `_mix_rho_L` for the right block.

    Contract theta with R *first*: contracting with W first would leave both
    MPO legs open — a chi^2 d^2 D^2 intermediate (38 GiB at chi=1024) instead
    of chi^2 d^2 D."""
    Y = torch.tensordot(theta, R, dims=([3], [2]))       # (l, p, q, k, u)
    Y = torch.tensordot(Y, W, dims=([2, 4], [3, 1]))     # (l, p, k, w, q)
    Y = Y.permute(4, 2, 0, 1, 3)                         # (q, k, l, p, w)
    q, k, l, p, w = Y.shape
    M = Y.reshape(q * k, l * p * w)                      # copy (non-contiguous view)
    del Y                                                # free before the big matmul
    mix = M @ M.conj().t()
    tr = torch.real(torch.diagonal(mix).sum())
    return mix * (amplitude / tr) if tr > 0 else mix * 0


def run_dmrg_torch(model, psi0=None, chi_max=256, n_sweeps=None, svd_min=1e-10,
                   device=None, dtype=None, lanczos_max_iter=20, verbose=False,
                   mixer=True, mixer_params=None, chi_list=None,
                   min_sweeps=None, max_sweeps=40,
                   max_E_err=1e-8, max_S_err=1e-5):
    """Two-site DMRG on dense torch tensors.

    Parameters mirror the tenpy engine where they exist. Returns
    ``(E, psi, info)`` with `psi` a tenpy MPS and `info` per-sweep stats.

    Sweep schedule (tenpy semantics; A/B-validated on production runs):
    - `chi_list`: {sweep_index: chi} ramp, e.g. {0: 256, 5: 512, 10: 1024}
      (entries capped at `chi_max`; None = fixed `chi_max` throughout).
      Deviation from stock tenpy: the mixer amplitude is RE-ARMED at each
      chi step — small-chi stages then act as branch selectors in the
      ordered phase instead of the full-chi start's manifold mixture.
    - convergence (checked every sweep, tenpy's criterion): stop when
      |dE|/max(|E|,1) < `max_E_err` AND |d mean-entropy| < `max_S_err`,
      counted only with the mixer off, the ramp finished, and sweep >=
      `min_sweeps` (default: last ramp step + mixer disable_after + 2).
      Hitting `max_sweeps` sets info['converged']=False — the cap is the
      cold-point budget, not a convergence claim.
    - `n_sweeps` (legacy fixed mode): run exactly this many sweeps, no
      convergence stop — the benchmark path, unchanged.
    """
    torch = _torch()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sites = model.lat.mps_sites()
    ucw = ucw_kwargs(model.lat)
    if psi0 is None:
        from tenpy.networks.mps import MPS
        psi0 = MPS.from_product_state(sites, [0] * len(sites), bc='finite',
                                      **ucw)

    Ws_np, IdL, IdR = export_mpo(model.H_MPO)
    Bs_np = export_mps(psi0)
    nL = len(Bs_np)
    if dtype is None:
        cplx = any(np.iscomplexobj(a) for a in Ws_np + Bs_np)
        dtype = torch.complex128 if cplx else torch.float64

    Ws = [torch.as_tensor(W, dtype=dtype, device=device) for W in Ws_np]
    Ms = [torch.as_tensor(B, dtype=dtype, device=device) for B in Bs_np]
    SVs = [np.ones(1)] + [None] * (nL - 1) + [np.ones(1)]

    # boundary environments: L[i, w, j], R[k, u, m]
    LE = [None] * nL      # LE[i]: environment left of site i
    RE = [None] * nL      # RE[i]: environment right of site i
    L0 = torch.zeros((1, Ws[0].shape[0], 1), dtype=dtype, device=device)
    L0[0, IdL, 0] = 1.0
    R0 = torch.zeros((1, Ws[-1].shape[1], 1), dtype=dtype, device=device)
    R0[0, IdR, 0] = 1.0
    LE[0], RE[nL - 1] = L0, R0
    for i in range(nL - 1, 0, -1):      # build right environments off the B's
        RE[i - 1] = _update_R(RE[i], Ws[i], Ms[i], torch)

    if mixer_params is None:
        mixer_params = {}
    mix_amp0 = mixer_params.get('amplitude', 1.e-5) if mixer else 0.0
    mix_decay = mixer_params.get('decay', 2.0)
    mix_disable_after = mixer_params.get('disable_after', 15)

    chi_steps = sorted(chi_list.items()) if chi_list else []
    last_chi_step = chi_steps[-1][0] if chi_steps else 0

    def chi_at(sweep):
        c = chi_max if not chi_steps else chi_steps[0][1]
        for s0, c0 in chi_steps:
            if sweep >= s0:
                c = c0
        return min(c, chi_max)

    fixed_schedule = n_sweeps is not None
    if fixed_schedule:
        min_sw = max_sw = n_sweeps
    else:
        max_sw = max_sweeps
        min_sw = min_sweeps if min_sweeps is not None else \
            last_chi_step + (mix_disable_after if mixer else 0) + 2
    chi_now = chi_at(0)   # read dynamically by the split closures below

    def split_svd(theta_mat, l, p, q, r):
        # CPU-only path (LAPACK SVD is fast and exact to 1e-16). On CUDA,
        # update_bond routes mixer-off truncation through split_mixed's
        # rho-eigh instead: cuSOLVER SVD costs 14.3 s at chi*d = 5120 vs
        # 0.35 s for the eigh (measured in a kernel shootout on A100),
        # which made mixer-off sweeps 3.6x slower than mixer-on ones.
        U, S, Vh = torch.linalg.svd(theta_mat, full_matrices=False)
        keep = max(1, min(chi_now, int((S > svd_min).sum().item())))
        disc = float((S[keep:] ** 2).sum())     # discarded weight (sum S^2 = 1)
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
        S = S / torch.linalg.vector_norm(S)
        return U.reshape(l, p, keep), S, Vh.reshape(keep, q, r), keep, disc

    def split_mixed(theta, theta_mat, to_right, amp, i, l, p, q, r):
        """Truncate via the perturbed reduced density matrix (the mixer)."""
        if to_right:
            rho = theta_mat @ theta_mat.conj().t()
            if amp > 0.0:
                rho = rho + _mix_rho_L(LE[i], Ws[i], theta, amp, torch)
        else:
            rho = theta_mat.conj().t() @ theta_mat
            if amp > 0.0:
                rho = rho + _mix_rho_R(RE[i + 1], Ws[i + 1], theta, amp, torch)
        evals, evecs = _safe_eigh(rho, torch)              # ascending
        evals, evecs = evals.flip(0), evecs.flip(1)        # descending
        keep = max(1, min(chi_now, int((evals > svd_min ** 2).sum().item())))
        clamped = torch.clamp(evals, min=0.0)
        disc = float(clamped[keep:].sum() / clamped.sum())  # discarded weight
        V = evecs[:, :keep]
        S = torch.sqrt(torch.clamp(evals[:keep], min=0.0))
        S = S / torch.linalg.vector_norm(S)
        if to_right:
            A = V.reshape(l, p, keep)
            center = V.conj().t() @ theta_mat              # (keep, q*r)
            center = center / torch.linalg.vector_norm(center)
            return A, S, center.reshape(keep, q, r), keep, disc
        else:
            B = V.conj().t().reshape(keep, q, r)
            center = theta_mat @ V                         # (l*p, keep)
            center = center / torch.linalg.vector_norm(center)
            return center.reshape(l, p, keep), S, B, keep, disc

    def update_bond(i, to_right, amp):
        theta0 = torch.tensordot(Ms[i], Ms[i + 1], dims=([2], [0]))  # (l, p, q, r)
        E, theta = _lanczos(LE[i], Ws[i], Ws[i + 1], RE[i + 1], theta0, torch,
                            max_iter=lanczos_max_iter)
        l, p, q, r = theta.shape
        theta_mat = theta.reshape(l * p, q * r)
        # rho-eigh split on CUDA even with the mixer off (amp=0 skips the
        # perturbation): cuSOLVER SVD is ~40x slower than eigh at these
        # sizes. Schmidt values floor at ~1e-8 (squaring) — same floor the
        # mixer path always had; E/observable impact ~1e-16.
        if amp > 0.0 or theta.is_cuda:
            left, S, right, keep, disc = split_mixed(theta, theta_mat, to_right, amp, i, l, p, q, r)
        else:
            A, S, B, keep, disc = split_svd(theta_mat, l, p, q, r)
            if to_right:
                left = A
                right = torch.tensordot(torch.diag(S).to(dtype), B, dims=([1], [0]))
            else:
                left = torch.tensordot(A, torch.diag(S).to(dtype), dims=([2], [0]))
                right = B
        if _DEBUG:
            if to_right:                 # left must be an exact left-isometry
                A = left.reshape(l * p, keep)
                err = float((A.conj().t() @ A
                             - torch.eye(keep, dtype=dtype, device=A.device)
                             ).abs().max())
            else:                        # right must be an exact right-isometry
                B = right.reshape(keep, q * r)
                err = float((B @ B.conj().t()
                             - torch.eye(keep, dtype=dtype, device=B.device)
                             ).abs().max())
            if err > 1e-10:
                print(f"    [ISO-BROKEN] bond {i} to_right={to_right} "
                      f"amp={amp:.1e} keep={keep} iso_err={err:.2e}", flush=True)
        SVs[i + 1] = S.cpu().numpy()
        Ms[i], Ms[i + 1] = left, right
        if to_right:                     # center moved to i+1
            LE[i + 1] = _update_L(LE[i], Ws[i], Ms[i], torch)
        else:                            # center moved to i
            RE[i] = _update_R(RE[i + 1], Ws[i + 1], Ms[i + 1], torch)
        return E, keep, disc

    def _entropy(sv):
        s2 = sv ** 2
        s2 = s2[s2 > 1e-300]
        return float(-(s2 * np.log(s2)).sum())

    info = dict(E=[], chi=[], t_sweep=[], S_mid=[], trunc_err=[], S_mean=[],
                converged=fixed_schedule)   # fixed mode makes no claim
    E = None
    E_prev = S_mean_prev = None
    last_rearm = 0
    for sweep in range(max_sw):
        t0 = time.time()
        new_chi = chi_at(sweep)
        if new_chi != chi_now:
            chi_now = new_chi
            last_rearm = sweep               # mixer re-arm at each chi step
        k = sweep - last_rearm
        amp = mix_amp0 / mix_decay ** k if k < mix_disable_after else 0.0
        chis, discs = [], []
        for i in range(nL - 1):                      # -> right
            E, chi, disc = update_bond(i, True, amp)
            chis.append(chi); discs.append(disc)
        for i in range(nL - 2, -1, -1):              # <- left
            E, chi, disc = update_bond(i, False, amp)
            chis.append(chi); discs.append(disc)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        Ss = [_entropy(SVs[b]) for b in range(1, nL)]
        S_mid, S_mean = Ss[nL // 2 - 1], float(np.mean(Ss))
        info['E'].append(E); info['chi'].append(max(chis))
        info['t_sweep'].append(time.time() - t0)
        info['S_mid'].append(S_mid); info['trunc_err'].append(max(discs))
        info['S_mean'].append(S_mean)
        if verbose:
            print(f"  sweep {sweep + 1}: E={E:.10f}  chi={max(chis)}  "
                  f"S_mid={S_mid:.6f}  trunc={max(discs):.2e}  "
                  f"[{info['t_sweep'][-1]:.2f}s]", flush=True)
        # tenpy convergence test, checked every sweep (GPU sweeps are pricey)
        if (not fixed_schedule and E_prev is not None
                and sweep + 1 >= min_sw and amp == 0.0 and sweep >= last_chi_step):
            dE = abs(E - E_prev) / max(abs(E), 1.0)
            dS = abs(S_mean - S_mean_prev)
            if dE < max_E_err and dS < max_S_err:
                info['converged'] = True
                if verbose:
                    print(f"  converged after {sweep + 1} sweeps "
                          f"(dE={dE:.1e}, dS={dS:.1e})", flush=True)
                break
        E_prev, S_mean_prev = E, S_mean

    Bs_out = [M.cpu().numpy() for M in Ms]
    psi = import_mps(sites, Bs_out, SVs,
                     unit_cell_width=ucw.get('unit_cell_width'))
    return E, psi, info


def _update_L(L, W, A, torch):
    """L'[i',w',j'] = sum L[i,w,j] conj(A)[i,p,i'] W[w,w',p,p'] A[j,p',j']."""
    y = torch.tensordot(L, A, dims=([2], [0]))               # (i, w, p', j')
    y = torch.tensordot(y, W, dims=([1, 2], [0, 3]))         # (i, j', w', p)
    y = torch.tensordot(A.conj(), y, dims=([0, 1], [0, 3]))  # (i', j', w')
    return y.permute(0, 2, 1)                                # (i', w', j')


def _update_R(R, W, B, torch):
    """R'[i,w,j] = sum conj(B)[i,p,i2] W[w,w2,p,p'] B[j,p',j2] R[i2,w2,j2]."""
    y = torch.tensordot(B, R, dims=([2], [2]))               # (j, p', i2, w2)
    y = torch.tensordot(W, y, dims=([1, 3], [3, 1]))         # (w, p, j, i2)
    y = torch.tensordot(B.conj(), y, dims=([1, 2], [1, 3]))  # (i, w, j)
    return y
