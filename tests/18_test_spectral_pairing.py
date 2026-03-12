"""
Supersymmetric spectral pairing across form degrees (DIR-S11).

The Hodge Laplacian on p-forms is Δ_p = d_{p-1} d_{p-1}^* + d_p^* d_p, where
d_p^* = M_p^{-1} d_p^H M_{p+1} is the adjoint of d_p with respect to the Hodge
inner products.  On an exact cochain complex (d_{p+1} d_p = 0), the "lower" and
"upper" parts act on orthogonal subspaces, producing a spectral pairing:

    nonzero eigenvalues of d_p^* d_p  =  nonzero eigenvalues of d_p d_p^*

Equivalently: every nonzero eigenvalue of Δ_p appears in an adjacent Laplacian.
The full nonzero spectrum of the complex decomposes as:

    spec(Δ_0) = σ²(B_0)
    spec(Δ_1) = σ²(B_0) ∪ σ²(B_1)
    spec(Δ_2) = σ²(B_1) ∪ σ²(B_2)
    spec(Δ_3) = σ²(B_2)

where B_p = M_{p+1}^{1/2} d_p M_p^{-1/2} and σ² denotes squared singular values.

This is a qualitatively different test from d_{p+1}d_p = 0: it verifies the
INTER-LEVEL spectral structure of the complex, not just the algebraic identity.
On the standard complex (d_1 d_0 ≠ 0), the lower and upper parts are not
orthogonal, eigenvalues couple, and the pairing breaks with O(1) residuals.

RAW OUTPUT:

  Kelvin N=2 at generic k=(0.17,0.23,0.31):
    Exact:  Δ₀↔Δ₁ max res = 2.6e-13, Δ₁↔Δ₂ max res = 7.8e-14, Δ₂↔Δ₃ max res = 2.5e-14
    Std:    Δ₀↔Δ₁ max res = 3.0e+00, mean = 7.3e-01, 95/96 mismatched > 0.01

  Kelvin N=2 at Γ:
    harmonics: (1, 3, 3, 1) — zero eigenvalues of Δ₀, Δ₁, Δ₂, Δ₃
    pairing holds for all nonzero eigenvalues (max res = 2.0e-13)

  Cross-talk ||D1_lower · D1_upper|| / (||D1_lower|| · ||D1_upper||):
    Exact:    5.6e-18
    Standard: 3.9e-02

  WP N=1 at Γ, generic k, R: max res < 4e-13 everywhere

ANSWER:
  The exact Bloch-DEC complex exhibits perfect spectral pairing across all
  form degrees (residuals < 1e-12), confirming that the four Hodge Laplacians
  are spectrally coupled via the differential operators d_p exactly as Hodge
  theory predicts.  The standard complex breaks this pairing with O(1) errors
  and 4% cross-talk between the Hodge subspaces.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/18_test_spectral_pairing.py
"""

import sys, os
import numpy as np
from numpy.linalg import norm, eigvalsh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import (build_kelvin_with_dual_info, build_wp_with_dual_info,
                           build_c15_with_dual_info)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup
from physics.bloch_complex import build_d2_bloch_exact, load_foam


def build_laplacians(mesh, k_vec, d1_builder='exact'):
    """Build all four Hodge Laplacians at k-point.

    Returns (eigs, B_ops, d_ops) where eigs[p] are eigenvalues of Δ_p.
    """
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    L_vec = mesh['L_vec']
    M0, M1, M2, M3 = mesh['M0'], mesh['M1'], mesh['M2'], mesh['M3']
    nC = mesh['nC']

    d0 = build_d0_bloch(V, E, k_vec, L_vec, mesh['shifts'])

    if d1_builder == 'exact':
        d1 = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0)
    else:
        crossings = compute_edge_crossings(V, E, mesh['L'])
        edge_lookup = build_edge_lookup(E, crossings)
        d1 = build_d1_bloch_standard(V, E, F, mesh['L'], k_vec,
                                     edge_lookup=edge_lookup)

    d2 = build_d2_bloch_exact(mesh['cfi'], mesh['face_edges'], d1, nC, len(F))

    M0s = np.sqrt(M0)
    M1s = np.sqrt(M1)
    M2s = np.sqrt(M2)
    M3s = np.sqrt(M3)

    B0 = np.diag(M1s) @ d0 @ np.diag(1.0 / M0s)
    B1 = np.diag(M2s) @ d1 @ np.diag(1.0 / M1s)
    B2 = np.diag(M3s) @ d2 @ np.diag(1.0 / M2s)

    D0 = B0.conj().T @ B0
    D1 = B0 @ B0.conj().T + B1.conj().T @ B1
    D2 = B1 @ B1.conj().T + B2.conj().T @ B2
    D3 = B2 @ B2.conj().T

    eigs = [np.sort(np.real(eigvalsh(D)))[::-1] for D in [D0, D1, D2, D3]]
    return eigs, (B0, B1, B2), (d0, d1, d2)


def match_eigenvalues(source, target):
    """Greedily match source eigenvalues to closest target eigenvalues.

    Returns (residuals, unmatched_target).
    Note: greedy matching can underreport failure on non-exact complexes
    (accidental near-matches).  Standard complex residuals are lower bounds.
    """
    pool = list(target)
    residuals = []
    for ev in source:
        if not pool:
            break
        diffs = [abs(ev - p) for p in pool]
        best = np.argmin(diffs)
        residuals.append(diffs[best])
        pool.pop(best)
    return residuals, np.array(pool)


def full_pairing_residuals(eigs):
    """Compute spectral pairing residuals across all four Laplacians.

    Returns max_residual and per-pair maxima (r01, r12, r23).
    """
    nz = [e[e > 1e-10] for e in eigs]

    # Δ₀ eigenvalues should appear in Δ₁
    r01, rem1 = match_eigenvalues(nz[0], nz[1])

    # Remaining Δ₁ eigenvalues should appear in Δ₂
    r12, _ = match_eigenvalues(rem1, nz[2])

    # Δ₃ eigenvalues should appear in Δ₂
    r23, _ = match_eigenvalues(nz[3], nz[2])

    all_res = r01 + r12 + r23
    max_res = max(all_res) if all_res else 0.0
    max01 = max(r01) if r01 else 0.0
    max12 = max(r12) if r12 else 0.0
    max23 = max(r23) if r23 else 0.0

    return max_res, (max01, max12, max23)


# ── Tests ────────────────────────────────────────────────────────────────

def test_1_exact_pairing_kelvin():
    """Spectral pairing holds to machine precision on exact Kelvin complex."""
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    k_frac = np.array([0.17, 0.23, 0.31])
    k_vec = 2 * np.pi * k_frac / mesh['L_vec']

    eigs, _, (d0, d1, d2) = build_laplacians(mesh, k_vec, 'exact')

    # Verify exactness
    assert norm(d1 @ d0) < 1e-10, f'||d1·d0|| = {norm(d1 @ d0):.2e}'
    assert norm(d2 @ d1) < 1e-10, f'||d2·d1|| = {norm(d2 @ d1):.2e}'

    nV, nE, nF, nC = len(mesh['V']), len(mesh['E']), len(mesh['F']), mesh['nC']

    # All eigenvalues nonzero at generic k (acyclic), counts match dimensions
    nz_counts = tuple(int(np.sum(np.abs(e) > 1e-10)) for e in eigs)
    assert nz_counts == (nV, nE, nF, nC), \
        f'Wrong nonzero counts: {nz_counts}, expected ({nV},{nE},{nF},{nC})'

    # Spectral pairing
    max_res, (r01, r12, r23) = full_pairing_residuals(eigs)

    print(f'  Kelvin N=2 (V={nV} E={nE} F={nF} C={nC}) at k_frac={k_frac}')
    print(f'  Δ₀↔Δ₁: {r01:.2e}   Δ₁↔Δ₂: {r12:.2e}   Δ₂↔Δ₃: {r23:.2e}')
    print(f'  max residual: {max_res:.2e}')
    assert max_res < 1e-10, f'Spectral pairing failed: max residual = {max_res:.2e}'
    print('  PASS')


def test_2_harmonics_at_gamma():
    """At Γ, zero eigenvalues reproduce Betti numbers (1,3,3,1)."""
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    eigs, _, _ = build_laplacians(mesh, np.zeros(3), 'exact')

    zeros = tuple(np.sum(np.abs(e) < 1e-10) for e in eigs)
    print(f'  Zero eigenvalues (harmonics): {zeros}')
    assert zeros == (1, 3, 3, 1), f'Expected (1,3,3,1), got {zeros}'

    # Pairing still holds for nonzero eigenvalues
    max_res, _ = full_pairing_residuals(eigs)
    print(f'  Nonzero pairing max residual: {max_res:.2e}')
    assert max_res < 1e-10
    print('  PASS')


def test_3_standard_pairing_fails():
    """Standard complex breaks spectral pairing with O(1) residuals.

    Note: d₂ is built from d₁_std via the same recurrence.  One might expect
    d₂·d₁_std = 0 by construction, concentrating the failure at Δ₀↔Δ₁ only.
    In fact d₂·d₁_std ≠ 0: the recurrence guarantees cancellation along the
    spanning tree of the face adjacency graph, but non-tree edges require
    trivial holonomy (product of d₁ ratios around cycles = 1).  The standard
    d₁ has nontrivial holonomy, so the failure propagates to ALL levels.
    """
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    k_frac = np.array([0.17, 0.23, 0.31])
    k_vec = 2 * np.pi * k_frac / mesh['L_vec']

    # Exact complex
    eigs_ex, _, _ = build_laplacians(mesh, k_vec, 'exact')
    max_ex, _ = full_pairing_residuals(eigs_ex)

    # Standard complex
    eigs_std, _, (d0, d1_std, _) = build_laplacians(mesh, k_vec, 'standard')
    max_std, _ = full_pairing_residuals(eigs_std)

    # d₁d₀ violation
    d1d0_norm = norm(d1_std @ d0)

    print(f'  ||d1_std · d0||  = {d1d0_norm:.2e}')
    print(f'  Exact max res:     {max_ex:.2e}')
    print(f'  Standard max res:  {max_std:.2e}')
    print(f'  Ratio std/exact:   {max_std/max_ex:.0e}')

    assert d1d0_norm > 1.0, f'Standard d1d0 too small: {d1d0_norm:.2e}'
    assert max_std > 0.1, f'Standard pairing residual unexpectedly small: {max_std:.2e}'
    assert max_ex < 1e-10, f'Exact pairing failed: {max_ex:.2e}'
    print('  PASS')


def test_4_cross_talk():
    """Lower and upper Laplacian parts are orthogonal iff complex is exact."""
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    k_frac = np.array([0.17, 0.23, 0.31])
    k_vec = 2 * np.pi * k_frac / mesh['L_vec']

    # Exact
    _, (B0_ex, B1_ex, _), _ = build_laplacians(mesh, k_vec, 'exact')
    D1_lo_ex = B0_ex @ B0_ex.conj().T
    D1_up_ex = B1_ex.conj().T @ B1_ex
    ct_ex = norm(D1_lo_ex @ D1_up_ex) / (norm(D1_lo_ex) * norm(D1_up_ex))

    # Standard
    _, (B0_std, B1_std, _), _ = build_laplacians(mesh, k_vec, 'standard')
    D1_lo_std = B0_std @ B0_std.conj().T
    D1_up_std = B1_std.conj().T @ B1_std
    ct_std = norm(D1_lo_std @ D1_up_std) / (norm(D1_lo_std) * norm(D1_up_std))

    print(f'  Cross-talk (normalized):')
    print(f'    Exact:    {ct_ex:.2e}')
    print(f'    Standard: {ct_std:.2e}')
    print(f'    Ratio:    {ct_std/ct_ex:.0e}')

    assert ct_ex < 1e-14, f'Exact cross-talk too large: {ct_ex:.2e}'
    assert ct_std > 1e-3, f'Standard cross-talk unexpectedly small: {ct_std:.2e}'
    print('  PASS')


def test_5_universality():
    """Spectral pairing holds on WP and C15 at multiple k-points."""
    structures = [
        ('WP N=1',  build_wp_with_dual_info,  1, 1.0),
        ('C15 N=1', build_c15_with_dual_info, 1, 1.0),
    ]

    all_pass = True
    for name, builder, N, Lc in structures:
        mesh = load_foam(builder, N=N, L_cell=Lc, with_stars=True)
        L_vec = mesh['L_vec']

        k_points = {
            'Γ':       np.zeros(3),
            'generic':  2 * np.pi * np.array([0.17, 0.23, 0.31]) / L_vec,
            'R':        2 * np.pi * np.array([0.5, 0.5, 0.5]) / L_vec,
        }

        nV = len(mesh['V'])
        print(f'  {name} (V={nV} E={len(mesh["E"])} F={len(mesh["F"])} C={mesh["nC"]})')
        for label, k_vec in k_points.items():
            eigs, _, _ = build_laplacians(mesh, k_vec, 'exact')
            max_res, _ = full_pairing_residuals(eigs)
            zeros = tuple(np.sum(np.abs(e) < 1e-10) for e in eigs)
            status = 'ok' if max_res < 1e-10 else 'FAIL'
            print(f'    {label:8s}: harmonics={zeros}, max res={max_res:.2e}  [{status}]')
            if max_res >= 1e-10:
                all_pass = False

    assert all_pass, 'Spectral pairing failed on some structure'
    print('  PASS')


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [test_1_exact_pairing_kelvin, test_2_harmonics_at_gamma,
             test_3_standard_pairing_fails, test_4_cross_talk,
             test_5_universality]

    n_pass = 0
    for t in tests:
        name = t.__name__
        print(f'\n{name}:')
        try:
            t()
            n_pass += 1
        except Exception as e:
            print(f'  FAIL: {e}')

    print(f'\n{"="*60}')
    print(f'{n_pass}/{len(tests)} tests passed')
    if n_pass == len(tests):
        print('All tests passed.')
    else:
        sys.exit(1)
