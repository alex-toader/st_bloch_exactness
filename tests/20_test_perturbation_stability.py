"""
Cohomological stability under mesh perturbation (DIR-S17).

The exactness-preserving construction d₁(k) is algebraic: d₁ is derived from
d₀ via the face-boundary recurrence, so d₁d₀ = 0 by construction regardless
of vertex positions.  This means Betti numbers, spectral pairing, and Hodge
decomposition should be invariant under vertex perturbation — as long as the
mesh topology remains valid (edge shifts unchanged).

We verify this by adding Gaussian noise to Kelvin vertex positions at amplitudes
up to 20% of the mean edge length.  At each perturbation level, all cohomological
and spectral properties are preserved to machine precision.

The construction breaks at ~25% perturbation (on Kelvin N=2), when vertex
displacements are large enough to change edge shift vectors (periodic boundary
crossing assignments).  This is a limitation of compute_edge_shifts, not of the
recurrence itself.  The threshold in absolute terms is roughly half the distance
from an edge midpoint to the nearest periodic boundary — so the relative threshold
(as % of edge length) increases on finer meshes.  A defensible claim: the
construction is invariant under vertex perturbation as long as edge shift vectors
are preserved.

RAW OUTPUT:

  Kelvin N=2, mean edge length = 0.685, seed=42:

  eps/L_edge  ||d1d0||  ||d2d1||  β(k)   β(Γ)      ||PE·PC||    pair_res
  0.00        8.8e-16  2.1e-15  (0,0)  (1,3)     1.2e-14    2.6e-13
  0.01        8.8e-16  2.1e-15  (0,0)  (1,3)     1.2e-14    3.3e-13
  0.05        8.8e-16  2.1e-15  (0,0)  (1,3)     1.1e-14    1.4e-13
  0.10        8.8e-16  2.1e-15  (0,0)  (1,3)     1.1e-14    1.7e-13
  0.20        8.8e-16  2.1e-15  (0,0)  (1,3)     1.2e-14    3.3e-13

  All properties invariant.  Construction breaks at eps ≈ 0.25 (edge shift
  misclassification under periodic boundary conditions).

ANSWER:
  The cochain recurrence is algebraically robust: Betti numbers, spectral
  pairing, and Hodge decomposition are invariant under vertex perturbation.
  Loss of stability occurs only when the geometric periodic edge classification
  (compute_edge_shifts) changes.  The complex structure is untouched; only the
  Hodge star accuracy degrades with perturbation.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/20_test_perturbation_stability.py
"""

import sys, os
import numpy as np
from numpy.linalg import norm, svd, eigvalsh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import build_kelvin_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch_complex import (build_face_edge_map, build_cell_face_incidence,
                                   build_d2_bloch_exact)


# ── Helpers ──────────────────────────────────────────────────────────────

def perturb_and_test(data, eps_frac, seed=42):
    """Perturb vertex positions and check all cohomological properties.

    Returns dict with all results, or raises if construction fails.
    """
    V_orig = data['V']
    E, F = data['E'], data['F']
    L_vec = data['L_vec']
    nV, nE, nF = len(V_orig), len(E), len(F)

    mean_edge = np.mean([norm(V_orig[e[1]] - V_orig[e[0]]) for e in E])
    eps = eps_frac * mean_edge

    # Perturb
    rng = np.random.RandomState(seed)
    data_p = dict(data)
    if eps > 0:
        data_p['V'] = V_orig + rng.randn(*V_orig.shape) * eps
    else:
        data_p['V'] = V_orig.copy()

    V = data_p['V']
    nC = len(data_p['cell_centers'])

    # Rebuild complex infrastructure
    shifts = compute_edge_shifts(V, E, L_vec)
    cfi, _ = build_cell_face_incidence(data_p)
    face_edges = build_face_edge_map(F, E)

    # Generic k
    k_vec = 2 * np.pi * np.array([0.17, 0.23, 0.31]) / L_vec
    d0 = build_d0_bloch(V, E, k_vec, L_vec, shifts)
    d1 = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0)
    d2 = build_d2_bloch_exact(cfi, face_edges, d1, nC, nF)

    res = {}
    res['d1d0'] = norm(d1 @ d0)
    res['d2d1'] = norm(d2 @ d1)

    # Betti at generic k
    r0 = np.linalg.matrix_rank(d0, tol=1e-10)
    r1 = np.linalg.matrix_rank(d1, tol=1e-10)
    res['b0_k'] = nV - r0
    res['b1_k'] = (nE - r1) - r0

    # Betti at Gamma
    d0_g = build_d0_bloch(V, E, np.zeros(3), L_vec, shifts)
    d1_g = build_d1_bloch_exact(V, E, F, np.zeros(3), L_vec, d0_g)
    r0_g = np.linalg.matrix_rank(d0_g, tol=1e-10)
    r1_g = np.linalg.matrix_rank(d1_g, tol=1e-10)
    res['b0_G'] = nV - r0_g
    res['b1_G'] = (nE - r1_g) - r0_g
    res['d0_g'] = d0_g
    res['d1_g'] = d1_g

    # Hodge stars — note: build_hodge_stars_voronoi uses perturbed V for primal
    # face areas but the original cell_centers for dual edge lengths (cell_centers
    # are precomputed and not recomputed from V).  This inconsistency doesn't
    # affect test conclusions: orthogonality depends on d₁d₀=0, not on M₁ accuracy.
    star1, star2 = build_hodge_stars_voronoi(data_p)
    M0 = np.ones(nV) * (L_vec[0]**3 / nV)
    M1, M2 = star1, star2

    # Hodge decomposition
    M0s, M1s, M2s = np.sqrt(M0), np.sqrt(M1), np.sqrt(M2)
    B0 = np.diag(M1s) @ d0 @ np.diag(1.0 / M0s)
    B1 = np.diag(M2s) @ d1 @ np.diag(1.0 / M1s)

    I = np.eye(nE, dtype=complex)
    U0, s0, _ = svd(B0, full_matrices=False)
    r0_svd = int(np.sum(s0 > 1e-10))
    P_E = U0[:, :r0_svd] @ U0[:, :r0_svd].conj().T

    U1H, s1H, _ = svd(B1.conj().T, full_matrices=False)
    r1_svd = int(np.sum(s1H > 1e-10))
    P_C = U1H[:, :r1_svd] @ U1H[:, :r1_svd].conj().T

    res['orth_EC'] = norm(P_E @ P_C)
    res['dim_E'] = r0_svd
    res['dim_C'] = r1_svd
    res['dim_H'] = nE - r0_svd - r1_svd

    # Spectral pairing (Δ₀ ↔ Δ₁)
    D0 = B0.conj().T @ B0
    D1 = B0 @ B0.conj().T + B1.conj().T @ B1
    e0 = np.sort(np.real(eigvalsh(D0)))[::-1]
    e1v = np.sort(np.real(eigvalsh(D1)))[::-1]
    nz0 = e0[e0 > 1e-10]
    pool = list(e1v[e1v > 1e-10])
    max_pair = 0.0
    for ev in nz0:
        diffs = [abs(ev - p) for p in pool]
        best = np.argmin(diffs)
        max_pair = max(max_pair, diffs[best])
        pool.pop(best)
    res['pair_res'] = max_pair

    res['nV'] = nV
    res['_M0'] = M0
    res['_M1'] = M1
    res['_M2'] = M2
    res['mean_edge'] = mean_edge
    res['eps'] = eps

    return res


# ── Tests ────────────────────────────────────────────────────────────────

def test_1_exactness_stable():
    """Exactness d₁d₀=0 and d₂d₁=0 invariant under vertex perturbation."""
    data = build_kelvin_with_dual_info(N=2, L_cell=1.0)

    print(f'  Kelvin N=2, testing exactness under vertex jitter')
    eps_fracs = [0.0, 0.01, 0.05, 0.10, 0.20]

    for eps in eps_fracs:
        res = perturb_and_test(data, eps)
        print(f'    eps={eps:5.2f}: ||d1d0||={res["d1d0"]:.1e}  ||d2d1||={res["d2d1"]:.1e}')
        assert res['d1d0'] < 1e-12, f'Exactness broken at eps={eps}'
        assert res['d2d1'] < 1e-12, f'd2d1 broken at eps={eps}'

    print('  PASS')


def test_2_betti_stable():
    """Betti numbers invariant under vertex perturbation."""
    data = build_kelvin_with_dual_info(N=2, L_cell=1.0)

    print(f'  Kelvin N=2, testing Betti stability')
    eps_fracs = [0.0, 0.01, 0.05, 0.10, 0.20]

    for eps in eps_fracs:
        res = perturb_and_test(data, eps)
        b_k = (res['b0_k'], res['b1_k'])
        b_G = (res['b0_G'], res['b1_G'])
        print(f'    eps={eps:5.2f}: β(k)={b_k}  β(Γ)={b_G}')
        assert b_k == (0, 0), f'Betti at k wrong: {b_k} at eps={eps}'
        assert b_G == (1, 3), f'Betti at Gamma wrong: {b_G} at eps={eps}'

    print('  PASS')


def test_3_hodge_stable():
    """Hodge decomposition orthogonality invariant under perturbation.

    Checks both generic k (dim_H=0) and Γ (dim_H=3).
    """
    data = build_kelvin_with_dual_info(N=2, L_cell=1.0)

    print(f'  Kelvin N=2, testing Hodge decomposition stability')
    print(f'  {"eps":>5s}  {"dim_E":>5s} {"dim_H":>5s} {"dim_C":>5s}  {"||PE·PC||":>10s}'
          f'  {"dim_H(Γ)":>8s}')
    eps_fracs = [0.0, 0.01, 0.05, 0.10, 0.20]

    for eps in eps_fracs:
        res = perturb_and_test(data, eps)

        # Hodge decomposition at Γ (reuse d0_g, d1_g from helper)
        nE = len(data['E'])
        M0s, M1s, M2s = np.sqrt(res['_M0']), np.sqrt(res['_M1']), np.sqrt(res['_M2'])
        B0g = np.diag(M1s) @ res['d0_g'] @ np.diag(1.0 / M0s)
        B1g = np.diag(M2s) @ res['d1_g'] @ np.diag(1.0 / M1s)
        U0g, s0g, _ = svd(B0g, full_matrices=False)
        r0g = int(np.sum(s0g > 1e-10))
        U1Hg, s1Hg, _ = svd(B1g.conj().T, full_matrices=False)
        r1g = int(np.sum(s1Hg > 1e-10))
        dim_H_G = nE - r0g - r1g

        print(f'  {eps:5.2f}  {res["dim_E"]:5d} {res["dim_H"]:5d} {res["dim_C"]:5d}'
              f'  {res["orth_EC"]:10.2e}  {dim_H_G:8d}')
        nV = res['nV']
        assert res['dim_E'] == nV, f'dim_E={res["dim_E"]} != nV={nV} at eps={eps}'
        assert res['dim_H'] == 0, f'dim_H wrong at eps={eps}'
        assert res['dim_C'] == nV, f'dim_C={res["dim_C"]} != nV={nV} at eps={eps}'
        assert res['orth_EC'] < 1e-12, f'Orthogonality broken at eps={eps}'
        assert dim_H_G == 3, f'dim_H(Γ)={dim_H_G} != 3 at eps={eps}'

    print('  PASS')


def test_4_pairing_stable():
    """Spectral pairing residual stable under perturbation."""
    data = build_kelvin_with_dual_info(N=2, L_cell=1.0)

    print(f'  Kelvin N=2, testing spectral pairing stability')
    eps_fracs = [0.0, 0.01, 0.05, 0.10, 0.20]

    for eps in eps_fracs:
        res = perturb_and_test(data, eps)
        print(f'    eps={eps:5.2f}: pair_res={res["pair_res"]:.2e}')
        assert res['pair_res'] < 1e-10, f'Pairing broken at eps={eps}'

    print('  PASS')


def test_5_multiple_seeds():
    """Stability holds across different random seeds at 10% perturbation."""
    data = build_kelvin_with_dual_info(N=2, L_cell=1.0)

    print(f'  Kelvin N=2, eps=0.10, testing 5 random seeds')
    all_pass = True
    for seed in [1, 17, 42, 99, 256]:
        res = perturb_and_test(data, 0.10, seed=seed)
        b_k = (res['b0_k'], res['b1_k'])
        ok = (res['d1d0'] < 1e-12 and b_k == (0, 0)
              and res['orth_EC'] < 1e-12 and res['pair_res'] < 1e-10)
        status = 'ok' if ok else 'FAIL'
        print(f'    seed={seed:3d}: ||d1d0||={res["d1d0"]:.1e}'
              f'  ||PE·PC||={res["orth_EC"]:.1e}  pair={res["pair_res"]:.1e}  [{status}]')
        if not ok:
            all_pass = False

    assert all_pass
    print('  PASS')


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [test_1_exactness_stable, test_2_betti_stable,
             test_3_hodge_stable, test_4_pairing_stable,
             test_5_multiple_seeds]

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
