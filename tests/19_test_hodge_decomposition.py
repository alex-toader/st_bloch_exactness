"""
Discrete Hodge decomposition on 1-forms across the Brillouin zone (DIR-M12).

On an exact cochain complex 0 → C⁰ →d₀→ C¹ →d₁→ C² → ..., the space of
1-cochains splits into three M₁-orthogonal subspaces:

    C¹ = im(d₀) ⊕ H¹ ⊕ im(d₁*)

where d₁* = M₁⁻¹ d₁ᴴ M₂ is the Hodge adjoint, and H¹ = ker(Δ₁) is the
space of harmonic 1-forms.  This is verified via SVD-based subspace projections:

    P_E = proj onto im(B₀),    B₀ = M₁^{1/2} d₀ M₀^{-1/2}
    P_C = proj onto im(B₁ᴴ),   B₁ = M₂^{1/2} d₁ M₁^{-1/2}
    P_H = I - P_E - P_C

The decomposition holds iff:
    (1) P_E + P_H + P_C = 0        (resolution of identity — by construction)
    (2) P_E · P_C = 0               (exact ⊥ coexact)
    (3) P_X² = P_X for X ∈ {E,C,H} (idempotency)
    (4) P_H ≥ 0                     (harmonic projection is PSD)
    (5) rank(P_H) = β₁(k)           (harmonic dimension = Betti number)

At generic k: β₁ = 0, so C¹ = im(d₀) ⊕ im(d₁*), dim = V + (E-V) = E.
At Γ:        β₁ = 3, so C¹ = im(d₀) ⊕ H¹ ⊕ im(d₁*), dim = (V-1) + 3 + (E-V-2) = E.

The standard complex (d₁d₀ ≠ 0) does NOT admit a Hodge decomposition:
im(B₀) and im(B₁ᴴ) overlap, P_H = I - P_E - P_C has negative eigenvalues.

RAW OUTPUT:

  Kelvin N=2 at generic k=(0.17,0.23,0.31):
    dim_E=96, dim_H=0, dim_C=96, sum=192=nE
    ||P_E·P_C|| = 1.16e-14
    P_E idempotent: 1.94e-14,  P_C idempotent: 1.71e-14
    P_H max |eigenvalue|: 4.75e-15 (zero matrix — no harmonic forms)

  Kelvin N=2 at Γ:
    dim_E=95, dim_H=3, dim_C=94, sum=192=nE
    ||P_E·P_C|| = 8.50e-15
    P_H eigenvalues: 3 at 1.0, rest < 4e-15
    Harmonic subspace matches ker(Δ₁): ||P_H·V_harm - V_harm|| = 3.41e-15

  Kelvin N=2 standard complex at generic k:
    ||P_E·P_C|| = 5.38  (subspaces overlap)
    P_H min eigenvalue = -1.00  (NOT a projector)

  WP N=1: decomposition holds at Γ, generic k, R

ANSWER:
  The exact Bloch-DEC complex admits a clean Hodge decomposition at every
  k-point in the Brillouin zone.  The three subspaces are mutually orthogonal
  to machine precision (||P_E·P_C|| < 2e-14), dimensions match the Betti
  numbers from test 13, and P_H is a genuine projection.  The standard complex
  fails qualitatively: subspaces overlap and P_H has negative eigenvalues.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/19_test_hodge_decomposition.py
"""

import sys, os
import numpy as np
from numpy.linalg import norm, svd, eigvalsh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import (build_kelvin_with_dual_info, build_wp_with_dual_info,
                           build_c15_with_dual_info)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup
from physics.bloch_complex import build_d2_bloch_exact, load_foam


def build_operators(mesh, k_vec, d1_builder='exact'):
    """Build differential operators and symmetrized B operators at k-point.

    Returns (B0, B1, d0, d1, d2).
    B0: nE × nV, B1: nF × nE — the symmetrized differentials acting on 1-forms.
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

    B0 = np.diag(M1s) @ d0 @ np.diag(1.0 / M0s)   # nE × nV
    B1 = np.diag(M2s) @ d1 @ np.diag(1.0 / M1s)   # nF × nE

    return B0, B1, d0, d1, d2


def hodge_decomposition(B0, B1, nE):
    """Compute Hodge decomposition projections on 1-form space.

    Returns (P_E, P_C, P_H, dims) where:
      P_E: projection onto im(B₀) — exact 1-forms
      P_C: projection onto im(B₁ᴴ) — coexact 1-forms
      P_H: I - P_E - P_C — harmonic 1-forms
      dims: (dim_E, dim_H, dim_C)
    """
    I = np.eye(nE, dtype=complex)

    # P_exact = projection onto column space of B0
    U0, s0, _ = svd(B0, full_matrices=False)
    r0 = int(np.sum(s0 > 1e-10))
    P_E = U0[:, :r0] @ U0[:, :r0].conj().T

    # P_coexact = projection onto column space of B1^H
    U1H, s1H, _ = svd(B1.conj().T, full_matrices=False)
    r1 = int(np.sum(s1H > 1e-10))
    P_C = U1H[:, :r1] @ U1H[:, :r1].conj().T

    P_H = I - P_E - P_C

    dim_E = r0
    dim_C = r1
    dim_H = nE - r0 - r1

    return P_E, P_C, P_H, (dim_E, dim_H, dim_C)


def check_projections(P_E, P_C, P_H, nE, label=''):
    """Check all projection properties, return dict of residuals."""
    I = np.eye(nE, dtype=complex)
    res = {}

    # Resolution of identity
    res['identity'] = norm(P_E + P_C + P_H - I)

    # Idempotency
    res['idem_E'] = norm(P_E @ P_E - P_E)
    res['idem_C'] = norm(P_C @ P_C - P_C)
    res['idem_H'] = norm(P_H @ P_H - P_H)

    # Mutual orthogonality (Frobenius norm — can exceed 1 on non-exact
    # complexes because it sums over all principal angles between subspaces;
    # the spectral norm ||P_E·P_C||₂ ≤ 1 always for orthogonal projections)
    res['orth_EC'] = norm(P_E @ P_C)
    res['orth_EH'] = norm(P_E @ P_H)
    res['orth_CH'] = norm(P_C @ P_H)

    # P_H positive semi-definite
    eigs_H = np.sort(np.real(eigvalsh(P_H)))
    res['PH_min_eig'] = eigs_H[0]
    res['PH_max_eig'] = eigs_H[-1]

    return res


# ── Tests ────────────────────────────────────────────────────────────────

def test_1_decomposition_generic_k():
    """Hodge decomposition at generic k: C¹ = im(d₀) ⊕ im(d₁*), no harmonics.

    Note: im(B₀) ⊥ im(B₁ᴴ) ⟺ B₁B₀ = M₂^{1/2}(d₁d₀)M₀^{-1/2} = 0, so
    ||P_E·P_C|| = 0 is algebraically equivalent to d₁d₀ = 0.  The value of
    this test is not the identity itself but its consequence: the orthogonal
    Hodge decomposition with correct dimensions.
    """
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    nE = len(mesh['E'])

    k_frac = np.array([0.17, 0.23, 0.31])
    k_vec = 2 * np.pi * k_frac / mesh['L_vec']

    B0, B1, d0, d1, d2 = build_operators(mesh, k_vec, 'exact')

    # Verify exactness
    assert norm(d1 @ d0) < 1e-10, f'||d1·d0|| = {norm(d1 @ d0):.2e}'
    assert norm(d2 @ d1) < 1e-10, f'||d2·d1|| = {norm(d2 @ d1):.2e}'

    P_E, P_C, P_H, (dim_E, dim_H, dim_C) = hodge_decomposition(B0, B1, nE)
    res = check_projections(P_E, P_C, P_H, nE)

    nV, nC = len(mesh['V']), mesh['nC']

    print(f'  Kelvin N=2 at k_frac={k_frac}')
    print(f'  dim_E={dim_E}, dim_H={dim_H}, dim_C={dim_C}, sum={dim_E+dim_H+dim_C}')
    print(f'  ||P_E·P_C|| = {res["orth_EC"]:.2e}')
    print(f'  idem_E={res["idem_E"]:.2e}  idem_C={res["idem_C"]:.2e}  idem_H={res["idem_H"]:.2e}')
    print(f'  P_H max |eig| = {max(abs(res["PH_min_eig"]), abs(res["PH_max_eig"])):.2e}')

    # Dimensions: at generic k, acyclic → no harmonics
    assert dim_E == nV, f'dim_E should be V={nV}, got {dim_E}'
    assert dim_C == nE - nV, f'dim_C should be E-V={nE-nV}, got {dim_C}'
    assert dim_H == 0, f'dim_H should be 0 at generic k, got {dim_H}'
    assert dim_E + dim_H + dim_C == nE

    # Projection properties
    tol = 1e-12
    assert res['identity'] < tol, f'P_E+P_C+P_H != I: {res["identity"]:.2e}'
    assert res['orth_EC'] < tol, f'P_E·P_C != 0: {res["orth_EC"]:.2e}'
    assert res['idem_E'] < tol, f'P_E not idempotent: {res["idem_E"]:.2e}'
    assert res['idem_C'] < tol, f'P_C not idempotent: {res["idem_C"]:.2e}'
    assert res['idem_H'] < tol, f'P_H not idempotent: {res["idem_H"]:.2e}'
    assert abs(res['PH_max_eig']) < tol, f'P_H not zero at generic k: max eig = {res["PH_max_eig"]:.2e}'
    print('  PASS')


def test_2_decomposition_gamma():
    """Hodge decomposition at Γ: 3 harmonic 1-forms matching β₁ = 3."""
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    nE = len(mesh['E'])
    nV = len(mesh['V'])

    B0, B1, d0, d1, d2 = build_operators(mesh, np.zeros(3), 'exact')

    P_E, P_C, P_H, (dim_E, dim_H, dim_C) = hodge_decomposition(B0, B1, nE)
    res = check_projections(P_E, P_C, P_H, nE)

    print(f'  Kelvin N=2 at Γ')
    print(f'  dim_E={dim_E}, dim_H={dim_H}, dim_C={dim_C}, sum={dim_E+dim_H+dim_C}')
    print(f'  ||P_E·P_C|| = {res["orth_EC"]:.2e}')
    print(f'  P_H eigs: min={res["PH_min_eig"]:.2e}, max={res["PH_max_eig"]:.2e}')

    # Dimensions at Γ
    assert dim_E == nV - 1, f'dim_E should be V-1={nV-1}, got {dim_E}'
    assert dim_H == 3, f'dim_H should be 3 (β₁), got {dim_H}'
    assert dim_C == nE - nV - 2, f'dim_C should be E-V-2={nE-nV-2}, got {dim_C}'

    # Projection properties
    tol = 1e-12
    assert res['identity'] < tol, f'P_E+P_C+P_H != I: {res["identity"]:.2e}'
    assert res['orth_EC'] < tol, f'P_E·P_C != 0: {res["orth_EC"]:.2e}'
    assert res['idem_H'] < tol, f'P_H not idempotent: {res["idem_H"]:.2e}'
    assert res['PH_min_eig'] > -tol, f'P_H not PSD: min eig = {res["PH_min_eig"]:.2e}'
    assert abs(res['PH_max_eig'] - 1.0) < tol, f'P_H max eig should be 1: {res["PH_max_eig"]:.2e}'

    # Harmonic subspace should match ker(Δ₁) from eigenvector computation
    M1s = np.sqrt(mesh['M1'])
    D1 = B0 @ B0.conj().T + B1.conj().T @ B1
    eigs_D1, vecs_D1 = np.linalg.eigh(D1)
    # Extract zero-eigenvalue eigenvectors
    zero_mask = np.abs(eigs_D1) < 1e-10
    n_zero = int(np.sum(zero_mask))
    assert n_zero == 3, f'ker(Δ₁) should have dim 3, got {n_zero}'
    V_harm = vecs_D1[:, zero_mask]  # nE × 3

    # Check: P_H should project onto the same subspace as V_harm
    # ||P_H · V_harm - V_harm|| should be zero
    residual = norm(P_H @ V_harm - V_harm)
    print(f'  ||P_H · V_harm - V_harm|| = {residual:.2e}')
    assert residual < tol, f'Harmonic subspace mismatch: {residual:.2e}'

    print('  PASS')


def test_3_standard_fails():
    """Standard complex: Hodge decomposition fails — subspaces overlap, P_H not PSD.

    The failure is im(d₀) ⊄ ker(d₁_std): since d₁_std·d₀ ≠ 0, the exact and
    coexact subspaces overlap and P_H = I - P_E - P_C acquires negative eigenvalues.
    ||P_E·P_C||_F (Frobenius norm) can exceed 1 because it sums over all
    principal angles between the subspaces.
    """
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    nE = len(mesh['E'])

    k_frac = np.array([0.17, 0.23, 0.31])
    k_vec = 2 * np.pi * k_frac / mesh['L_vec']

    # Exact complex
    B0_ex, B1_ex, _, _, _ = build_operators(mesh, k_vec, 'exact')
    P_E_ex, P_C_ex, P_H_ex, _ = hodge_decomposition(B0_ex, B1_ex, nE)
    res_ex = check_projections(P_E_ex, P_C_ex, P_H_ex, nE)

    # Standard complex
    B0_std, B1_std, d0, d1_std, _ = build_operators(mesh, k_vec, 'standard')
    P_E_std, P_C_std, P_H_std, _ = hodge_decomposition(B0_std, B1_std, nE)
    res_std = check_projections(P_E_std, P_C_std, P_H_std, nE)

    print(f'  ||P_E·P_C||:  exact={res_ex["orth_EC"]:.2e}  standard={res_std["orth_EC"]:.2e}')
    print(f'  P_H min eig:   exact={res_ex["PH_min_eig"]:.2e}  standard={res_std["PH_min_eig"]:.2e}')
    print(f'  idem_H:        exact={res_ex["idem_H"]:.2e}  standard={res_std["idem_H"]:.2e}')

    # Exact: clean decomposition
    assert res_ex['orth_EC'] < 1e-12
    assert res_ex['idem_H'] < 1e-12

    # Standard: subspaces overlap
    assert res_std['orth_EC'] > 0.01, \
        f'Standard ||P_E·P_C|| unexpectedly small: {res_std["orth_EC"]:.2e}'

    # Standard: P_H has negative eigenvalues (not a genuine projection)
    assert res_std['PH_min_eig'] < -0.01, \
        f'Standard P_H min eig unexpectedly non-negative: {res_std["PH_min_eig"]:.2e}'

    print('  PASS')


def test_4_bz_path():
    """Hodge decomposition holds along full BZ path Γ-X-M-R-Γ."""
    mesh = load_foam(build_kelvin_with_dual_info, N=2, L_cell=1.0, with_stars=True)
    nE = len(mesh['E'])
    nV = len(mesh['V'])
    L = mesh['L']
    k_scale = 2 * np.pi / L

    k_points = [
        ('Γ',       np.zeros(3)),
        ('0.1[100]', k_scale * 0.10 * np.array([1, 0, 0.])),
        ('X',       k_scale * np.array([0.5, 0, 0.])),
        ('M',       k_scale * np.array([0.5, 0.5, 0.])),
        ('R',       k_scale * np.array([0.5, 0.5, 0.5])),
        ('generic', 2 * np.pi * np.array([0.17, 0.23, 0.31]) / mesh['L_vec']),
    ]

    print(f'  {"k":>10s}  {"dim_E":>5s} {"dim_H":>5s} {"dim_C":>5s}'
          f'  {"||PE·PC||":>10s} {"idem_H":>8s} {"PH_min":>8s}')

    all_pass = True
    for label, k_vec in k_points:
        B0, B1, d0, d1, d2 = build_operators(mesh, k_vec, 'exact')
        assert norm(d1 @ d0) < 1e-10
        assert norm(d2 @ d1) < 1e-10

        P_E, P_C, P_H, (dim_E, dim_H, dim_C) = hodge_decomposition(B0, B1, nE)
        res = check_projections(P_E, P_C, P_H, nE)

        is_gamma = np.allclose(k_vec, 0)
        expected_H = 3 if is_gamma else 0

        ok = (res['orth_EC'] < 1e-12
              and res['idem_H'] < 1e-12
              and dim_H == expected_H
              and res['PH_min_eig'] > -1e-12)

        status = 'ok' if ok else 'FAIL'
        print(f'  {label:>10s}  {dim_E:5d} {dim_H:5d} {dim_C:5d}'
              f'  {res["orth_EC"]:10.2e} {res["idem_H"]:8.2e} {res["PH_min_eig"]:8.2e}'
              f'  [{status}]')

        if not ok:
            all_pass = False

    assert all_pass, 'Hodge decomposition failed at some k-point'
    print('  PASS')


def test_5_universality():
    """Hodge decomposition holds on WP and C15 at Γ and generic k."""
    structures = [
        ('WP N=1',  build_wp_with_dual_info,  1, 1.0),
        ('C15 N=1', build_c15_with_dual_info, 1, 1.0),
    ]

    all_pass = True
    for name, builder, N, Lc in structures:
        mesh = load_foam(builder, N=N, L_cell=Lc, with_stars=True)
        nE = len(mesh['E'])

        print(f'  {name} (V={len(mesh["V"])} E={nE} F={len(mesh["F"])} C={mesh["nC"]})')

        k_points = {
            'Γ':       np.zeros(3),
            'generic': 2 * np.pi * np.array([0.17, 0.23, 0.31]) / mesh['L_vec'],
            'R':       2 * np.pi * np.array([0.5, 0.5, 0.5]) / mesh['L_vec'],
        }

        for label, k_vec in k_points.items():
            B0, B1, d0, d1, d2 = build_operators(mesh, k_vec, 'exact')
            assert norm(d1 @ d0) < 1e-10
            assert norm(d2 @ d1) < 1e-10

            P_E, P_C, P_H, (dim_E, dim_H, dim_C) = hodge_decomposition(B0, B1, nE)
            res = check_projections(P_E, P_C, P_H, nE)

            is_gamma = np.allclose(k_vec, 0)
            expected_H = 3 if is_gamma else 0

            ok = (res['orth_EC'] < 1e-12
                  and res['idem_H'] < 1e-12
                  and dim_H == expected_H
                  and res['PH_min_eig'] > -1e-12)

            status = 'ok' if ok else 'FAIL'
            print(f'    {label:8s}: dim=({dim_E},{dim_H},{dim_C})'
                  f'  ||PE·PC||={res["orth_EC"]:.2e}  [{status}]')
            if not ok:
                all_pass = False

    assert all_pass, 'Hodge decomposition failed on some structure'
    print('  PASS')


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [test_1_decomposition_generic_k, test_2_decomposition_gamma,
             test_3_standard_fails, test_4_bz_path, test_5_universality]

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
