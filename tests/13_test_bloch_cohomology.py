"""
Discrete Bloch cohomology: the exact complex reproduces H*(T³, L_χ).

The Bloch character χ_k: π₁(T³) → U(1) defined by χ(γᵢ) = e^{ikᵢLᵢ} specifies
a unitary flat line bundle L_χ over T³. The exactness-preserving DEC complex
    0 → C⁰ →d₀(k)→ C¹ →d₁(k)→ C² →d₂(k)→ C³ → 0
is a genuine cochain complex at all k (d₁d₀=0, d₂d₁=0), computing the twisted
cohomology H^p(T³, L_χ) with well-defined Betti numbers β_p(k) = dim H^p(k).

The standard (broken) Bloch complex has d₁d₀ ≠ 0 at k≠0, so it is NOT a cochain
complex and cohomology is undefined — not wrong, but nonexistent.

Key results:
  - β(Γ) = (1, 3, 3, 1) = H*(T³, ℝ)  [trivial character, de Rham cohomology]
  - β(k≠0) = (0, 0, 0, 0)  [nontrivial unitary character, monodromy has no
    fixed vectors, twisted cohomology acyclic]
  - Euler χ = V - E + F - C = 0 everywhere (topological invariant)
  - Transition is sharp: only at Γ, not at BZ boundary (X, M, R)
  - Numerical ranks computed with tolerance 1e-10; results stable under
    moderate variation (min singular value at k_frac=1e-4 is 9.1e-5 >> tol)

Physical interpretation at Γ:
  β₀ = 1: one constant 0-form (connected component)
  β₁ = 3: three harmonic 1-forms (constant E-field polarizations)
  β₂ = 3: three harmonic 2-forms (constant B-flux directions)
  β₃ = 1: one volume 3-form

Rank structure (universal across all tested topologies):
  At Γ:   r₀ = V-1, r₁ = E-V-2, r₂ = C-1  → β = (1,3,3,1)
  At k≠0: r₀ = V,   r₁ = E-V,   r₂ = C    → β = (0,0,0,0) = acyclic

  The rank deficiency at Γ comes from constant cochains that survive d_p at k=0
  (because e^{i·0·n}=1) but are killed at k≠0 (because e^{ik·n}≠1).

RAW OUTPUT:

  SC N=3 (V=27, E=81, F=81, C=27):
    Gamma: beta=(1,3,3,1)  ranks=(26,52,26)
    X:     beta=(0,0,0,0)  ranks=(27,54,27)

  Kelvin N=2 (V=96, E=192, F=112, C=16):
    Gamma: beta=(1,3,3,1)  ranks=(95,94,15)
    X:     beta=(0,0,0,0)  ranks=(96,96,16)

  C15 N=1 (V=136, E=272, F=160, C=24):
    Gamma: beta=(1,3,3,1)  ranks=(135,134,23)
    X:     beta=(0,0,0,0)  ranks=(136,136,24)

  WP N=1 (V=46, E=92, F=54, C=8):
    Gamma: beta=(1,3,3,1)  ranks=(45,44,7)
    X:     beta=(0,0,0,0)  ranks=(46,46,8)

  Fine BZ scan (Kelvin, 36 k-points on Gamma-X-M-R-Gamma):
    beta = (1,3,3,1) ONLY at Gamma, (0,0,0,0) everywhere else.

  BZ boundary sensitivity (Kelvin, k_frac → 0):
    min_sv(d0) = 9.1e-5 at k_frac=1e-4, >> tol=1e-10. Betti correct.

  Standard complex: NOT a cochain complex at k≠0 (||d1d0|| ~ O(k)).
    Rank of d1_std jumps erratically (94 to 102 on Kelvin).
    Cohomology is undefined.

ANSWER:
  The exact discrete Bloch complex faithfully reproduces the cohomology of
  the unitary flat line bundle associated with the Bloch character χ_k on T³.
  The standard complex cannot even define cohomology at k≠0. This is the
  strongest validation of the exactness-preserving construction: correct
  topology, not just d₁d₀=0.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/13_test_bloch_cohomology.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from core_math.builders.solids_periodic import build_sc_solid_periodic
from core_math.operators.incidence import build_d2
from physics.hodge import build_kelvin_with_dual_info, build_c15_with_dual_info, build_wp_with_dual_info
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch_complex import (build_face_edge_map, build_cell_face_incidence,
                                   build_d2_bloch_exact, load_foam)


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_betti(V, E, F, L_vec, cfi, face_edges, k_vec, nC):
    """Compute Betti numbers β₀, β₁, β₂, β₃ at given k-point.

    Returns (b0, b1, b2, b3, r0, r1, r2).
    """
    nV, nE, nF = len(V), len(E), len(F)
    shifts = compute_edge_shifts(V, E, L_vec)

    d0_k = build_d0_bloch(V, E, k_vec, L_vec, shifts)
    d1_k = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0_k)
    d2_k = build_d2_bloch_exact(cfi, face_edges, d1_k, nC, nF)

    # Verify exactness
    assert np.linalg.norm(d1_k @ d0_k) < 1e-10, \
        f'd1d0 != 0: {np.linalg.norm(d1_k @ d0_k):.2e}'
    assert np.linalg.norm(d2_k @ d1_k) < 1e-10, \
        f'd2d1 != 0: {np.linalg.norm(d2_k @ d1_k):.2e}'

    r0 = np.linalg.matrix_rank(d0_k, tol=1e-10)
    r1 = np.linalg.matrix_rank(d1_k, tol=1e-10)
    r2 = np.linalg.matrix_rank(d2_k, tol=1e-10)

    b0 = nV - r0
    b1 = (nE - r1) - r0
    b2 = (nF - r2) - r1
    b3 = nC - r2

    return b0, b1, b2, b3, r0, r1, r2


def _load_foam(builder, N, L_cell):
    """Load foam and prepare d2 infrastructure (returns tuple for legacy compat)."""
    mesh = load_foam(builder, N, L_cell)
    return (mesh['V'], mesh['E'], mesh['F'], mesh['L'], mesh['L_vec'],
            mesh['nC'], mesh['cfi'], mesh['face_edges'], mesh['n_flipped'])


def _load_sc():
    """Load SC cubic and prepare d2 infrastructure."""
    mesh = build_sc_solid_periodic(N=3)
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    L = mesh['period_L']
    L_vec = np.array([L, L, L])
    cfi = mesh['cell_face_incidence']
    nC = mesh['n_cells']
    d2_top = build_d2(cfi, len(F))
    face_edges = build_face_edge_map(F, E)
    return V, E, F, L, L_vec, nC, cfi, face_edges, 0


# ── Test 1: Betti numbers on all structures ──────────────────────────────

def test_betti_all_structures():
    """β(Γ) = (1,3,3,1) and β(k≠0) = (0,0,0,0) on all 4 structures."""
    print(f"\n{'=' * 70}")
    print(f"  DISCRETE BLOCH COHOMOLOGY — ALL STRUCTURES")
    print(f"{'=' * 70}")

    structures = [
        ('SC N=3', _load_sc),
        ('Kelvin N=2', lambda: _load_foam(build_kelvin_with_dual_info, 2, 4.0)),
        ('C15 N=1', lambda: _load_foam(build_c15_with_dual_info, 1, 4.0)),
        ('WP N=1', lambda: _load_foam(build_wp_with_dual_info, 1, 4.0)),
    ]

    for name, loader in structures:
        V, E, F, L, L_vec, nC, cfi, face_edges, n_flipped = loader()
        nV, nE, nF = len(V), len(E), len(F)
        chi = nV - nE + nF - nC

        flip_str = f", d2 flips={n_flipped}/{nF}" if n_flipped > 0 else ""
        print(f"\n  {name}: V={nV}, E={nE}, F={nF}, C={nC}, chi={chi}{flip_str}")
        assert chi == 0, f"Euler characteristic should be 0, got {chi}"

        k_scale = 2 * np.pi / L

        k_points = [
            ('Gamma', np.zeros(3)),
            ('0.10 [100]', k_scale * 0.10 * np.array([1, 0, 0.])),
            ('X', k_scale * 0.50 * np.array([1, 0, 0.])),
            ('M', k_scale * np.array([0.5, 0.5, 0.])),
            ('R', k_scale * np.array([0.5, 0.5, 0.5])),
        ]

        for label, kv in k_points:
            b0, b1, b2, b3, r0, r1, r2 = compute_betti(
                V, E, F, L_vec, cfi, face_edges, kv, nC)

            # Euler check
            assert b0 - b1 + b2 - b3 == 0, \
                f"{name} at {label}: Euler check failed"

            is_gamma = np.allclose(kv, 0)
            if is_gamma:
                assert (b0, b1, b2, b3) == (1, 3, 3, 1), \
                    f"{name} at Gamma: expected (1,3,3,1), got ({b0},{b1},{b2},{b3})"
            else:
                assert (b0, b1, b2, b3) == (0, 0, 0, 0), \
                    f"{name} at {label}: expected (0,0,0,0), got ({b0},{b1},{b2},{b3})"

            print(f"    {label:>12s}: beta=({b0},{b1},{b2},{b3})"
                  f"  ranks=({r0},{r1},{r2})")

    print(f"\n  All structures: beta(Gamma) = (1,3,3,1), beta(k!=0) = (0,0,0,0).")


# ── Test 2: Fine BZ scan ─────────────────────────────────────────────────

def test_bz_scan():
    """β = (0,0,0,0) across entire BZ path except at Γ."""
    print(f"\n{'=' * 70}")
    print(f"  FINE BZ SCAN — Kelvin N=2")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, nC, cfi, face_edges, _ = \
        _load_foam(build_kelvin_with_dual_info, 2, 4.0)

    k_scale = 2 * np.pi / L
    N_pts = 8

    segments = [
        ('Gamma-X', np.array([0, 0, 0.]), np.array([0.5, 0, 0.])),
        ('X-M', np.array([0.5, 0, 0.]), np.array([0.5, 0.5, 0.])),
        ('M-R', np.array([0.5, 0.5, 0.]), np.array([0.5, 0.5, 0.5])),
        ('R-Gamma', np.array([0.5, 0.5, 0.5]), np.array([0, 0, 0.])),
    ]

    n_nontrivial = 0
    n_acyclic = 0
    n_total = 0

    print(f"\n  {'segment':>12s} {'t':>5s} {'k_frac':>20s}"
          f"  {'b0':>3s} {'b1':>3s} {'b2':>3s} {'b3':>3s}")
    print(f"  {'-' * 60}")

    for seg_name, k_start, k_end in segments:
        for i in range(N_pts + 1):
            t = i / N_pts
            k_frac = (1 - t) * k_start + t * k_end
            k_vec = k_scale * k_frac

            b0, b1, b2, b3, _, _, _ = compute_betti(
                V, E, F, L_vec, cfi, face_edges, k_vec, nC)

            is_gamma = np.allclose(k_frac, 0)
            beta = (b0, b1, b2, b3)
            n_total += 1

            if beta != (0, 0, 0, 0):
                assert is_gamma, \
                    f"Non-trivial cohomology at non-Gamma k_frac={k_frac}: {beta}"
                assert beta == (1, 3, 3, 1), \
                    f"Wrong Betti at Gamma: {beta}"
                n_nontrivial += 1
                marker = ' <-- Gamma'
            else:
                n_acyclic += 1
                marker = ''

            k_str = f'({k_frac[0]:.2f},{k_frac[1]:.2f},{k_frac[2]:.2f})'
            print(f"  {seg_name:>12s} {t:5.2f} {k_str:>20s}"
                  f"  {b0:3d} {b1:3d} {b2:3d} {b3:3d}{marker}")

    # Path visits Gamma exactly twice: start of Gamma-X, end of R-Gamma.
    # If n_nontrivial < 2, one Gamma visit gave (0,0,0,0) — a serious bug.
    print(f"\n  Scanned {n_total} k-points: {n_nontrivial} with beta=(1,3,3,1),"
          f" {n_acyclic} acyclic (0,0,0,0).")
    assert n_nontrivial >= 2, \
        f"Path should visit Gamma at least twice, got {n_nontrivial} non-trivial"
    assert n_acyclic == n_total - n_nontrivial


# ── Test 3: Standard complex cannot define cohomology ─────────────────────

def test_standard_no_cohomology():
    """Standard complex: d₁d₀ ≠ 0 at k≠0, so cohomology is undefined."""
    print(f"\n{'=' * 70}")
    print(f"  STANDARD COMPLEX — NO COHOMOLOGY AT k≠0")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, nC, cfi, face_edges, _ = \
        _load_foam(build_kelvin_with_dual_info, 2, 4.0)
    nV, nE, nF = len(V), len(E), len(F)
    shifts = compute_edge_shifts(V, E, L_vec)

    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)
    k_scale = 2 * np.pi / L

    print(f"\n  {'k-point':>12s}  {'||d1d0||':>10s}"
          f"  {'r1_std':>6s} {'r1_exact':>8s}  {'complex?':>10s}")
    print(f"  {'-' * 55}")

    for label, k_vec in [
        ('Gamma', np.zeros(3)),
        ('0.05 [100]', k_scale * 0.05 * np.array([1, 0, 0.])),
        ('0.10 [100]', k_scale * 0.10 * np.array([1, 0, 0.])),
        ('X', k_scale * 0.50 * np.array([1, 0, 0.])),
        ('R', k_scale * np.array([0.5, 0.5, 0.5])),
    ]:
        d0_k = build_d0_bloch(V, E, k_vec, L_vec, shifts)
        d1_std = build_d1_bloch_standard(V, E, F, L, k_vec, edge_lookup, crossings)
        d1_ex = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0_k)

        norm_d1d0 = np.linalg.norm(d1_std @ d0_k)
        r1_std = np.linalg.matrix_rank(d1_std, tol=1e-10)
        r1_ex = np.linalg.matrix_rank(d1_ex, tol=1e-10)

        is_complex = norm_d1d0 < 1e-10
        status = 'YES' if is_complex else 'NO'

        print(f"  {label:>12s}  {norm_d1d0:10.2e}  {r1_std:6d} {r1_ex:8d}  {status:>10s}")

        if np.allclose(k_vec, 0):
            assert is_complex, "Standard should be a complex at k=0"
            assert r1_std == r1_ex, "Ranks should match at k=0"
        else:
            assert not is_complex, "Standard should NOT be a complex at k!=0"

    print(f"\n  Standard complex: d₁d₀ ~ O(k) at k≠0.")
    print(f"  Rank of d₁_std varies (94-102); d₁_exact is stable (96).")
    print(f"  Cohomology is undefined for the standard construction.")


# ── Test 4: Rank structure ────────────────────────────────────────────────

def test_rank_structure():
    """Verify rank identities at Γ and generic k."""
    print(f"\n{'=' * 70}")
    print(f"  RANK STRUCTURE IDENTITIES")
    print(f"{'=' * 70}")

    structures = [
        ('SC N=3', _load_sc),
        ('Kelvin N=2', lambda: _load_foam(build_kelvin_with_dual_info, 2, 4.0)),
        ('C15 N=1', lambda: _load_foam(build_c15_with_dual_info, 1, 4.0)),
        ('WP N=1', lambda: _load_foam(build_wp_with_dual_info, 1, 4.0)),
    ]

    for name, loader in structures:
        V, E, F, L, L_vec, nC, cfi, face_edges, _ = loader()
        nV, nE, nF = len(V), len(E), len(F)

        k_scale = 2 * np.pi / L

        # At Gamma
        b0, b1, b2, b3, r0, r1, r2 = compute_betti(
            V, E, F, L_vec, cfi, face_edges, np.zeros(3), nC)
        assert r0 == nV - 1, f"{name}: r0 at Gamma should be V-1"
        assert r2 == nC - 1, f"{name}: r2 at Gamma should be C-1"
        # r1 = E - V - 2 (from b1=3: r1 = E - r0 - b1 = E - (V-1) - 3 = E - V - 2)
        assert r1 == nE - nV - 2, f"{name}: r1 at Gamma should be E-V-2"

        # At generic k
        k_gen = k_scale * 0.10 * np.array([1, 0, 0.])
        b0, b1, b2, b3, r0, r1, r2 = compute_betti(
            V, E, F, L_vec, cfi, face_edges, k_gen, nC)
        assert r0 == nV, f"{name}: r0 at k!=0 should be V (full rank)"
        assert r1 == nE - nV, f"{name}: r1 at k!=0 should be E-V"
        assert r2 == nC, f"{name}: r2 at k!=0 should be C (full rank)"

        print(f"  {name:>12s}: Gamma r=(V-1, E-V-2, C-1),"
              f" k!=0 r=(V, E-V, C). VERIFIED.")

    print(f"\n  Rank structure is universal across all foam families.")


# ── Test 5: BZ boundary rank sensitivity ──────────────────────────────────

def test_bz_boundary_tolerance():
    """Betti numbers remain correct near BZ boundary where rank(d₀) may degrade."""
    print(f"\n{'=' * 70}")
    print(f"  BZ BOUNDARY RANK SENSITIVITY")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, nC, cfi, face_edges, _ = \
        _load_foam(build_kelvin_with_dual_info, 2, 4.0)

    k_scale = 2 * np.pi / L
    shifts = compute_edge_shifts(V, E, L_vec)

    # Test near BZ boundary: k_frac approaching 0.5 (X point)
    # At X, Bloch phase = e^{iπ} = -1, a non-trivial character.
    # Singular values of d₀ shrink as k→0 (approaching Γ), so test
    # both directions: approaching X from below and approaching Γ from above.
    # Note: k_frac=1.0 wraps to Γ (e^{i2π}=1), so we only go up to 0.5.
    print(f"\n  {'k_frac':>8s}  {'r0':>4s} {'r1':>4s} {'r2':>4s}"
          f"  {'b0':>3s} {'b1':>3s} {'b2':>3s} {'b3':>3s}"
          f"  {'min_sv(d0)':>12s}")
    print(f"  {'-' * 65}")

    # Near X point (k_frac → 0.5)
    for frac in [0.40, 0.45, 0.49, 0.499, 0.4999, 0.50]:
        k_vec = k_scale * frac * np.array([1, 0, 0.])

        d0_k = build_d0_bloch(V, E, k_vec, L_vec, shifts)
        sv = np.linalg.svd(d0_k, compute_uv=False)
        min_sv = sv[sv > 1e-14].min() if np.any(sv > 1e-14) else 0.0

        b0, b1, b2, b3, r0, r1, r2 = compute_betti(
            V, E, F, L_vec, cfi, face_edges, k_vec, nC)

        beta = (b0, b1, b2, b3)
        assert beta == (0, 0, 0, 0), \
            f"k_frac={frac}: expected (0,0,0,0), got {beta}"

        print(f"  {frac:8.4f}  {r0:4d} {r1:4d} {r2:4d}"
              f"  {b0:3d} {b1:3d} {b2:3d} {b3:3d}"
              f"  {min_sv:12.6f}  OK")

    # Near Gamma (k_frac → 0)
    print()
    for frac in [0.10, 0.01, 0.001, 0.0001]:
        k_vec = k_scale * frac * np.array([1, 0, 0.])

        d0_k = build_d0_bloch(V, E, k_vec, L_vec, shifts)
        sv = np.linalg.svd(d0_k, compute_uv=False)
        min_sv = sv[sv > 1e-14].min() if np.any(sv > 1e-14) else 0.0

        b0, b1, b2, b3, r0, r1, r2 = compute_betti(
            V, E, F, L_vec, cfi, face_edges, k_vec, nC)

        beta = (b0, b1, b2, b3)
        assert beta == (0, 0, 0, 0), \
            f"k_frac={frac}: expected (0,0,0,0), got {beta}"

        print(f"  {frac:8.4f}  {r0:4d} {r1:4d} {r2:4d}"
              f"  {b0:3d} {b1:3d} {b2:3d} {b3:3d}"
              f"  {min_sv:12.6f}  OK")

    print(f"\n  Betti numbers correct near both BZ boundary and Gamma.")
    print(f"  min_sv(d0) shrinks near Gamma but stays >> tol=1e-10.")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DISCRETE BLOCH COHOMOLOGY TESTS")
    print("Does the exact complex reproduce H*(T³, L_k)?")
    print("=" * 70)

    test_betti_all_structures()
    test_bz_scan()
    test_standard_no_cohomology()
    test_rank_structure()
    test_bz_boundary_tolerance()

    print(f"\n{'=' * 70}")
    print("ALL COHOMOLOGY TESTS PASSED.")
    print("The exact discrete Bloch complex faithfully reproduces")
    print("the cohomology of the unitary flat line bundle L_chi on T³.")
    print("=" * 70)


if __name__ == '__main__':
    main()
    print("\nDone.")
