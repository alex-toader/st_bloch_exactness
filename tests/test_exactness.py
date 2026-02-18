"""
M5 Verification: Exactness-Preserving d₁(k) on Periodic Voronoi Complexes
==========================================================================

Demonstrates that standard Bloch-phased d₁ breaks exactness (d₁d₀ ≠ 0)
on unstructured periodic meshes at k ≠ 0, while the recurrence construction
preserves it (d₁d₀ = 0 at machine precision).

Tests on 3 periodic Voronoi complexes: C15, Kelvin (BCC), Weaire-Phelan (A15).

Key results:
  1. Exactness: ||d₁_exact·d₀|| vs ||d₁_standard·d₀||
  2. Holonomy: product of Bloch phases around each face = 1 (flat connection)
  3. Gauge kernel: n_zero = n_V (exact = image of d₀)
  4. Acoustic modes: 2-fold degenerate, ω² ∝ k²
  5. Seed independence: spectrum invariant under phase[0] → exp(iθ)

Run:
  OPENBLAS_NUM_THREADS=1 /usr/bin/python3 publishing/wip/4_test_M5_exactness.py

RAW OUTPUT (run Feb 2026):

TEST 1: Exactness ||d₁(k)·d₀(k)|| at k ≠ 0
  Structure      V     E     F k-direction    ||exact||   ||standard||
  C15          136   272   160 [100]          5.23e-16       8.32e+00
                              [110]          8.29e-16       1.18e+01
                              [111]          1.14e-15       1.43e+01
  Kelvin        96   192   112 [100]          6.27e-16       7.43e+00
                              [110]          9.50e-16       9.61e+00
                              [111]          9.57e-16       1.10e+01
  WP            46    92    54 [100]          5.00e-16       5.07e+00
                              [110]          6.54e-16       7.23e+00
                              [111]          8.82e-16       9.03e+00

TEST 2: Holonomy (flat connection: Π_∂f phases = 1)
  C15:    max|hol - 1| = 3.33e-16
  Kelvin: max|hol - 1| = 4.58e-16
  WP:     max|hol - 1| = 2.29e-16

TEST 3: Gauge kernel dim = n_V
  C15:    n_zero = 136 = n_V (standard: 127)
  Kelvin: n_zero =  96 = n_V (standard:  90)
  WP:     n_zero =  46 = n_V (standard:  43)

TEST 4: Acoustic modes (2-fold degeneracy + k² scaling)
  C15:    c² = 0.9991, max_split = 1.1e-11, k²_spread = 0.0024
  Kelvin: c² = 0.9989, max_split = 1.9e-12, k²_spread = 0.0028
  WP:     c² = 0.9976, max_split = 4.4e-13, k²_spread = 0.0059

TEST 5: Seed independence
  C15 at k = 5% BZ [111]: max|Δeig| = 1.73e-14 across 6 seeds

SUMMARY:
  Structure   V    E    F   ||d₁d₀||_exact  n_zero=V  c_gauge   δc/c
  C15        136  272  160      5.23e-16     136=136   0.9994  0.11%
  Kelvin      96  192  112      6.27e-16      96=96    0.9993  0.13%
  WP          46   92   54      5.00e-16      46=46    0.9984  0.27%

ALL 5 TESTS PASSED

Feb 2026
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh

# Path setup: script is in publishing/wip/, code is in ST_11/src/1_foam/
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

from physics.hodge import (
    build_c15_with_dual_info,
    build_kelvin_with_dual_info,
    build_wp_with_dual_info,
    build_hodge_stars_voronoi,
    wrap_delta,
)
from physics.bloch import (
    compute_edge_crossings,
    build_edge_lookup,
    build_d0_bloch as bloch_build_d0,
    build_d1_bloch_standard as bloch_build_d1,
)
from physics.gauge_bloch import (
    compute_edge_shifts,
    build_d0_bloch as gauge_build_d0,
    build_d1_bloch_exact as gauge_build_d1,
)


# =============================================================================
# HELPERS
# =============================================================================

def build_standard_d1(V, E, F, L, k):
    """Build standard Bloch d₁ using bloch.py pipeline."""
    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)
    return bloch_build_d1(V, E, F, L, k, edge_lookup, crossings)


def build_exact_d1(V, E, F, L_vec, k):
    """Build exactness-preserving d₁ using gauge_bloch.py pipeline."""
    shifts = compute_edge_shifts(V, E, L_vec)
    d0_k = gauge_build_d0(V, E, k, L_vec, shifts)
    d1_k = gauge_build_d1(V, E, F, k, L_vec, d0_k)
    return d0_k, d1_k


def build_K_M(d1_k, star1, star2):
    """Build K = d₁† *₂ d₁ and M = *₁ for generalized EVP."""
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    return K, M


def physical_eigenvalues(K, M, threshold_rel=1e-12):
    """Solve K a = ω² M a, return physical eigenvalues above zero-mode gap."""
    eigvals = np.sort(np.real(eigh(K, M, eigvals_only=True)))
    max_eig = np.max(np.abs(eigvals)) if len(eigvals) > 0 else 1.0
    threshold = max(max_eig * threshold_rel, 1e-14)
    n_zero = np.sum(np.abs(eigvals) < threshold)
    physical = eigvals[np.abs(eigvals) >= threshold]
    return physical, n_zero


def compute_holonomy(d0_k, E, F):
    """Compute holonomy product around each face using d₀(k) entries.

    For flat connection: product of phases around boundary = 1.
    holonomy[f] = Π_edges(phase_from_d0) where phases extracted from d₀.
    """
    edge_map = {}
    for idx, (i, j) in enumerate(E):
        edge_map[(i, j)] = (idx, +1)
        edge_map[(j, i)] = (idx, -1)

    max_dev = 0.0
    for face in F:
        n_verts = len(face)
        hol = 1.0 + 0j
        for v_pos in range(n_verts):
            vi = face[v_pos]
            vj = face[(v_pos + 1) % n_verts]
            e_idx, orient = edge_map[(vi, vj)]
            # d₀(k)[e, vj] / d₀(k)[e, vi] gives the connection along this edge
            # For oriented edge (vi→vj): phase = d₀[e, vj] if orient=+1
            d0_target = d0_k[e_idx, vj]
            d0_source = d0_k[e_idx, vi]
            if abs(d0_target) > 1e-14 and abs(d0_source) > 1e-14:
                ratio = d0_target / d0_source
                hol *= -ratio  # d₀ = [-1, +phase], so ratio = -phase
        max_dev = max(max_dev, abs(hol - 1.0))
    return max_dev


# =============================================================================
# STRUCTURES
# =============================================================================

STRUCTURES = [
    ("C15",    build_c15_with_dual_info,    1, 4.0),
    ("Kelvin", build_kelvin_with_dual_info, 2, 4.0),
    ("WP",     build_wp_with_dual_info,     1, 4.0),
]


# =============================================================================
# TEST 1: EXACTNESS COMPARISON (all structures, multiple k-points)
# =============================================================================

def test_exactness():
    """Compare ||d₁·d₀|| for exact vs standard construction."""
    print("=" * 70)
    print("TEST 1: Exactness ||d₁(k)·d₀(k)|| at k ≠ 0")
    print("=" * 70)
    print()
    print(f"  {'Structure':<10} {'V':>5} {'E':>5} {'F':>5} "
          f"{'k-direction':<10} {'||exact||':>12} {'||standard||':>14}")
    print("  " + "-" * 66)

    all_ok = True

    for name, builder, N, L_cell in STRUCTURES:
        data = builder(N=N, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        n_V, n_E, n_F = len(V), len(E), len(F)

        k_scale = 2 * np.pi / L
        k_tests = [
            ("[100]", k_scale * np.array([0.1, 0.0, 0.0])),
            ("[110]", k_scale * np.array([0.1, 0.1, 0.0])),
            ("[111]", k_scale * np.array([0.1, 0.1, 0.1])),
        ]

        for k_label, k in k_tests:
            # Exact construction
            d0_k, d1_exact = build_exact_d1(V, E, F, L_vec, k)
            err_exact = np.linalg.norm(d1_exact @ d0_k)

            # Standard construction
            d1_standard = build_standard_d1(V, E, F, L, k)
            crossings = compute_edge_crossings(V, E, L)
            d0_bloch = bloch_build_d0(V, E, L, k, crossings)
            err_standard = np.linalg.norm(d1_standard @ d0_bloch)

            status = "OK" if err_exact < 1e-10 else "FAIL"
            if err_exact >= 1e-10:
                all_ok = False

            if k_label == "[100]":
                print(f"  {name:<10} {n_V:>5} {n_E:>5} {n_F:>5} "
                      f"{k_label:<10} {err_exact:>12.2e} {err_standard:>14.2e}  {status}")
            else:
                print(f"  {'':>27} "
                      f"{k_label:<10} {err_exact:>12.2e} {err_standard:>14.2e}  {status}")

    print()
    assert all_ok, "Exactness check failed on at least one structure/k-point"
    print("  PASSED: d₁_exact·d₀ = 0 at machine precision on all structures")
    print()


# =============================================================================
# TEST 2: HOLONOMY (flat connection check)
# =============================================================================

def test_holonomy():
    """Verify holonomy = 1 on all faces (flat U(1) connection)."""
    print("=" * 70)
    print("TEST 2: Holonomy (flat connection: Π_∂f phases = 1)")
    print("=" * 70)
    print()
    print(f"  {'Structure':<10} {'n_faces':>8} {'max|hol - 1|':>14}")
    print("  " + "-" * 36)

    all_ok = True
    k_test = np.array([0.3, 0.2, 0.1])  # generic k

    for name, builder, N, L_cell in STRUCTURES:
        data = builder(N=N, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L_vec = data['L_vec']

        shifts = compute_edge_shifts(V, E, L_vec)
        d0_k = gauge_build_d0(V, E, k_test, L_vec, shifts)

        max_dev = compute_holonomy(d0_k, E, F)
        status = "OK" if max_dev < 1e-12 else "FAIL"
        if max_dev >= 1e-12:
            all_ok = False

        print(f"  {name:<10} {len(F):>8} {max_dev:>14.2e}  {status}")

    print()
    assert all_ok, "Holonomy check failed"
    print("  PASSED: Flat connection verified on all faces")
    print()


# =============================================================================
# TEST 3: GAUGE KERNEL (n_zero = n_V)
# =============================================================================

def test_gauge_kernel():
    """Verify n_zero = n_V (gauge kernel = image of d₀)."""
    print("=" * 70)
    print("TEST 3: Gauge kernel dim = n_V")
    print("=" * 70)
    print()
    print(f"  {'Structure':<10} {'n_V':>5} {'n_zero_exact':>13} {'n_zero_std':>11} {'match':>7}")
    print("  " + "-" * 50)

    all_ok = True

    for name, builder, N, L_cell in STRUCTURES:
        data = builder(N=N, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        star1, star2 = build_hodge_stars_voronoi(data)
        n_V = len(V)

        k_scale = 2 * np.pi / L
        k = k_scale * np.array([0.1, 0.0, 0.0])

        # Exact d₁
        d0_k, d1_exact = build_exact_d1(V, E, F, L_vec, k)
        K_ex, M = build_K_M(d1_exact, star1, star2)
        _, nz_ex = physical_eigenvalues(K_ex, M)

        # Standard d₁
        d1_std = build_standard_d1(V, E, F, L, k)
        K_st, _ = build_K_M(d1_std, star1, star2)
        _, nz_st = physical_eigenvalues(K_st, M)

        match = "YES" if nz_ex == n_V else "NO"
        if nz_ex != n_V:
            all_ok = False

        print(f"  {name:<10} {n_V:>5} {nz_ex:>13} {nz_st:>11} {match:>7}")

    print()
    assert all_ok, "Gauge kernel check failed: n_zero != n_V"
    print("  PASSED: ker(K) = im(d₀) on all structures")
    print()


# =============================================================================
# TEST 4: ACOUSTIC MODES (2-fold degeneracy + ω² ∝ k²)
# =============================================================================

def test_acoustic_modes():
    """Verify 2 degenerate acoustic modes with ω² = c²k²."""
    print("=" * 70)
    print("TEST 4: Acoustic modes (2-fold degeneracy + k² scaling)")
    print("=" * 70)
    print()

    all_ok = True

    for name, builder, N, L_cell in STRUCTURES:
        data = builder(N=N, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        star1, star2 = build_hodge_stars_voronoi(data)

        k_scale = 2 * np.pi / L
        direction = np.array([1.0, 0.0, 0.0])
        k_fracs = [0.02, 0.05, 0.10, 0.15]

        eig_pairs = []
        k_mags = []

        for frac in k_fracs:
            k = k_scale * frac * direction
            d0_k, d1_k = build_exact_d1(V, E, F, L_vec, k)
            K, M = build_K_M(d1_k, star1, star2)
            phys, _ = physical_eigenvalues(K, M)
            eig_pairs.append((phys[0], phys[1]))
            k_mags.append(np.linalg.norm(k))

        k_mags = np.array(k_mags)
        eig0 = np.array([p[0] for p in eig_pairs])
        eig1 = np.array([p[1] for p in eig_pairs])

        # Degeneracy: relative split
        splits = np.abs(eig1 - eig0) / np.maximum(eig0, 1e-14)
        max_split = np.max(splits)

        # k² scaling: eig/k² should be constant
        ratio = eig0 / k_mags**2
        spread = (np.max(ratio) - np.min(ratio)) / np.mean(ratio)

        # Effective speed
        c_sq = np.mean(ratio)

        ok_degen = max_split < 1e-3
        ok_scaling = spread < 0.10
        if not ok_degen or not ok_scaling:
            all_ok = False

        print(f"  {name}:")
        print(f"    k_frac    eig[0]        eig[1]        split       eig/k²")
        print(f"    " + "-" * 60)
        for i, frac in enumerate(k_fracs):
            print(f"    {frac:<8.2f}  {eig_pairs[i][0]:<13.6e} {eig_pairs[i][1]:<13.6e} "
                  f"{splits[i]:<11.2e} {ratio[i]:<10.6f}")
        print(f"    c² = {c_sq:.4f}  |  max_split = {max_split:.2e}  |  k²_spread = {spread:.4f}")
        degen_status = "OK" if ok_degen else "FAIL"
        scale_status = "OK" if ok_scaling else "FAIL"
        print(f"    Degeneracy: {degen_status}  |  k² scaling: {scale_status}")
        print()

    assert all_ok, "Acoustic mode check failed"
    print("  PASSED: 2-fold degenerate acoustic modes with ω² ∝ k² on all structures")
    print()


# =============================================================================
# TEST 5: SEED INDEPENDENCE
# =============================================================================

def test_seed_independence():
    """Verify spectrum is independent of initial phase seed."""
    print("=" * 70)
    print("TEST 5: Seed independence (phase[0] = exp(iθ))")
    print("=" * 70)
    print()

    # Use C15 (largest complex, most stringent test)
    data = build_c15_with_dual_info(N=1, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)

    k_scale = 2 * np.pi / data['L']
    k = k_scale * np.array([0.05, 0.05, 0.05])

    shifts = compute_edge_shifts(V, E, L_vec)
    d0_k = gauge_build_d0(V, E, k, L_vec, shifts)

    # Build d₁ with seed = 1 (reference)
    d1_ref = gauge_build_d1(V, E, F, k, L_vec, d0_k)
    K_ref, M = build_K_M(d1_ref, star1, star2)
    phys_ref, nz_ref = physical_eigenvalues(K_ref, M)

    # Build edge_map for custom seed construction
    edge_map = {}
    for idx, (i, j) in enumerate(E):
        edge_map[(i, j)] = (idx, +1)
        edge_map[(j, i)] = (idx, -1)

    thetas = [0.3, 0.7, np.pi/2, np.pi, 1.5*np.pi, 2*np.pi - 0.01]
    max_diff = 0.0

    print(f"  C15 at k = 5% BZ [111], reference: n_zero={nz_ref}")
    print(f"  {'theta':>8} {'max|Δeig|':>12} {'rel_diff':>10}")
    print("  " + "-" * 34)

    for theta in thetas:
        seed = np.exp(1j * theta)
        # Rebuild d₁ with different seed
        n_E = len(E)
        n_F = len(F)
        d1_test = np.zeros((n_F, n_E), dtype=complex)
        for f_idx, face in enumerate(F):
            n_verts = len(face)
            edges_info = []
            for v_pos in range(n_verts):
                vi = face[v_pos]
                vj = face[(v_pos + 1) % n_verts]
                e_idx, orient = edge_map[(vi, vj)]
                edges_info.append((e_idx, orient, vi, vj))
            phases = [seed]
            for i in range(1, n_verts):
                e_prev, o_prev, _, _ = edges_info[i-1]
                e_curr, o_curr, _, _ = edges_info[i]
                v = face[i]
                phase_prev = phases[i-1]
                phase_curr = -o_prev * phase_prev * d0_k[e_prev, v] / (o_curr * d0_k[e_curr, v])
                phases.append(phase_curr)
            for i, (e_idx, orient, _, _) in enumerate(edges_info):
                d1_test[f_idx, e_idx] = orient * phases[i]

        K_test, _ = build_K_M(d1_test, star1, star2)
        phys_test, _ = physical_eigenvalues(K_test, M)

        n_cmp = min(len(phys_ref), len(phys_test), 10)
        diff = np.max(np.abs(phys_ref[:n_cmp] - phys_test[:n_cmp]))
        rel = diff / np.max(np.abs(phys_ref[:n_cmp]))
        max_diff = max(max_diff, diff)

        print(f"  {theta:>8.4f} {diff:>12.2e} {rel:>10.2e}")

    print(f"\n  Max eigenvalue difference across seeds: {max_diff:.2e}")
    assert max_diff < 1e-10, f"Spectrum depends on seed! max diff = {max_diff:.2e}"
    print("  PASSED: Spectrum invariant under seed rotation")
    print()


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def summary_table():
    """Print compact summary of all structures."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Structure':<10} {'V':>5} {'E':>5} {'F':>5} "
          f"{'||d₁d₀||_exact':>15} {'n_zero=V':>9} {'c_gauge':>8} {'δc/c':>8}")
    print("  " + "-" * 70)

    k_test_frac = 0.10

    for name, builder, N, L_cell in STRUCTURES:
        data = builder(N=N, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        star1, star2 = build_hodge_stars_voronoi(data)
        n_V, n_E, n_F = len(V), len(E), len(F)

        k_scale = 2 * np.pi / L
        k = k_scale * np.array([k_test_frac, 0.0, 0.0])

        d0_k, d1_k = build_exact_d1(V, E, F, L_vec, k)
        err = np.linalg.norm(d1_k @ d0_k)

        K, M = build_K_M(d1_k, star1, star2)
        phys, nz = physical_eigenvalues(K, M)
        nz_match = f"{nz}={n_V}" if nz == n_V else f"{nz}!={n_V}"

        # Speed from multiple k-points
        direction = np.array([1.0, 0.0, 0.0])
        k_fracs = [0.05, 0.10, 0.15]
        c_vals = []
        for frac in k_fracs:
            ki = k_scale * frac * direction
            d0_i, d1_i = build_exact_d1(V, E, F, L_vec, ki)
            Ki, Mi = build_K_M(d1_i, star1, star2)
            pi, _ = physical_eigenvalues(Ki, Mi)
            c_vals.append(np.sqrt(pi[0]) / np.linalg.norm(ki))
        c_mean = np.mean(c_vals)
        delta_c = (np.max(c_vals) - np.min(c_vals)) / c_mean

        print(f"  {name:<10} {n_V:>5} {n_E:>5} {n_F:>5} "
              f"{err:>15.2e} {nz_match:>9} {c_mean:>8.4f} {delta_c:>8.4%}")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 70)
    print("M5: Exactness-Preserving d₁(k) on Periodic Voronoi Complexes")
    print("=" * 70)
    print()

    test_exactness()
    test_holonomy()
    test_gauge_kernel()
    test_acoustic_modes()
    test_seed_independence()
    summary_table()

    print("ALL TESTS PASSED")
    print()


if __name__ == '__main__':
    main()
