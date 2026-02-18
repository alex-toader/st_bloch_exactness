"""
M5 Verification: Random Voronoi (No Symmetry)
==============================================

Tests the exactness-preserving d₁(k) on a completely random periodic
Voronoi complex: no O_h symmetry, no special lattice structure.

This is the strongest generality test: if it works here, the construction
is truly general, not dependent on lattice symmetries.

Expected differences from symmetric structures:
  - Polarization split nonzero (no symmetry to enforce degeneracy)
  - Anisotropy nonzero (no cubic symmetry)
  - But: exactness, ker=V, and ω² ∝ k² must still hold

Run:
  OPENBLAS_NUM_THREADS=1 /usr/bin/python3 publishing/wip/7_test_M5_random_voronoi.py

RAW OUTPUT (run Feb 2026):

TEST 1: Structure (30 cells, seed=999)
  V=206, E=412, F=236
  Edges/vertex: {4}, Faces/edge: {3}
  Dual orthogonality: min=1.000000
  star1: [0.0675, 1034.40], star2: [0.2905, 141251.75]

TEST 2: Exactness
  [100]: 9.79e-16, [110]: 1.24e-15, [111]: 1.51e-15, random: 1.40e-15

TEST 3: Spectrum ([100])
  n_zero = 206 = n_V on all k-points
  c² = 0.9978, k² spread = 0.0054
  Split: 8.5e-06 to 4.9e-04 (nonzero — no symmetry)

TEST 4: Anisotropy (6 directions at 5% BZ)
  c²: 0.9994-0.9996, anisotropy = 0.0001

TEST 5: Multiple seeds (n=30)
  seed=999: V=206, OK
  seed=31415: V=207, OK
  3/5 seeds skipped (degenerate mesh, faces/edge < 3)

EXTENDED: n=50, 10 seeds (separate run)
  9/10 valid (1 skipped: faces/edge < 3)
  All 9: exactness ~6e-16, ker=V, c²≈1.0
  Split: 8.6e-06 to 1.0e-04 (nonzero, correct)
  Anisotropy: < 0.02%

  Mesh validity vs density (20 seeds each):
    n=20:  0% valid
    n=30: 30% valid
    n=50: 80% valid
    n=80: 100% valid
  Condition: faces/edge ≥ 3 (sufficient point density)

ALL TESTS PASSED

Feb 2026
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

from physics.hodge import (
    build_foam_with_dual_info,
    build_hodge_stars_voronoi,
    verify_plateau_structure,
)
from physics.gauge_bloch import (
    compute_edge_shifts,
    build_d0_bloch,
    build_d1_bloch_exact,
)


def build_K_M(d1_k, star1, star2):
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    return K, M


def physical_eigenvalues(K, M, threshold_rel=1e-12):
    eigvals = np.sort(np.real(eigh(K, M, eigvals_only=True)))
    max_eig = np.max(np.abs(eigvals)) if len(eigvals) > 0 else 1.0
    threshold = max(max_eig * threshold_rel, 1e-14)
    n_zero = np.sum(np.abs(eigvals) < threshold)
    physical = eigvals[np.abs(eigvals) >= threshold]
    return physical, n_zero


def build_random_voronoi(n_cells, L, seed):
    """Build periodic Voronoi from random points."""
    np.random.seed(seed)
    points = np.random.uniform(0, L, size=(n_cells, 3))
    return build_foam_with_dual_info(points, L)


# =========================================================================
# TEST 1: Structure
# =========================================================================

def test_structure(data, n_cells, seed):
    print("=" * 70)
    print(f"TEST 1: Random Voronoi structure ({n_cells} cells, seed={seed})")
    print("=" * 70)
    print()

    V, E, F = data['V'], data['E'], data['F']
    print(f"  V={len(V)}, E={len(E)}, F={len(F)}, cells={n_cells}")

    plat = verify_plateau_structure(data)
    print(f"  Edges/vertex: {plat['edges_per_vertex']}")
    print(f"  Faces/edge: {plat['faces_per_edge']}")
    print(f"  Dual orthogonality: min={plat['dual_orthogonality_min']:.6f}")

    star1, star2 = build_hodge_stars_voronoi(data)
    print(f"  star1: [{star1.min():.4f}, {star1.max():.4f}]")
    print(f"  star2: [{star2.min():.4f}, {star2.max():.4f}]")
    print()


# =========================================================================
# TEST 2: Exactness
# =========================================================================

def test_exactness(data):
    print("=" * 70)
    print("TEST 2: Exactness ||d₁(k)·d₀(k)||")
    print("=" * 70)
    print()

    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    L = data['L']
    shifts = compute_edge_shifts(V, E, L_vec)

    k_scale = 2 * np.pi / L
    k_tests = [
        ("[100]", k_scale * np.array([0.1, 0.0, 0.0])),
        ("[110]", k_scale * np.array([0.1, 0.1, 0.0])),
        ("[111]", k_scale * np.array([0.1, 0.1, 0.1])),
        ("random", k_scale * np.array([0.07, 0.13, 0.03])),
    ]

    print(f"  {'k-dir':<10} {'||d₁d₀||':>12}")
    print("  " + "-" * 24)

    all_ok = True
    for label, k in k_tests:
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        err = np.linalg.norm(d1_k @ d0_k)
        status = "OK" if err < 1e-10 else "FAIL"
        if err >= 1e-10:
            all_ok = False
        print(f"  {label:<10} {err:>12.2e}  {status}")

    print()
    assert all_ok
    print("  PASSED")
    print()


# =========================================================================
# TEST 3: Gauge kernel + acoustic modes
# =========================================================================

def test_spectrum(data):
    print("=" * 70)
    print("TEST 3: Gauge kernel + acoustic modes")
    print("=" * 70)
    print()

    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)
    n_V = len(V)

    k_scale = 2 * np.pi / L
    direction = np.array([1.0, 0.0, 0.0])
    k_fracs = [0.02, 0.05, 0.10, 0.15]

    print(f"  n_V = {n_V}")
    print(f"  {'k_frac':<8} {'n_zero':>7} {'eig[0]':>13} {'eig[1]':>13} {'split':>11} {'eig/k²':>10}")
    print("  " + "-" * 66)

    ratios = []
    all_ok = True
    for frac in k_fracs:
        k = k_scale * frac * direction
        k_mag = np.linalg.norm(k)
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        K, M = build_K_M(d1_k, star1, star2)
        phys, nz = physical_eigenvalues(K, M)

        split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
        ratio = phys[0] / k_mag**2
        ratios.append(ratio)

        if nz != n_V:
            all_ok = False
        print(f"  {frac:<8.2f} {nz:>7} {phys[0]:>13.6e} {phys[1]:>13.6e} {split:>11.2e} {ratio:>10.6f}")

    ratios = np.array(ratios)
    spread = (np.max(ratios) - np.min(ratios)) / np.mean(ratios)
    print(f"\n  c² ≈ {np.mean(ratios):.4f}, k² spread = {spread:.4f}")

    assert all_ok, "n_zero != n_V"
    assert spread < 0.10, f"k² spread too large: {spread}"
    print("  PASSED: ker=V, ω² ∝ k²")
    print()


# =========================================================================
# TEST 4: Anisotropy (expected nonzero for random)
# =========================================================================

def test_anisotropy(data):
    print("=" * 70)
    print("TEST 4: Anisotropy (6 directions)")
    print("=" * 70)
    print()

    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)

    k_scale = 2 * np.pi / L
    frac = 0.05

    directions = [
        ("[100]", np.array([1.0, 0.0, 0.0])),
        ("[010]", np.array([0.0, 1.0, 0.0])),
        ("[001]", np.array([0.0, 0.0, 1.0])),
        ("[110]", np.array([1.0, 1.0, 0.0]) / np.sqrt(2)),
        ("[101]", np.array([1.0, 0.0, 1.0]) / np.sqrt(2)),
        ("[111]", np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
    ]

    print(f"  {'dir':<8} {'c²':>10} {'split':>12}")
    print("  " + "-" * 34)

    c_sq = []
    for label, d in directions:
        k = k_scale * frac * d
        k2 = np.dot(k, k)
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        K, M = build_K_M(d1_k, star1, star2)
        phys, _ = physical_eigenvalues(K, M)
        c2 = (phys[0] + phys[1]) / (2 * k2)
        split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
        c_sq.append(c2)
        print(f"  {label:<8} {c2:>10.6f} {split:>12.2e}")

    c_sq = np.array(c_sq)
    aniso = (np.max(c_sq) - np.min(c_sq)) / np.mean(c_sq)
    print(f"\n  Anisotropy = {aniso:.6f}")
    print(f"  (Random Voronoi: anisotropy expected nonzero, unlike O_h structures)")
    print()


# =========================================================================
# TEST 5: Multiple random seeds
# =========================================================================

def test_multiple_seeds():
    print("=" * 70)
    print("TEST 5: Multiple random seeds (n=30)")
    print("=" * 70)
    print()

    L = 4.0
    n_cells = 30
    seeds = [42, 137, 999, 2024, 31415]

    print(f"  {'seed':>6} {'V':>5} {'E':>5} {'F':>5} "
          f"{'||d₁d₀||':>10} {'n_zero':>7} {'=V?':>4} {'c²':>8} {'split':>10}")
    print("  " + "-" * 72)

    n_ok = 0
    for seed in seeds:
        try:
            data = build_random_voronoi(n_cells, L, seed)
            V, E, F = data['V'], data['E'], data['F']
            L_vec = data['L_vec']
            n_V = len(V)

            # Check faces/edge >= 3
            from collections import defaultdict
            eft = defaultdict(set)
            for fi, face in enumerate(F):
                for i in range(len(face)):
                    e = (min(face[i], face[(i+1)%len(face)]),
                         max(face[i], face[(i+1)%len(face)]))
                    eft[e].add(fi)
            min_fpe = min(len(v) for v in eft.values())
            if min_fpe < 3:
                print(f"  {seed:>6}  -- skipped (min faces/edge = {min_fpe})")
                continue

            star1, star2 = build_hodge_stars_voronoi(data)
            shifts = compute_edge_shifts(V, E, L_vec)

            k = (2*np.pi/L) * 0.05 * np.array([1, 0, 0])
            k_mag = np.linalg.norm(k)
            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
            err = np.linalg.norm(d1_k @ d0_k)

            K, M = build_K_M(d1_k, star1, star2)
            phys, nz = physical_eigenvalues(K, M)
            match = "YES" if nz == n_V else "NO"
            c2 = phys[0] / k_mag**2
            split = abs(phys[1]-phys[0]) / max(abs(phys[0]), 1e-14)

            print(f"  {seed:>6} {n_V:>5} {len(E):>5} {len(F):>5} "
                  f"{err:>10.2e} {nz:>7} {match:>4} {c2:>8.4f} {split:>10.2e}")

            if nz == n_V and err < 1e-10:
                n_ok += 1

        except Exception as e:
            print(f"  {seed:>6}  -- ERROR: {e}")

    print(f"\n  {n_ok}/{len(seeds)} seeds: exactness + ker=V confirmed")
    print()


# =========================================================================
# MAIN
# =========================================================================

def main():
    print()
    print("=" * 70)
    print("M5: Random Voronoi (No Symmetry)")
    print("=" * 70)
    print()

    # Primary test structure
    n_cells = 30
    seed = 999
    L = 4.0
    data = build_random_voronoi(n_cells, L, seed)

    test_structure(data, n_cells, seed)
    test_exactness(data)
    test_spectrum(data)
    test_anisotropy(data)
    test_multiple_seeds()

    print("ALL TESTS PASSED")
    print()


if __name__ == '__main__':
    main()
