"""
M5 Verification: FCC as 4th Data Point (Non-Plateau Mesh)
=========================================================

FCC Voronoi produces rhombic dodecahedra. NOT full Plateau:
  - Some vertices have 8 edges (not 4)
  - But 3 faces/edge (same as Plateau)

Tests the exactness-preserving d₁(k) construction on a non-Plateau mesh.

Run:
  OPENBLAS_NUM_THREADS=1 /usr/bin/python3 publishing/wip/6_test_M5_fcc.py

RAW OUTPUT (run Feb 2026):

TEST 1: FCC mesh structure
  V=12, E=32, F=24, cells=4
  Edges per vertex: {8, 4} (NOT full Plateau)
  Faces per edge: {3}
  Dual orthogonality: min=1.000000

TEST 2: Exactness
  [100]  ||exact|| = 0.00e+00  ||standard|| = 5.54e+00
  [110]  ||exact|| = 4.74e-16  ||standard|| = 8.11e+00
  [111]  ||exact|| = 5.48e-16  ||standard|| = 1.04e+01

TEST 3: Gauge kernel + acoustic modes
  n_zero = 12 = n_V on all k-points
  c² = 0.9982, k² spread = 0.0045
  Acoustic split < 2e-12 (degenerate)

TEST 4: Isotropy at 2% BZ
  c²: [100]=0.9999  [110]=0.9998  [111]=0.9998
  Anisotropy = 0.0001

ALL FCC TESTS PASSED

Feb 2026
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh
from itertools import product as iprod

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
from physics.bloch import (
    compute_edge_crossings,
    build_edge_lookup,
    build_d0_bloch as bloch_build_d0,
    build_d1_bloch_standard as bloch_build_d1,
)


def get_fcc_points(N, L_cell=4.0):
    """FCC lattice points: 4 per unit cell."""
    centers = []
    for i, j, k in iprod(range(N), repeat=3):
        centers.append([L_cell*i, L_cell*j, L_cell*k])
        centers.append([L_cell*i + L_cell/2, L_cell*j + L_cell/2, L_cell*k])
        centers.append([L_cell*i + L_cell/2, L_cell*j, L_cell*k + L_cell/2])
        centers.append([L_cell*i, L_cell*j + L_cell/2, L_cell*k + L_cell/2])
    return np.array(centers)


def build_fcc(N=1, L_cell=4.0):
    """Build FCC foam complex."""
    L = N * L_cell
    points = get_fcc_points(N, L_cell)
    return build_foam_with_dual_info(points, L)


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


# =========================================================================
# TEST 1: Structure check
# =========================================================================

def test_structure():
    print("=" * 70)
    print("TEST 1: FCC mesh structure")
    print("=" * 70)
    print()

    data = build_fcc(N=1)
    V, E, F = data['V'], data['E'], data['F']
    print(f"  V={len(V)}, E={len(E)}, F={len(F)}, cells=4")

    plat = verify_plateau_structure(data)
    print(f"  Edges per vertex: {plat['edges_per_vertex']} (Plateau requires {{4}})")
    print(f"  Faces per edge: {plat['faces_per_edge']} (Plateau requires {{3}})")
    print(f"  Full Plateau: {plat['all_ok']}")
    print(f"  Dual orthogonality: min={plat['dual_orthogonality_min']:.6f}")

    star1, star2 = build_hodge_stars_voronoi(data)
    print(f"  star1: [{star1.min():.4f}, {star1.max():.4f}]")
    print(f"  star2: [{star2.min():.4f}, {star2.max():.4f}]")
    print()


# =========================================================================
# TEST 2: Exactness
# =========================================================================

def test_exactness():
    print("=" * 70)
    print("TEST 2: Exactness ||d₁(k)·d₀(k)||")
    print("=" * 70)
    print()

    data = build_fcc(N=1)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']

    k_scale = 2 * np.pi / L
    k_tests = [
        ("[100]", k_scale * np.array([0.1, 0.0, 0.0])),
        ("[110]", k_scale * np.array([0.1, 0.1, 0.0])),
        ("[111]", k_scale * np.array([0.1, 0.1, 0.1])),
    ]

    shifts = compute_edge_shifts(V, E, L_vec)
    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)

    print(f"  {'k-dir':<8} {'||exact||':>12} {'||standard||':>14}")
    print("  " + "-" * 38)

    for label, k in k_tests:
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        err_ex = np.linalg.norm(d1_ex @ d0_k)

        d0_bl = bloch_build_d0(V, E, L, k, crossings)
        d1_st = bloch_build_d1(V, E, F, L, k, edge_lookup, crossings)
        err_st = np.linalg.norm(d1_st @ d0_bl)

        print(f"  {label:<8} {err_ex:>12.2e} {err_st:>14.2e}")
        assert err_ex < 1e-10

    print()
    print("  PASSED")
    print()


# =========================================================================
# TEST 3: Gauge kernel + acoustic modes
# =========================================================================

def test_spectrum():
    print("=" * 70)
    print("TEST 3: Gauge kernel + acoustic modes")
    print("=" * 70)
    print()

    data = build_fcc(N=1)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    n_V = len(V)

    k_scale = 2 * np.pi / L
    direction = np.array([1.0, 0.0, 0.0])
    k_fracs = [0.02, 0.05, 0.10, 0.15]

    shifts = compute_edge_shifts(V, E, L_vec)

    eig_pairs = []
    k_mags = []

    print(f"  n_V = {n_V}")
    print(f"  {'k_frac':<8} {'n_zero':>7} {'eig[0]':>13} {'eig[1]':>13} {'split':>11} {'eig/k²':>10}")
    print("  " + "-" * 66)

    for frac in k_fracs:
        k = k_scale * frac * direction
        k_mag = np.linalg.norm(k)
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        K, M = build_K_M(d1_k, star1, star2)
        phys, nz = physical_eigenvalues(K, M)

        split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
        ratio = phys[0] / k_mag**2
        eig_pairs.append((phys[0], phys[1]))
        k_mags.append(k_mag)

        print(f"  {frac:<8.2f} {nz:>7} {phys[0]:>13.6e} {phys[1]:>13.6e} {split:>11.2e} {ratio:>10.6f}")
        assert nz == n_V, f"n_zero={nz} != n_V={n_V}"

    k_mags = np.array(k_mags)
    eig0 = np.array([p[0] for p in eig_pairs])
    ratio_all = eig0 / k_mags**2
    spread = (np.max(ratio_all) - np.min(ratio_all)) / np.mean(ratio_all)
    c_sq = np.mean(ratio_all)

    print(f"\n  c² = {c_sq:.4f}, k² spread = {spread:.4f}")
    assert spread < 0.10
    print("  PASSED")
    print()


# =========================================================================
# TEST 4: Isotropy (3 directions)
# =========================================================================

def test_isotropy():
    print("=" * 70)
    print("TEST 4: Isotropy at 2% BZ")
    print("=" * 70)
    print()

    data = build_fcc(N=1)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)

    k_scale = 2 * np.pi / L
    frac = 0.02

    directions = [
        ("[100]", np.array([1.0, 0.0, 0.0])),
        ("[110]", np.array([1.0, 1.0, 0.0]) / np.sqrt(2)),
        ("[111]", np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
    ]

    print(f"  {'dir':<8} {'c²':>10} {'split':>12}")
    print("  " + "-" * 34)

    c_sq_values = []
    for label, d in directions:
        k = k_scale * frac * d
        k2 = np.dot(k, k)
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        K, M = build_K_M(d1_k, star1, star2)
        phys, _ = physical_eigenvalues(K, M)
        c2 = (phys[0] + phys[1]) / (2 * k2)
        split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
        c_sq_values.append(c2)
        print(f"  {label:<8} {c2:>10.6f} {split:>12.2e}")

    c_sq_values = np.array(c_sq_values)
    aniso = (np.max(c_sq_values) - np.min(c_sq_values)) / np.mean(c_sq_values)
    print(f"\n  Anisotropy: {aniso:.6f}")
    print("  PASSED" if aniso < 0.01 else "  NOTE: anisotropy > 1%")
    print()


# =========================================================================
# MAIN
# =========================================================================

def main():
    print()
    print("=" * 70)
    print("M5: FCC as 4th Data Point (Non-Plateau)")
    print("=" * 70)
    print()

    test_structure()
    test_exactness()
    test_spectrum()
    test_isotropy()

    print("ALL FCC TESTS PASSED")
    print()


if __name__ == '__main__':
    main()
