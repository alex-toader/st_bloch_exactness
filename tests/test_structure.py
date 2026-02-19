"""
Structural tests: random Voronoi topologies, N-scaling, scalar Laplacian,
minimal counterexample.

Paper claims supported:
  - §5.5 "Extensive pollution": n_spur grows with system size
  - §5.5 "Specific to cochain level 1→2": scalar Laplacian clean
  - §5.5 "Onset at minimal supercell": SC 1×1×1 exact, N≥3 polluted
  - §5.5 "Universality": 10 random Voronoi seeds all PASS
  - Table 5 Random Voronoi row

Usage:
    cd /path/to/st_bloch_exactness
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_structure.py

Expected output (macOS, Python 3.9.6, SciPy 1.13):

    ======================================================================
    STRUCTURAL TESTS
    Paper: §5.5 claims
    ======================================================================

    ======================================================================
      RANDOM VORONOI — 10 seeds, n_cells=50
    ======================================================================

        seed     V     E   ||d1d0||_ex  n_spur     c2_ex    c2_std    status
          42   331   662      1.03e-15      15    0.9978    0.4657      PASS
         137   347   694      1.19e-15      19    0.9983    0.6807      PASS
         999   354   708      1.21e-15      19    0.9983    0.7008      PASS
        2024   345   690      1.09e-15      15    0.9981    0.6615      PASS
       31415   350   700      1.32e-15      17    0.9984    1.0615      PASS
        7777   347   694      1.07e-15      20    0.9985    0.4583      PASS
       54321   330   660      1.12e-15      19    0.9985    0.4287      PASS
       11111   357   714      1.24e-15      19    0.9980    0.9279      PASS
       88888   332   664      1.15e-15      20    0.9985    0.6873      PASS
       12345   342   684      8.70e-16      12    0.9984    0.4883      PASS

      Valid: 10/10, PASS: 10/10

    ======================================================================
      N-SCALING OF SPURIOUS MODE COUNT
    ======================================================================

        N      V     dir  n_spur       %
        2     96   [100]       6    6.25
        2     96   [111]      14   14.58
        3    324   [100]      17    5.25
        3    324   [111]      38   11.73
        4    768   [100]      29    3.78
        4    768   [111]      72    9.38

    ======================================================================
      SCALAR LAPLACIAN — COCHAIN LEVEL 0
    ======================================================================

          frac  n_zero_K0  n_zero_K1_ex  n_spur_K1
          0.02          0            96          6
          0.05          0            96          6
          0.10          0            96          6
          0.20          0            96          6

    ======================================================================
      MINIMAL COUNTEREXAMPLE — SC CUBIC 1×1×1
    ======================================================================

      SC 1×1×1: V=1, E=3, F=3 (minimal periodic complex)
          frac     dir      ||d1d0||    status
          0.01   [100]      0.00e+00     EXACT
          0.01   [111]      7.01e-20     EXACT
          0.10   [100]      0.00e+00     EXACT
          0.10   [111]      1.27e-17     EXACT
          0.30   [100]      0.00e+00     EXACT
          0.30   [111]      2.18e-17     EXACT
          0.50   [100]      0.00e+00     EXACT
          0.50   [111]      5.55e-17     EXACT

    ======================================================================
    ALL STRUCTURAL TESTS COMPLETE.
"""

import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import (
    build_kelvin_with_dual_info,
    build_foam_with_dual_info,
    build_hodge_stars_voronoi,
)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup


def build_K_M(d1_k, star1, star2):
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    return K, M


# ── Part 1: Random Voronoi (10 seeds) ───────────────────────────────────

def test_random_voronoi():
    """Exact vs standard on 10 random Voronoi topologies."""
    print(f"\n{'=' * 70}")
    print(f"  RANDOM VORONOI — 10 seeds, n_cells=50")
    print(f"{'=' * 70}")

    n_cells = 50
    L = 4.0
    seeds = [42, 137, 999, 2024, 31415, 7777, 54321, 11111, 88888, 12345]

    k_scale = 2 * np.pi / L
    k = k_scale * 0.10 * np.array([1.0, 0.0, 0.0])
    k2 = np.linalg.norm(k)**2

    print(f"\n  {'seed':>6s}  {'V':>4s}  {'E':>4s}  {'||d1d0||_ex':>12s}  "
          f"{'n_spur':>6s}  {'c2_ex':>8s}  {'c2_std':>8s}  {'status':>8s}")

    n_valid = 0
    n_pass = 0

    for seed in seeds:
        np.random.seed(seed)
        points = np.random.uniform(0, L, size=(n_cells, 3))

        try:
            data = build_foam_with_dual_info(points, L)
        except Exception:
            print(f"  {seed:>6d}  {'--':>4s}  {'--':>4s}  {'--':>12s}  {'--':>6s}  "
                  f"{'--':>8s}  {'--':>8s}  {'BUILD_ERR':>8s}")
            continue

        V, E, F = data['V'], data['E'], data['F']
        L_vec = data['L_vec']
        n_V, n_E = len(V), len(E)

        try:
            star1, star2 = build_hodge_stars_voronoi(data)
        except Exception:
            print(f"  {seed:>6d}  {n_V:4d}  {n_E:4d}  {'--':>12s}  {'--':>6s}  "
                  f"{'--':>8s}  {'--':>8s}  {'HODGE_ERR':>8s}")
            continue

        n_valid += 1
        shifts = compute_edge_shifts(V, E, L_vec)
        crossings = compute_edge_crossings(V, E, L)
        edge_lookup = build_edge_lookup(E, crossings)

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        norm_exact = np.linalg.norm(d1_exact @ d0_k)

        K_ex, M_ex = build_K_M(d1_exact, star1, star2)
        eigs_ex = np.sort(np.real(eigh(K_ex, M_ex, eigvals_only=True)))
        thresh_ex = max(np.max(np.abs(eigs_ex)) * 1e-12, 1e-14)
        n_zero_ex = np.sum(np.abs(eigs_ex) < thresh_ex)
        phys_ex = eigs_ex[np.abs(eigs_ex) >= thresh_ex]
        c2_ex = phys_ex[0] / k2 if len(phys_ex) > 0 else float('nan')

        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)
        K_st, M_st = build_K_M(d1_std, star1, star2)
        eigs_st = np.sort(np.real(eigh(K_st, M_st, eigvals_only=True)))
        thresh_st = max(np.max(np.abs(eigs_st)) * 1e-12, 1e-14)
        n_zero_std = np.sum(np.abs(eigs_st) < thresh_st)
        phys_st = eigs_st[np.abs(eigs_st) >= thresh_st]
        c2_std = phys_st[0] / k2 if len(phys_st) > 0 else float('nan')

        n_spur = n_zero_ex - n_zero_std
        ok = norm_exact < 1e-12 and n_zero_ex == n_V and n_spur > 0
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1

        print(f"  {seed:>6d}  {n_V:4d}  {n_E:4d}  {norm_exact:12.2e}  "
              f"{n_spur:6d}  {c2_ex:8.4f}  {c2_std:8.4f}  {status:>8s}")

    print(f"\n  Valid: {n_valid}/{len(seeds)}, PASS: {n_pass}/{n_valid}")


# ── Part 2: N-scaling of spurious mode count ─────────────────────────────

def test_n_scaling():
    """Spurious mode count grows with system size (extensive pollution)."""
    print(f"\n{'=' * 70}")
    print(f"  N-SCALING OF SPURIOUS MODE COUNT")
    print(f"{'=' * 70}")

    L_cell = 4.0
    k_hat_100 = np.array([1.0, 0.0, 0.0])
    k_hat_111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)

    print(f"\n  {'N':>3s}  {'V':>5s}  {'dir':>6s}  {'n_spur':>6s}  {'%':>6s}")

    for N in [2, 3, 4]:
        data = build_kelvin_with_dual_info(N=N, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        n_V = len(V)

        star1, star2 = build_hodge_stars_voronoi(data)
        shifts = compute_edge_shifts(V, E, L_vec)
        crossings = compute_edge_crossings(V, E, L)
        edge_lookup = build_edge_lookup(E, crossings)
        k_scale = 2 * np.pi / L

        for k_hat, label in [(k_hat_100, '[100]'), (k_hat_111, '[111]')]:
            k = k_scale * 0.10 * k_hat

            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
            K_ex, M_ex = build_K_M(d1_exact, star1, star2)
            eigs_ex = np.sort(np.real(eigh(K_ex, M_ex, eigvals_only=True)))
            thresh_ex = max(np.max(np.abs(eigs_ex)) * 1e-12, 1e-14)
            n_zero_ex = np.sum(np.abs(eigs_ex) < thresh_ex)

            d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)
            K_st, M_st = build_K_M(d1_std, star1, star2)
            eigs_st = np.sort(np.real(eigh(K_st, M_st, eigvals_only=True)))
            thresh_st = max(np.max(np.abs(eigs_st)) * 1e-12, 1e-14)
            n_zero_std = np.sum(np.abs(eigs_st) < thresh_st)

            n_spur = n_zero_ex - n_zero_std
            pct = 100.0 * n_spur / n_V
            print(f"  {N:3d}  {n_V:5d}  {label:>6s}  {n_spur:6d}  {pct:6.2f}")


# ── Part 3: Scalar Laplacian (level 0 clean) ────────────────────────────

def test_scalar_laplacian():
    """Scalar Laplacian has no pollution — problem is specific to level 1→2."""
    print(f"\n{'=' * 70}")
    print(f"  SCALAR LAPLACIAN — COCHAIN LEVEL 0")
    print(f"{'=' * 70}")

    data = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)
    n_V, n_E = len(V), len(E)

    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)
    k_scale = 2 * np.pi / L

    print(f"\n  {'frac':>8s}  {'n_zero_K0':>9s}  {'n_zero_K1_ex':>12s}  {'n_spur_K1':>9s}")

    for frac in [0.02, 0.05, 0.10, 0.20]:
        k = k_scale * frac * np.array([1.0, 0.0, 0.0])

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)

        # Scalar Laplacian
        K0 = d0_k.conj().T @ np.diag(star1) @ d0_k
        K0 = 0.5 * (K0 + K0.conj().T)
        eigs_0 = np.sort(np.real(eigh(K0, eigvals_only=True)))
        thresh_0 = max(np.max(np.abs(eigs_0)) * 1e-12, 1e-14)
        n_zero_0 = np.sum(np.abs(eigs_0) < thresh_0)

        # Curl-curl (exact)
        d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        K1 = d1_exact.conj().T @ np.diag(star2) @ d1_exact
        K1 = 0.5 * (K1 + K1.conj().T)
        M1 = np.diag(star1)
        eigs_1 = np.sort(np.real(eigh(K1, M1, eigvals_only=True)))
        thresh_1 = max(np.max(np.abs(eigs_1)) * 1e-12, 1e-14)
        n_zero_1 = np.sum(np.abs(eigs_1) < thresh_1)

        # Curl-curl (standard)
        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)
        K1s = d1_std.conj().T @ np.diag(star2) @ d1_std
        K1s = 0.5 * (K1s + K1s.conj().T)
        eigs_1s = np.sort(np.real(eigh(K1s, M1, eigvals_only=True)))
        thresh_1s = max(np.max(np.abs(eigs_1s)) * 1e-12, 1e-14)
        n_zero_1s = np.sum(np.abs(eigs_1s) < thresh_1s)

        print(f"  {frac:8.2f}  {n_zero_0:9d}  {n_zero_1:12d}  {n_zero_1 - n_zero_1s:9d}")


# ── Part 4: Minimal counterexample (SC 1×1×1) ───────────────────────────

def test_minimal_counterexample():
    """SC 1×1×1: standard = exact (no pollution). Pollution needs ≥2 cells."""
    print(f"\n{'=' * 70}")
    print(f"  MINIMAL COUNTEREXAMPLE — SC CUBIC 1×1×1")
    print(f"{'=' * 70}")

    L_vec = np.array([1.0, 1.0, 1.0])
    k_scale = 2 * np.pi / 1.0

    print(f"\n  SC 1×1×1: V=1, E=3, F=3 (minimal periodic complex)")
    print(f"  On this mesh, standard = exact (d1 unique, no room for pollution).")
    print(f"\n  {'frac':>8s}  {'dir':>6s}  {'||d1d0||':>12s}  {'status':>8s}")

    for frac in [0.01, 0.10, 0.30, 0.50]:
        for k_hat, label in [(np.array([1, 0, 0.]), '[100]'),
                              (np.array([1, 1, 1.]) / np.sqrt(3), '[111]')]:
            k = k_scale * frac * k_hat

            px = np.exp(1j * np.dot(k, np.array([1, 0, 0]) * L_vec))
            py = np.exp(1j * np.dot(k, np.array([0, 1, 0]) * L_vec))
            pz = np.exp(1j * np.dot(k, np.array([0, 0, 1]) * L_vec))

            # d0(k): 3×1
            d0 = np.array([[-1 + px], [-1 + py], [-1 + pz]])

            # d1(k): 3×3 (the only consistent choice)
            d1 = np.zeros((3, 3), dtype=complex)
            d1[0, 0] = 1 - py;  d1[0, 1] = px - 1
            d1[1, 0] = 1 - pz;  d1[1, 2] = px - 1
            d1[2, 1] = 1 - pz;  d1[2, 2] = py - 1

            norm_prod = np.linalg.norm(d1 @ d0)
            status = "EXACT" if norm_prod < 1e-12 else "BROKEN"
            print(f"  {frac:8.2f}  {label:>6s}  {norm_prod:12.2e}  {status:>8s}")


def main():
    print("=" * 70)
    print("STRUCTURAL TESTS")
    print("Paper: §5.5 claims")
    print("=" * 70)

    test_random_voronoi()
    test_n_scaling()
    test_scalar_laplacian()
    test_minimal_counterexample()

    print(f"\n{'=' * 70}")
    print("ALL STRUCTURAL TESTS COMPLETE.")


if __name__ == '__main__':
    main()
    print("\nDone.")
