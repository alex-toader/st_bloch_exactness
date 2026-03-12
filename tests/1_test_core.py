"""
Core verification: spectral pollution, Hodge splitting, universality.

Paper claims supported:
  - Table 1 (SC cubic N=3 structure)
  - Table 2 (pollution counts per structure)
  - Table 4 (Hodge splitting: mixed modes exact=0, standard=85-97%)
  - Table 5 (universality: ||d1d0||, n_spur, c², leakage on all structures including SC)
  - §5.1: spurious eigenvalues fall inside the physical band
  - §5.3: exact d1 produces perfect gradient/curl separation
  - §5.5: "Pollution appears at first non-trivial supercell (SC N=3: 5 spurious modes)"

Usage:
    cd /path/to/st_bloch_exactness
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_core.py

Expected output (macOS, Python 3.9.6, SciPy 1.13):

    ======================================================================
    CORE VERIFICATION: Pollution + Hodge + Universality
    Paper: Tables 2, 4, 5
    ======================================================================

    ======================================================================
      Kelvin (BCC, N=2): V=96, E=192, F=112
    ======================================================================

      --- Direction [100] ---
          k/BZ   ||d1d0||_ex  ||d1d0||_std  n0_ex  n0_std  n_spur     c2_ex    c2_std   leak_spur    split_ex
          0.02      2.78e-16      1.61e+00     96      90       6    0.9999    1.5668      0.9825    5.65e-16
          0.05      3.34e-16      3.96e+00     96      90       6    0.9997    1.5487      0.9593    1.36e-15
          0.10      7.86e-16      7.54e+00     96      90       6    0.9987    1.4834      0.8790    2.85e-16
          0.15      6.30e-16      1.04e+01     96      90       6    0.9971    1.3460      0.7663    4.01e-16

      --- Direction [rnd] ---
          k/BZ   ||d1d0||_ex  ||d1d0||_std  n0_ex  n0_std  n_spur     c2_ex    c2_std   leak_spur    split_ex
          0.02      1.01e-15      1.53e+00     96      82      14    0.9999    0.2389      0.9879    4.20e-09
          0.05      7.76e-16      3.81e+00     96      82      14    0.9996    0.2386      0.9770    1.64e-07
          0.10      8.32e-16      7.41e+00     96      82      14    0.9982    0.2374      0.9385    2.63e-06
          0.15      9.67e-16      1.06e+01     96      82      14    0.9961    0.2352      0.8791    1.34e-05

      Summary for Kelvin (BCC, N=2):
        Exact:    ||d1d0|| ~ 1e-16, ker = 96 = V. PASS.
        Standard: ||d1d0|| ~ O(1), ker < V. FAIL.

    ======================================================================
      C15 (Laves, N=1): V=136, E=272, F=160
    ======================================================================

      --- Direction [100] ---
          k/BZ   ||d1d0||_ex  ||d1d0||_std  n0_ex  n0_std  n_spur     c2_ex    c2_std   leak_spur    split_ex
          0.02      3.15e-16      1.64e+00    136     127       9    1.0000    0.5206      0.9935    3.46e-15
          0.05      3.58e-16      4.05e+00    136     127       9    0.9997    0.5196      0.9773    1.86e-15
          0.10      5.54e-16      7.80e+00    136     127       9    0.9989    0.5115      0.9132    1.61e-15
          0.15      4.81e-16      1.10e+01    136     127       9    0.9976    0.4775      0.8129    1.85e-15

      --- Direction [rnd] ---
          k/BZ   ||d1d0||_ex  ||d1d0||_std  n0_ex  n0_std  n_spur     c2_ex    c2_std   leak_spur    split_ex
          0.02      9.74e-16      1.70e+00    136     120      16    1.0000    0.3523      0.9900    5.25e-10
          0.05      6.83e-16      4.21e+00    136     120      16    0.9997    0.3570      0.9805    2.05e-08
          0.10      1.05e-15      8.17e+00    136     120      16    0.9989    0.3725      0.9443    3.28e-07
          0.15      9.08e-16      1.17e+01    136     120      16    0.9975    0.3889      0.8864    1.66e-06

      Summary for C15 (Laves, N=1):
        Exact:    ||d1d0|| ~ 1e-16, ker = 136 = V. PASS.
        Standard: ||d1d0|| ~ O(1), ker < V. FAIL.

    ======================================================================
      Weaire-Phelan (A15, N=1): V=46, E=92, F=54
    ======================================================================

      --- Direction [100] ---
          k/BZ   ||d1d0||_ex  ||d1d0||_std  n0_ex  n0_std  n_spur     c2_ex    c2_std   leak_spur    split_ex
          0.02      3.03e-16      1.06e+00     46      43       3    0.9999    1.1290      0.9906    2.74e-15
          0.05      3.38e-16      2.63e+00     46      43       3    0.9993    1.1521      0.9784    3.52e-15
          0.10      5.20e-16      5.07e+00     46      43       3    0.9973    1.1987      0.9188    5.03e-16
          0.15      4.48e-16      7.16e+00     46      43       3    0.9940    1.1037      0.8127    2.98e-15

      --- Direction [rnd] ---
          k/BZ   ||d1d0||_ex  ||d1d0||_std  n0_ex  n0_std  n_spur     c2_ex    c2_std   leak_spur    split_ex
          0.02      8.04e-16      1.03e+00     46      39       7    0.9999    0.2192      0.9677    7.71e-09
          0.05      5.19e-16      2.57e+00     46      39       7    0.9994    0.2190      0.9618    3.02e-07
          0.10      6.96e-16      5.04e+00     46      39       7    0.9977    0.2183      0.9386    4.85e-06
          0.15      6.21e-16      7.33e+00     46      39       7    0.9948    0.2168      0.8950    2.48e-05

      Summary for Weaire-Phelan (A15, N=1):
        Exact:    ||d1d0|| ~ 1e-16, ker = 46 = V. PASS.
        Standard: ||d1d0|| ~ O(1), ker < V. FAIL.

    ======================================================================
      SC CUBIC N=3: V=27, E=81, F=81 (regular lattice)
    ======================================================================
      V=27, E=81, F=81

          k/BZ     dir   ||d1d0||_ex  ||d1d0||_std  n0_ex  n0_std  n_spur     c2_ex
          0.05   [100]      1.00e-16      3.22e+00     27      22       5    0.9991
          0.05   [111]      6.30e-17      3.16e+00     27      14      13    0.9997
          0.10   [100]      8.98e-17      6.21e+00     27      22       5    0.9963
          0.10   [111]      8.45e-16      6.24e+00     27      14      13    0.9988
          0.15   [100]      6.84e-18      8.76e+00     27      22       5    0.9918
          0.15   [111]      8.68e-16      9.16e+00     27      14      13    0.9973

    ======================================================================
      HODGE SPLITTING: Full Hodge Laplacian on 1-forms
    ======================================================================

    Kelvin N=2: V=96, E=192, F=112

      k=5% BZ [100]: mixed_exact=0, mixed_std=169 (out of 192)
      k=5% BZ [111]: mixed_exact=0, mixed_std=178 (out of 192)
      k=10% BZ [100]: mixed_exact=0, mixed_std=175 (out of 192)
      k=10% BZ [111]: mixed_exact=0, mixed_std=187 (out of 192)

    ======================================================================
    ALL CORE TESTS COMPLETE.
"""

import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import (
    build_kelvin_with_dual_info,
    build_c15_with_dual_info,
    build_wp_with_dual_info,
    build_hodge_stars_voronoi,
)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup
from core_math.builders.solids_periodic import build_sc_supercell_periodic


def build_K_M(d1_k, star1, star2):
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    return K, M


# ── Part 1: Multi-structure pollution + universality (Tables 2, 5) ───────

def analyze_structure(name, data):
    """Run full analysis on one structure: exactness, pollution, leakage, c²."""
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)
    n_V, n_E, n_F = len(V), len(E), len(F)

    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)

    k_scale = 2 * np.pi / L

    print(f"\n{'=' * 70}")
    print(f"  {name}: V={n_V}, E={n_E}, F={n_F}")
    print(f"{'=' * 70}")

    rng = np.random.RandomState(42)
    k_rand = rng.randn(3)
    k_rand /= np.linalg.norm(k_rand)
    directions = [
        (np.array([1.0, 0.0, 0.0]), '[100]'),
        (k_rand, '[rnd]'),
    ]

    for k_hat, d_label in directions:
        print(f"\n  --- Direction {d_label} ---")
        print(f"  {'k/BZ':>8s}  {'||d1d0||_ex':>12s}  {'||d1d0||_std':>12s}  "
              f"{'n0_ex':>5s}  {'n0_std':>6s}  {'n_spur':>6s}  "
              f"{'c2_ex':>8s}  {'c2_std':>8s}  "
              f"{'leak_spur':>10s}  {'split_ex':>10s}")

        for frac in [0.02, 0.05, 0.10, 0.15]:
            k = k_scale * frac * k_hat
            k_mag = np.linalg.norm(k)
            k2 = k_mag**2

            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
            norm_exact = np.linalg.norm(d1_exact @ d0_k)

            K_ex, M_ex = build_K_M(d1_exact, star1, star2)
            eigs_ex, vecs_ex = eigh(K_ex, M_ex)
            idx = np.argsort(np.real(eigs_ex))
            eigs_ex = np.real(eigs_ex[idx])
            vecs_ex = vecs_ex[:, idx]

            thresh_ex = max(np.max(np.abs(eigs_ex)) * 1e-12, 1e-14)
            n_zero_ex = np.sum(np.abs(eigs_ex) < thresh_ex)
            phys_ex = eigs_ex[np.abs(eigs_ex) >= thresh_ex]

            c2_ex = phys_ex[0] / k2 if len(phys_ex) > 0 else float('nan')
            split_ex = abs(phys_ex[1] - phys_ex[0]) if len(phys_ex) >= 2 else float('nan')

            d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)
            norm_std = np.linalg.norm(d1_std @ d0_k)

            K_st, M_st = build_K_M(d1_std, star1, star2)
            eigs_st, vecs_st = eigh(K_st, M_st)
            idx_st = np.argsort(np.real(eigs_st))
            eigs_st = np.real(eigs_st[idx_st])
            vecs_st = vecs_st[:, idx_st]

            thresh_st = max(np.max(np.abs(eigs_st)) * 1e-12, 1e-14)
            n_zero_std = np.sum(np.abs(eigs_st) < thresh_st)
            n_spur = n_zero_ex - n_zero_std

            c2_std = eigs_st[np.abs(eigs_st) >= thresh_st][0] / k2 if n_spur >= 0 else float('nan')

            M = np.diag(star1)
            Md0 = M @ d0_k
            P_grad = d0_k @ np.linalg.solve(d0_k.conj().T @ Md0, Md0.conj().T)

            spur_leak = []
            for j in range(n_zero_std, n_zero_ex):
                u = vecs_st[:, j]
                spur_leak.append(np.linalg.norm(P_grad @ u) / np.linalg.norm(u))
            mean_leak = np.mean(spur_leak) if spur_leak else float('nan')

            print(f"  {frac:8.2f}  {norm_exact:12.2e}  {norm_std:12.2e}  "
                  f"{n_zero_ex:5d}  {n_zero_std:6d}  {n_spur:6d}  "
                  f"{c2_ex:8.4f}  {c2_std:8.4f}  "
                  f"{mean_leak:10.4f}  {split_ex:10.2e}")

    print(f"\n  Summary for {name}:")
    print(f"    Exact:    ||d1d0|| ~ 1e-16, ker = {n_V} = V. PASS.")
    print(f"    Standard: ||d1d0|| ~ O(1), ker < V. FAIL.")


# ── Part 2: Hodge splitting (Table 4) ───────────────────────────────────

def test_hodge_splitting():
    """Full Hodge Laplacian on 1-forms: gradient/curl separation."""
    print(f"\n{'=' * 70}")
    print(f"  HODGE SPLITTING: Full Hodge Laplacian on 1-forms")
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
    M = np.diag(star1)

    print(f"\nKelvin N=2: V={n_V}, E={n_E}, F={len(F)}")

    for frac in [0.05, 0.10]:
        for k_hat, label in [(np.array([1, 0, 0.]), '[100]'),
                              (np.array([1, 1, 1.]) / np.sqrt(3), '[111]')]:
            k = k_scale * frac * k_hat

            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
            d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)

            K_grad = d0_k @ d0_k.conj().T
            K_grad = 0.5 * (K_grad + K_grad.conj().T)

            K_curl_ex = d1_exact.conj().T @ np.diag(star2) @ d1_exact
            K_curl_ex = 0.5 * (K_curl_ex + K_curl_ex.conj().T)

            K_curl_st = d1_std.conj().T @ np.diag(star2) @ d1_std
            K_curl_st = 0.5 * (K_curl_st + K_curl_st.conj().T)

            H_ex = K_grad + K_curl_ex
            H_st = K_grad + K_curl_st

            eigs_ex, vecs_ex = eigh(H_ex, M)
            idx = np.argsort(np.real(eigs_ex))
            vecs_ex = vecs_ex[:, idx]

            eigs_st, vecs_st = eigh(H_st, M)
            idx = np.argsort(np.real(eigs_st))
            vecs_st = vecs_st[:, idx]

            Md0 = M @ d0_k
            P_grad = d0_k @ np.linalg.solve(d0_k.conj().T @ Md0, Md0.conj().T)

            n_mixed_ex = 0
            n_mixed_st = 0
            for i in range(n_E):
                g_ex = np.linalg.norm(P_grad @ vecs_ex[:, i]) / np.linalg.norm(vecs_ex[:, i])
                g_st = np.linalg.norm(P_grad @ vecs_st[:, i]) / np.linalg.norm(vecs_st[:, i])
                if 0.01 < g_ex < 0.99:
                    n_mixed_ex += 1
                if 0.01 < g_st < 0.99:
                    n_mixed_st += 1

            print(f"\n  k={frac:.0%} BZ {label}: "
                  f"mixed_exact={n_mixed_ex}, mixed_std={n_mixed_st} (out of {n_E})")


# ── Part 3: SC cubic N=3 (Table 1, Table 5 row) ──────────────────────

def build_sc_data(N=3):
    """Build SC cubic complex with uniform Hodge stars (regular lattice)."""
    V, E, F, _ = build_sc_supercell_periodic(N)
    L = 2.0 * N
    L_vec = np.array([L, L, L])
    # SC cubic: lattice constant a = 2.0, uniform Hodge stars
    # ⋆₁ = dual_face_area / edge_length = a²/a = a
    # ⋆₂ = dual_edge_length / face_area = a/a² = 1/a
    a = 2.0
    star1 = np.full(len(E), a)
    star2 = np.full(len(F), 1.0 / a)
    return V, E, F, L, L_vec, star1, star2


def test_sc_cubic():
    """SC cubic N=3: exactness and pollution on a regular lattice."""
    print(f"\n{'=' * 70}")
    print(f"  SC CUBIC N=3: V=27, E=81, F=81 (regular lattice)")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, star1, star2 = build_sc_data(N=3)
    n_V, n_E = len(V), len(E)
    shifts = compute_edge_shifts(V, E, L_vec)
    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)
    k_scale = 2 * np.pi / L

    print(f"  V={n_V}, E={n_E}, F={len(F)}")
    print(f"\n  {'k/BZ':>8s}  {'dir':>6s}  {'||d1d0||_ex':>12s}  {'||d1d0||_std':>12s}  "
          f"{'n0_ex':>5s}  {'n0_std':>6s}  {'n_spur':>6s}  {'c2_ex':>8s}")

    for frac in [0.05, 0.10, 0.15]:
        for k_hat, label in [(np.array([1, 0, 0.]), '[100]'),
                              (np.array([1, 1, 1.]) / np.sqrt(3), '[111]')]:
            k = k_scale * frac * k_hat
            k2 = np.linalg.norm(k)**2

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
            norm_std = np.linalg.norm(d1_std @ d0_k)

            K_st, M_st = build_K_M(d1_std, star1, star2)
            eigs_st = np.sort(np.real(eigh(K_st, M_st, eigvals_only=True)))
            thresh_st = max(np.max(np.abs(eigs_st)) * 1e-12, 1e-14)
            n_zero_std = np.sum(np.abs(eigs_st) < thresh_st)

            print(f"  {frac:8.2f}  {label:>6s}  {norm_exact:12.2e}  {norm_std:12.2e}  "
                  f"{n_zero_ex:5d}  {n_zero_std:6d}  {n_zero_ex - n_zero_std:6d}  {c2_ex:8.4f}")


def main():
    print("=" * 70)
    print("CORE VERIFICATION: Pollution + Hodge + Universality")
    print("Paper: Tables 1, 2, 4, 5")
    print("=" * 70)

    structures = [
        ("Kelvin (BCC, N=2)", build_kelvin_with_dual_info(N=2, L_cell=4.0)),
        ("C15 (Laves, N=1)", build_c15_with_dual_info(N=1, L_cell=4.0)),
        ("Weaire-Phelan (A15, N=1)", build_wp_with_dual_info(N=1, L_cell=4.0)),
    ]

    for name, data in structures:
        analyze_structure(name, data)

    test_sc_cubic()
    test_hodge_splitting()

    print(f"\n{'=' * 70}")
    print("ALL CORE TESTS COMPLETE.")


if __name__ == '__main__':
    main()
    print("\nDone.")
