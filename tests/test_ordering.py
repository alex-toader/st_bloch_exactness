"""
M5 Verification: Ordering Independence of K = d₁(k)† *₂ d₁(k)
===============================================================

The exactness recurrence builds d₁(k) face-by-face, starting from
phase[0] = 1 at the first vertex. Two questions:

  Q1: Does the spectrum change if we start from a different vertex?
  Q2: Does the spectrum change if we reverse face orientation?

Answer: d₁ itself changes (||Δd₁|| ~ 10¹), but K = d₁†*₂d₁ does NOT.

Reason: all recurrence phases have unit modulus (flat connection,
holonomy = 1). Changing the starting vertex multiplies row d₁[f,:]
by α_f with |α_f| = 1. Reversing orientation multiplies by -1.
In both cases |α_f|² = 1, so K is invariant.

Tests:
  1. Random cyclic rotation of starting vertex (all faces)
  2. Random reversal of face orientation (50% of faces)
  3. Both simultaneously
  4. Multiple random seeds to confirm robustness
  5. All 3 structures (C15, Kelvin, WP)

Run:
  OPENBLAS_NUM_THREADS=1 /usr/bin/python3 publishing/wip/5_test_M5_ordering.py

RAW OUTPUT (run Feb 2026):

TEST 1: Per-structure ordering independence (rotation + reversal)
  Structure  n_F   ||d1_diff||   ||K_diff||  max|eig_diff|  |phase|-1
  C15        160      4.12e+01     1.07e-14       3.55e-14   2.22e-16
  Kelvin     112      3.71e+01     5.70e-15       4.44e-15   2.22e-16
  WP          54      2.42e+01     5.59e-15       4.44e-15   2.22e-16

TEST 2: Multiple scrambling seeds on C15
  seed    ||d1_diff||   ||K_diff||  max|eig_diff|
  0          3.79e+01     1.03e-14       3.20e-14
  1          4.02e+01     9.70e-15       1.91e-14
  42         4.12e+01     1.07e-14       3.55e-14
  99         3.90e+01     9.06e-15       4.09e-14
  777        3.76e+01     1.08e-14       3.38e-14

TEST 3: Separate effects on C15
  rotation only:      ||K_diff|| = 1.01e-14, max|eig_diff| = 2.09e-14
  reversal only:      ||K_diff|| = 7.76e-15, max|eig_diff| = 2.31e-14
  rotation+reversal:  ||K_diff|| = 1.07e-14, max|eig_diff| = 3.55e-14

CONCLUSION:
  d₁(k) is NOT unique (depends on face vertex ordering).
  K = d₁†*₂d₁ IS unique (invariant under ordering choices).
  Reason: phases have |α| = 1 (flat connection), so |α|² = 1 in K.

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
    build_c15_with_dual_info,
    build_kelvin_with_dual_info,
    build_wp_with_dual_info,
    build_hodge_stars_voronoi,
)
from physics.gauge_bloch import (
    compute_edge_shifts,
    build_d0_bloch,
    build_d1_bloch_exact,
)


STRUCTURES = [
    ("C15",    build_c15_with_dual_info,    1, 4.0),
    ("Kelvin", build_kelvin_with_dual_info, 2, 4.0),
    ("WP",     build_wp_with_dual_info,     1, 4.0),
]

K_GENERIC = np.array([0.37, 0.23, 0.11])


def scramble_faces(F, seed, rotate=True, reverse=True):
    """Randomly rotate starting vertex and/or reverse orientation."""
    rng = np.random.RandomState(seed)
    F_new = []
    for face in F:
        n = len(face)
        f = list(face)
        if rotate:
            shift = rng.randint(0, n)
            f = f[shift:] + f[:shift]
        if reverse and rng.random() < 0.5:
            f = list(reversed(f))
        F_new.append(f)
    return F_new


def build_and_solve(V, E, F, L_vec, k, star1, star2):
    """Build d₁, K, solve EVP. Return d1, K, sorted eigenvalues."""
    shifts = compute_edge_shifts(V, E, L_vec)
    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    eigvals = np.sort(np.real(eigh(K, M, eigvals_only=True)))
    return d1_k, K, eigvals


def max_phase_deviation(d1, E, F):
    """Check that all phases in d₁ have unit modulus."""
    edge_map = {}
    for idx, (i, j) in enumerate(E):
        edge_map[(i, j)] = (idx, +1)
        edge_map[(j, i)] = (idx, -1)
    max_dev = 0.0
    for f_idx, face in enumerate(F):
        for v_pos in range(len(face)):
            vi = face[v_pos]
            vj = face[(v_pos + 1) % len(face)]
            e_idx, orient = edge_map[(vi, vj)]
            phase = d1[f_idx, e_idx] / orient
            max_dev = max(max_dev, abs(abs(phase) - 1.0))
    return max_dev


# =========================================================================
# TEST 1: All structures, rotation + reversal
# =========================================================================

def test_all_structures():
    print("=" * 70)
    print("TEST 1: Per-structure ordering independence (rotation + reversal)")
    print("=" * 70)
    print()
    print(f"  {'Structure':<10} {'n_F':>4} {'||d1_diff||':>13} "
          f"{'||K_diff||':>12} {'max|eig_diff|':>14} {'|phase|-1':>10}")
    print("  " + "-" * 67)

    all_ok = True
    for name, builder, N, L_cell in STRUCTURES:
        data = builder(N=N, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L_vec = data['L_vec']
        star1, star2 = build_hodge_stars_voronoi(data)

        d1_ref, K_ref, eig_ref = build_and_solve(V, E, F, L_vec, K_GENERIC, star1, star2)
        phase_dev = max_phase_deviation(d1_ref, E, F)

        F_scr = scramble_faces(F, seed=42)
        d1_scr, K_scr, eig_scr = build_and_solve(V, E, F_scr, L_vec, K_GENERIC, star1, star2)

        d1_diff = np.linalg.norm(d1_ref - d1_scr)
        K_diff = np.linalg.norm(K_ref - K_scr)
        eig_diff = np.max(np.abs(eig_ref - eig_scr))

        print(f"  {name:<10} {len(F):>4} {d1_diff:>13.2e} "
              f"{K_diff:>12.2e} {eig_diff:>14.2e} {phase_dev:>10.2e}")

        assert eig_diff < 1e-10, f"{name}: eig_diff = {eig_diff:.2e}"
        assert K_diff < 1e-10, f"{name}: K_diff = {K_diff:.2e}"
        assert d1_diff > 1.0, f"{name}: d1 should be different, got ||diff|| = {d1_diff:.2e}"
        assert phase_dev < 1e-14, f"{name}: phases not unit modulus, dev = {phase_dev:.2e}"

    print()
    print("  PASSED: K invariant on all structures despite different d₁")
    print()


# =========================================================================
# TEST 2: Multiple scrambling seeds on C15
# =========================================================================

def test_multiple_seeds():
    print("=" * 70)
    print("TEST 2: Multiple scrambling seeds on C15")
    print("=" * 70)
    print()

    data = build_c15_with_dual_info(N=1, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)

    d1_ref, K_ref, eig_ref = build_and_solve(V, E, F, L_vec, K_GENERIC, star1, star2)

    seeds = [0, 1, 42, 99, 777]
    print(f"  {'seed':>6} {'||d1_diff||':>13} {'||K_diff||':>12} {'max|eig_diff|':>14}")
    print("  " + "-" * 49)

    max_eig_diff_all = 0.0
    for seed in seeds:
        F_scr = scramble_faces(F, seed=seed)
        d1_scr, K_scr, eig_scr = build_and_solve(V, E, F_scr, L_vec, K_GENERIC, star1, star2)

        d1_diff = np.linalg.norm(d1_ref - d1_scr)
        K_diff = np.linalg.norm(K_ref - K_scr)
        eig_diff = np.max(np.abs(eig_ref - eig_scr))
        max_eig_diff_all = max(max_eig_diff_all, eig_diff)

        print(f"  {seed:>6} {d1_diff:>13.2e} {K_diff:>12.2e} {eig_diff:>14.2e}")

    print()
    assert max_eig_diff_all < 1e-10, f"max eig_diff across seeds = {max_eig_diff_all:.2e}"
    print(f"  PASSED: Invariant across {len(seeds)} random scramblings")
    print()


# =========================================================================
# TEST 3: Separate rotation vs reversal effects
# =========================================================================

def test_separate_effects():
    print("=" * 70)
    print("TEST 3: Separate effects on C15")
    print("=" * 70)
    print()

    data = build_c15_with_dual_info(N=1, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)

    _, K_ref, eig_ref = build_and_solve(V, E, F, L_vec, K_GENERIC, star1, star2)

    configs = [
        ("rotation only",      True,  False),
        ("reversal only",      False, True),
        ("rotation+reversal",  True,  True),
    ]

    for label, do_rot, do_rev in configs:
        F_mod = scramble_faces(F, seed=42, rotate=do_rot, reverse=do_rev)
        _, K_mod, eig_mod = build_and_solve(V, E, F_mod, L_vec, K_GENERIC, star1, star2)

        K_diff = np.linalg.norm(K_ref - K_mod)
        eig_diff = np.max(np.abs(eig_ref - eig_mod))

        print(f"  {label:<22} ||K_diff|| = {K_diff:.2e}, max|eig_diff| = {eig_diff:.2e}")

        assert eig_diff < 1e-10, f"{label}: eig_diff = {eig_diff:.2e}"

    print()
    print("  PASSED: Both rotation and reversal separately leave K invariant")
    print()


# =========================================================================
# MAIN
# =========================================================================

def main():
    print()
    print("=" * 70)
    print("M5: Ordering Independence of Curl-Curl Operator")
    print("=" * 70)
    print()

    test_all_structures()
    test_multiple_seeds()
    test_separate_effects()

    print("ALL TESTS PASSED")
    print()


if __name__ == '__main__':
    main()
