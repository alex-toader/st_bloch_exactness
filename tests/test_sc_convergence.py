"""
M5 Verification: SC Cubic Analytic Benchmark
=============================================

SC cubic is the analytic benchmark: c² = a² = 4.0 exactly.

Tests:
  1. c² vs k: dispersion converges to c²=4 as k→0
  2. Finite-size: c² at fixed k_frac improves with N (supercell size)
  3. Degeneracy: split ~ 10⁻¹² (machine precision)
  4. Perturbed SC Voronoi: exactness+kernel robust, degeneracy breaks with ε
  5. Anisotropic box: exactness+kernel robust, c² becomes directional
  6. Perturbed Hodge stars: ker=V always (topological), c²+split change (metric)

Run:
  OPENBLAS_NUM_THREADS=1 /usr/bin/python3 publishing/wip/8_test_M5_sc_convergence.py

RAW OUTPUT (run Feb 2026):

TEST 1: c² vs k (N=3, a=2, analytic c²=4)
  k_frac  c²_[100]    c²_[111]    |c²-4|
  0.005   3.999963    3.999988    3.7e-05
  0.010   3.999854    3.999951    1.5e-04
  0.020   3.999415    3.999805    5.9e-04
  0.050   3.996346    3.998782    3.7e-03
  0.100   3.985400    3.995128    1.5e-02
  0.200   3.941854    3.980542    5.8e-02
  Isotropy at k_frac=0.005: |c²_100 - c²_111| = 2.44e-05

TEST 2: Finite-size at 2% BZ [100]
  N=3 (V=27):  c² = 3.9994, |c²-4| = 5.9e-04
  N=4 (V=64):  c² = 3.9997, |c²-4| = 3.3e-04
  N=5 (V=125): c² = 3.9998, |c²-4| = 2.1e-04

TEST 3: Degeneracy at 2% BZ
  [100]: split = 2.25e-12
  [110]: split = 8.13e-12
  [111]: split = 4.90e-13

TEST 4: Perturbed SC Voronoi (3 seeds × 3 dirs, 2% BZ)
  eps=0.00: split=3.2e-12 (machine), exact+ker=YES
  eps=0.01: split=2.2e-07, exact+ker=YES
  eps=0.05: split=9.2e-07, exact+ker=YES
  eps=0.10: split=1.6e-06, exact+ker=YES
  eps=0.20: split=2.8e-06, exact+ker=YES
  Degeneracy breaks (symmetry), exactness+kernel robust (topology)

TEST 5: Anisotropic box (non-cubic L_vec)
  cubic:    c²_100=4.00, c²_010=4.00, c²_001=4.00 (isotropic)
  mild:     c²_100=4.00, c²_010=4.84, c²_001=3.24 (directional)
  moderate: c²_100=4.00, c²_010=6.76, c²_001=1.96 (directional)
  Exactness+ker=V on all (topological)

TEST 6: Perturbed Hodge stars
  6a: *₁ perturbation: c² shifts up (4.00→4.09 at eps=0.20), ker=V always
  6b: *₂ perturbation: c² shifts down (4.00→3.70 at eps=0.20), ker=V always
  Split breaks from 10⁻¹² to 10⁻² (metric, not topological)

ALL TESTS PASSED

Feb 2026
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

from core_math.builders.solids_periodic import build_sc_supercell_periodic
from physics.bloch import build_hodge_stars_uniform
from physics.hodge import build_foam_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact


def solve_sc(N, k):
    """Build SC mesh, solve Maxwell EVP, return c² and split."""
    V, E, F, _ = build_sc_supercell_periodic(N)
    L = 2.0 * N
    L_vec = np.array([L, L, L])
    a = 2.0

    _, s1m, s2m, _, _ = build_hodge_stars_uniform(len(V), len(E), len(F), a=a)
    star1 = np.diag(s1m)
    star2 = np.diag(s2m)
    shifts = compute_edge_shifts(V, E, L_vec)

    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    eigvals = np.sort(np.real(eigh(K, M, eigvals_only=True)))
    thr = max(np.max(np.abs(eigvals)) * 1e-12, 1e-14)
    nz = int(np.sum(np.abs(eigvals) < thr))
    phys = eigvals[np.abs(eigvals) >= thr]

    assert nz == len(V), f"ker={nz} != V={len(V)}"

    k2 = np.dot(k, k)
    c2 = (phys[0] + phys[1]) / (2 * k2)
    split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
    return c2, split, nz, len(V)


# =========================================================================
# TEST 1: c² vs k
# =========================================================================

def test_dispersion():
    print("=" * 70)
    print("TEST 1: c² vs k (N=3, a=2, analytic c²=a²=4)")
    print("=" * 70)
    print()

    N = 3
    L = 2.0 * N
    k_scale = 2 * np.pi / L

    k_fracs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

    print(f"  {'k_frac':>8} {'c²_[100]':>10} {'c²_[111]':>10} {'|c²-4|_100':>12}")
    print("  " + "-" * 44)

    for frac in k_fracs:
        c2_100, _, _, _ = solve_sc(N, k_scale * frac * np.array([1, 0, 0]))
        c2_111, _, _, _ = solve_sc(N, k_scale * frac * np.array([1, 1, 1]) / np.sqrt(3))
        err = abs(c2_100 - 4.0)
        print(f"  {frac:>8.3f} {c2_100:>10.6f} {c2_111:>10.6f} {err:>12.6f}")

    # At smallest k, error should be < 0.01%
    c2_best, _, _, _ = solve_sc(N, k_scale * 0.005 * np.array([1, 0, 0]))
    c2_111_best, _, _, _ = solve_sc(N, k_scale * 0.005 * np.array([1, 1, 1]) / np.sqrt(3))
    assert abs(c2_best - 4.0) < 0.001, f"c² at smallest k: {c2_best:.6f}, error {abs(c2_best-4):.6f}"
    assert abs(c2_best - c2_111_best) < 1e-3, f"|c²_100 - c²_111| = {abs(c2_best - c2_111_best):.2e}"
    print()
    print(f"  Isotropy at k_frac=0.005: |c²_100 - c²_111| = {abs(c2_best - c2_111_best):.2e}")
    print("  PASSED: c² → 4.0 as k → 0, isotropic")
    print()


# =========================================================================
# TEST 2: Finite-size
# =========================================================================

def test_finite_size():
    print("=" * 70)
    print("TEST 2: Finite-size at 2% BZ [100]")
    print("=" * 70)
    print()

    print(f"  {'N':>3} {'V':>5} {'c²':>10} {'|c²-4|':>10}")
    print("  " + "-" * 32)

    prev_err = None
    for N in [3, 4, 5]:
        L = 2.0 * N
        k = (2 * np.pi / L) * 0.02 * np.array([1, 0, 0])
        c2, _, _, n_V = solve_sc(N, k)
        err = abs(c2 - 4.0)
        print(f"  {N:>3} {n_V:>5} {c2:>10.6f} {err:>10.6f}")
        if prev_err is not None:
            assert err < prev_err, f"Error not decreasing: N={N} err={err} > prev={prev_err}"
        prev_err = err

    print()
    print("  PASSED: error decreases with N")
    print()


# =========================================================================
# TEST 3: Degeneracy
# =========================================================================

def test_degeneracy():
    print("=" * 70)
    print("TEST 3: Degeneracy at 2% BZ")
    print("=" * 70)
    print()

    N = 3
    L = 2.0 * N
    k_scale = 2 * np.pi / L
    frac = 0.02

    directions = [
        ("[100]", np.array([1.0, 0.0, 0.0])),
        ("[110]", np.array([1.0, 1.0, 0.0]) / np.sqrt(2)),
        ("[111]", np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
    ]

    print(f"  {'dir':<8} {'split':>12}")
    print("  " + "-" * 22)

    for label, d in directions:
        k = k_scale * frac * d
        _, split, _, _ = solve_sc(N, k)
        print(f"  {label:<8} {split:>12.2e}")
        assert split < 1e-8, f"Split too large at {label}: {split}"

    print()
    print("  PASSED: machine-precision degeneracy")
    print()


# =========================================================================
# TEST 4: Perturbed SC — degeneracy breaking
# =========================================================================

def test_perturbed():
    print("=" * 70)
    print("TEST 4: Perturbed SC Voronoi — symmetry breaking")
    print("=" * 70)
    print()

    N = 3
    L_cell = 2.0
    L = N * L_cell
    pts0 = []
    for i in range(N):
        for j in range(N):
            for kk in range(N):
                pts0.append([L_cell*i + L_cell/2, L_cell*j + L_cell/2,
                             L_cell*kk + L_cell/2])
    pts0 = np.array(pts0)

    k_scale = 2 * np.pi / L
    epsilons = [0.0, 0.01, 0.05, 0.10, 0.20]
    seeds = [42, 137, 999]
    dirs = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([1, 1, 1]) / np.sqrt(3),
    ]

    print(f"  {'eps':>6} {'mean_split':>12} {'max_split':>12} {'mean_c²':>10} {'exact+ker':>10}")
    print("  " + "-" * 52)

    prev_mean = None
    for eps in epsilons:
        splits, c2s, all_ok = [], [], True
        for seed in seeds:
            np.random.seed(seed)
            pts = pts0 + eps * np.random.randn(*pts0.shape)
            pts = pts % L
            data = build_foam_with_dual_info(pts, L)
            V, E, F = data['V'], data['E'], data['F']
            L_vec = data['L_vec']
            star1, star2 = build_hodge_stars_voronoi(data)
            shifts = compute_edge_shifts(V, E, L_vec)
            for d in dirs:
                k = k_scale * 0.02 * d
                d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
                d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
                err = np.linalg.norm(d1_k @ d0_k)
                K = d1_k.conj().T @ np.diag(star2) @ d1_k
                K = 0.5 * (K + K.conj().T)
                M = np.diag(star1)
                eigvals = np.sort(np.real(eigh(K, M, eigvals_only=True)))
                thr = max(np.max(np.abs(eigvals)) * 1e-12, 1e-14)
                nz = int(np.sum(np.abs(eigvals) < thr))
                phys = eigvals[np.abs(eigvals) >= thr]
                k2 = np.dot(k, k)
                c2 = (phys[0] + phys[1]) / (2 * k2)
                split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
                splits.append(split)
                c2s.append(c2)
                if err > 1e-10 or nz != len(V):
                    all_ok = False

        ms = np.mean(splits)
        label = "YES" if all_ok else "NO"
        print(f"  {eps:>6.2f} {ms:>12.2e} {np.max(splits):>12.2e} "
              f"{np.mean(c2s):>10.6f} {label:>10}")

        assert all_ok, f"Exactness or kernel failed at eps={eps}"
        if eps > 0:
            assert ms > 1e-10, f"Split should be nonzero at eps={eps}"
        prev_mean = ms

    print()
    print("  PASSED: exactness+kernel robust; degeneracy breaks with perturbation")
    print()


# =========================================================================
# TEST 5: Anisotropic box — L_vec != cubic
# =========================================================================

def test_anisotropic_box():
    print("=" * 70)
    print("TEST 5: Anisotropic box (non-cubic L_vec)")
    print("=" * 70)
    print()

    N = 3
    V, E, F, _ = build_sc_supercell_periodic(N)
    a = 2.0
    _, s1m, s2m, _, _ = build_hodge_stars_uniform(len(V), len(E), len(F), a=a)
    star1 = np.diag(s1m)
    star2 = np.diag(s2m)
    L0 = 2.0 * N
    n_V = len(V)

    scalings = [
        ("cubic",    [1.0, 1.0, 1.0]),
        ("mild",     [1.0, 1.1, 0.9]),
        ("moderate", [1.0, 1.3, 0.7]),
    ]

    dirs = [
        ("[100]", np.array([1.0, 0.0, 0.0])),
        ("[010]", np.array([0.0, 1.0, 0.0])),
        ("[001]", np.array([0.0, 0.0, 1.0])),
    ]

    print(f"  {'scaling':<10} {'dir':<8} {'exactness':>10} {'ker':>4} "
          f"{'c²':>10} {'split':>10}")
    print("  " + "-" * 56)

    all_ok = True
    for slabel, s in scalings:
        L_vec = np.array([L0 * s[0], L0 * s[1], L0 * s[2]])
        shifts = compute_edge_shifts(V, E, L_vec)

        for dname, d in dirs:
            k = (2 * np.pi / L_vec) * 0.02 * d
            k2 = np.dot(k, k)
            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
            err = np.linalg.norm(d1_k @ d0_k)
            K = d1_k.conj().T @ np.diag(star2) @ d1_k
            K = 0.5 * (K + K.conj().T)
            M = np.diag(star1)
            eigvals = np.sort(np.real(eigh(K, M, eigvals_only=True)))
            thr = max(np.max(np.abs(eigvals)) * 1e-12, 1e-14)
            nz = int(np.sum(np.abs(eigvals) < thr))
            phys = eigvals[np.abs(eigvals) >= thr]
            c2 = (phys[0] + phys[1]) / (2 * k2)
            split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)

            if err > 1e-10 or nz != n_V:
                all_ok = False
            print(f"  {slabel:<10} {dname:<8} {err:>10.2e} {nz:>4} "
                  f"{c2:>10.4f} {split:>10.2e}")

    assert all_ok, "Exactness or kernel failed under anisotropic stretching"
    print()
    print("  PASSED: exactness+kernel robust; c² becomes directional")
    print()


# =========================================================================
# TEST 6: Perturbed Hodge stars
# =========================================================================

def test_perturbed_stars():
    print("=" * 70)
    print("TEST 6: Perturbed Hodge stars (*₁ and *₂)")
    print("=" * 70)
    print()

    N = 3
    V, E, F, _ = build_sc_supercell_periodic(N)
    a = 2.0
    _, s1m, s2m, _, _ = build_hodge_stars_uniform(len(V), len(E), len(F), a=a)
    star1_0 = np.diag(s1m)
    star2_0 = np.diag(s2m)
    n_V = len(V)

    L = 2.0 * N
    L_vec = np.array([L, L, L])
    shifts = compute_edge_shifts(V, E, L_vec)

    k = (2 * np.pi / L) * 0.02 * np.array([1.0, 0.0, 0.0])
    k2 = np.dot(k, k)

    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_k = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    err = np.linalg.norm(d1_k @ d0_k)
    K0 = d1_k.conj().T @ np.diag(star2_0) @ d1_k
    K0 = 0.5 * (K0 + K0.conj().T)

    epsilons = [0.0, 0.01, 0.05, 0.10, 0.20]

    # 6a: perturb star1
    print("  6a: Perturbed *₁ (mass matrix)")
    print(f"  {'eps':>6} {'ker':>4} {'c²':>10} {'split':>10}")
    print("  " + "-" * 34)

    all_ok = True
    for eps in epsilons:
        np.random.seed(42)
        star1 = star1_0 * (1.0 + eps * np.random.randn(len(star1_0)))
        star1 = np.abs(star1)
        M = np.diag(star1)
        eigvals = np.sort(np.real(eigh(K0, M, eigvals_only=True)))
        thr = max(np.max(np.abs(eigvals)) * 1e-12, 1e-14)
        nz = int(np.sum(np.abs(eigvals) < thr))
        phys = eigvals[np.abs(eigvals) >= thr]
        c2 = (phys[0] + phys[1]) / (2 * k2)
        split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
        if nz != n_V:
            all_ok = False
        print(f"  {eps:>6.2f} {nz:>4} {c2:>10.4f} {split:>10.2e}")

    # 6b: perturb star2
    print()
    print("  6b: Perturbed *₂ (stiffness weight)")
    print(f"  {'eps':>6} {'ker':>4} {'c²':>10} {'split':>10}")
    print("  " + "-" * 34)

    for eps in epsilons:
        np.random.seed(42)
        star2_p = star2_0 * (1.0 + eps * np.random.randn(len(star2_0)))
        star2_p = np.abs(star2_p)
        K_p = d1_k.conj().T @ np.diag(star2_p) @ d1_k
        K_p = 0.5 * (K_p + K_p.conj().T)
        M = np.diag(star1_0)
        eigvals = np.sort(np.real(eigh(K_p, M, eigvals_only=True)))
        thr = max(np.max(np.abs(eigvals)) * 1e-12, 1e-14)
        nz = int(np.sum(np.abs(eigvals) < thr))
        phys = eigvals[np.abs(eigvals) >= thr]
        c2 = (phys[0] + phys[1]) / (2 * k2)
        split = abs(phys[1] - phys[0]) / max(abs(phys[0]), 1e-14)
        if nz != n_V:
            all_ok = False
        print(f"  {eps:>6.2f} {nz:>4} {c2:>10.4f} {split:>10.2e}")

    assert all_ok, "Kernel count changed under star perturbation"
    print()
    print("  PASSED: ker=V always (topological); c² and split change (metric)")
    print()


# =========================================================================
# MAIN
# =========================================================================

def main():
    print()
    print("=" * 70)
    print("M5: SC Cubic Analytic Benchmark (c² = a² = 4)")
    print("=" * 70)
    print()

    test_dispersion()
    test_finite_size()
    test_degeneracy()
    test_perturbed()
    test_anisotropic_box()
    test_perturbed_stars()

    print("ALL TESTS PASSED")
    print()


if __name__ == '__main__':
    main()
