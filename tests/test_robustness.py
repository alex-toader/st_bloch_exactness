"""
Robustness tests: k→0 asymptotics, random k-directions, mesh distortion,
BZ boundary behavior, face ordering independence, gauge transform invariance,
operator perturbation norms.

Paper claims supported:
  - Table 6 (robustness summary, including gauge transform row)
  - §6: "failure persists under refinement, random topology, random k-directions,
         metric perturbations, and k→0 asymptotics"
  - §6: "operator norm ||K_std - K_exact||_F as a function of |k|"
  - Proposition 2 numerical confirmation (canonical K)
  - Reproducibility section: "operator perturbation norms"

Usage:
    cd /path/to/st_bloch_exactness
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_robustness.py

Expected output (macOS, Python 3.9.6, SciPy 1.13):

    ======================================================================
    ROBUSTNESS TESTS
    Paper: Table 6, §6
    ======================================================================

    ======================================================================
      k→0 ASYMPTOTICS + THRESHOLD SENSITIVITY
    ======================================================================

    --- (a) c² = ω²/|k|² as k→0 ---
            k/BZ      c2_exact        c2_std  n_zero_std
         1.0e-04    0.99999999    1.57025974          90
         1.0e-03    0.99999987    1.57025105          90
         1.0e-02    0.99998715    1.56938240          90
         5.0e-02    0.99967878    1.54866092          90
         1.0e-01    0.99871579    1.48335031          90
         2.0e-01    0.99487390    1.06273617          90

    --- (b) Threshold sensitivity ---
       threshold_rel    n_zero  expected    OK
               1e-08        96        96   YES
               1e-10        96        96   YES
               1e-12        96        96   YES
               1e-14        96        96   YES

    ======================================================================
      RANDOM k-DIRECTIONS
    ======================================================================

           dir  n0_std  n0_ex  n_spur   ||d1d0||_std   ||d1d0||_ex
         [100]      90     96       6     7.5444e+00      7.86e-16
         [010]      89     96       7     7.6450e+00      8.58e-16
         [001]      90     96       6     7.5033e+00      8.35e-16
         rnd01      82     96      14     7.4104e+00      8.32e-16
         rnd02      82     96      14     7.8396e+00      1.02e-15
         ...        82     96      14     ~7-8            ~1e-15
         rnd20      82     96      14     7.6948e+00      8.84e-16

    ======================================================================
      MESH DISTORTION ROBUSTNESS
    ======================================================================

         eps   seed     V   ||d1d0||_ex  n0_ex  n_spur    status
        0.00     42    96      7.86e-16     96       6      PASS
        0.01     42    96      9.59e-16     96      10      PASS
        0.05     42    96      7.88e-16     96       9      PASS
        0.10     42    96      1.08e-15     96       9      PASS
        ...  (all 12 configurations PASS)

    ======================================================================
      BZ BOUNDARY BEHAVIOR
    ======================================================================

          frac   rank_d0   ||d1d0||_ex  n0_ex      status
         0.100        96      7.86e-16     96        PASS
         0.500        96      6.36e-32     96        PASS
         0.900        96      4.82e-16     96        PASS
         0.999        96      2.56e-16     96        PASS
         1.000        95      2.55e-31     98    DEGRADED

    ======================================================================
      FACE ORDERING INDEPENDENCE (Proposition 2)
    ======================================================================

              mode   ||K-K_ref||_F     max|Δeig|      status
            cycle1        2.91e-15      9.77e-15   IDENTICAL
            cycle2        5.16e-15      3.66e-15   IDENTICAL
           reverse        4.56e-15      5.33e-15   IDENTICAL
            random        3.39e-15      5.11e-15   IDENTICAL
            random        3.07e-15      4.00e-15   IDENTICAL
            random        3.00e-15      5.33e-15   IDENTICAL

    ======================================================================
      GAUGE TRANSFORM INVARIANCE (Table 6)
    ======================================================================

        seed     max|Δeig|      ||d1d0||      status
          42      3.55e-15      1.76e-15   IDENTICAL
         137      4.66e-15      1.87e-15   IDENTICAL
         999      5.35e-15      1.88e-15   IDENTICAL
        2024      4.44e-15      1.64e-15   IDENTICAL
       31415      5.33e-15      2.19e-15   IDENTICAL

    ======================================================================
      OPERATOR NORM ||K_std - K_exact||_F vs |k|
    ======================================================================

          k/BZ  ||K_std-K_ex||_F    ||K_ex||_F    rel_diff   ||d1d0||_std
          0.01        2.0929e+00    7.2737e+01      0.0288     8.0412e-01
          0.02        4.1771e+00    7.2737e+01      0.0574     1.6051e+00
          0.05        1.0290e+01    7.2737e+01      0.1415     3.9588e+00
          0.10        1.9526e+01    7.2737e+01      0.2684     7.5444e+00
          0.15        2.6834e+01    7.2737e+01      0.3689     1.0447e+01
          0.20        3.1669e+01    7.2737e+01      0.4354     1.2471e+01
          0.30        3.3772e+01    7.2737e+01      0.4643     1.3823e+01

    ======================================================================
    ALL ROBUSTNESS TESTS COMPLETE.
"""

import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import (
    build_kelvin_with_dual_info,
    get_bcc_points, build_foam_with_dual_info,
    build_hodge_stars_voronoi,
)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup


def build_K_M(d1_k, star1, star2):
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    return K, M


def setup_kelvin():
    """Build Kelvin N=2 with all derived quantities."""
    data = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)
    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)
    return V, E, F, L, L_vec, star1, star2, shifts, crossings, edge_lookup


# ── Part 1: k→0 asymptotics + threshold sensitivity ─────────────────────

def test_asymptotics():
    """c² stability as k→0, gradient leakage persistence, threshold robustness."""
    print(f"\n{'=' * 70}")
    print(f"  k→0 ASYMPTOTICS + THRESHOLD SENSITIVITY")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, star1, star2, shifts, crossings, edge_lookup = setup_kelvin()
    n_V, n_E = len(V), len(E)
    k_scale = 2 * np.pi / L

    # (a) Dispersion ratio
    print(f"\n--- (a) c² = ω²/|k|² as k→0 ---")
    print(f"  {'k/BZ':>10s}  {'c2_exact':>12s}  {'c2_std':>12s}  {'n_zero_std':>10s}")

    for frac in [1e-4, 1e-3, 1e-2, 5e-2, 0.10, 0.20]:
        k = k_scale * frac * np.array([1.0, 0.0, 0.0])
        k2 = np.linalg.norm(k)**2

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        K_ex, M_ex = build_K_M(d1_exact, star1, star2)
        eigs_ex = np.sort(np.real(eigh(K_ex, M_ex, eigvals_only=True)))
        thresh_ex = max(np.max(np.abs(eigs_ex)) * 1e-12, 1e-14)
        phys_ex = eigs_ex[np.abs(eigs_ex) >= thresh_ex]

        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)
        K_st, M_st = build_K_M(d1_std, star1, star2)
        eigs_st = np.sort(np.real(eigh(K_st, M_st, eigvals_only=True)))
        thresh_st = max(np.max(np.abs(eigs_st)) * 1e-12, 1e-14)
        n_zero_std = np.sum(np.abs(eigs_st) < thresh_st)
        phys_st = eigs_st[np.abs(eigs_st) >= thresh_st]

        c2_ex = phys_ex[0] / k2 if len(phys_ex) > 0 else float('nan')
        c2_st = phys_st[0] / k2 if len(phys_st) > 0 else float('nan')
        print(f"  {frac:10.1e}  {c2_ex:12.8f}  {c2_st:12.8f}  {n_zero_std:10d}")

    # (b) Threshold sensitivity
    print(f"\n--- (b) Threshold sensitivity ---")
    k = k_scale * 0.10 * np.array([1.0, 0.0, 0.0])
    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    K_ex, M_ex = build_K_M(d1_exact, star1, star2)
    eigs_ex = np.sort(np.real(eigh(K_ex, M_ex, eigvals_only=True)))
    max_eig = np.max(np.abs(eigs_ex))

    print(f"  {'threshold_rel':>14s}  {'n_zero':>8s}  {'expected':>8s}  {'OK':>4s}")
    for thr_rel in [1e-8, 1e-10, 1e-12, 1e-14]:
        thr_abs = max(max_eig * thr_rel, 1e-18)
        n_z = np.sum(np.abs(eigs_ex) < thr_abs)
        ok = "YES" if n_z == n_V else "NO"
        print(f"  {thr_rel:14.0e}  {n_z:8d}  {n_V:8d}  {ok:>4s}")


# ── Part 2: Random k-directions ─────────────────────────────────────────

def test_random_directions():
    """Direction-dependence of spurious mode count."""
    print(f"\n{'=' * 70}")
    print(f"  RANDOM k-DIRECTIONS")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, star1, star2, shifts, crossings, edge_lookup = setup_kelvin()
    n_V, n_E = len(V), len(E)
    k_scale = 2 * np.pi / L
    frac = 0.10

    rng = np.random.RandomState(42)
    dirs = rng.randn(20, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    axis_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    all_dirs = np.vstack([axis_dirs, dirs])
    labels = ['[100]', '[010]', '[001]'] + [f'rnd{i+1:02d}' for i in range(20)]

    print(f"\n  {'dir':>8s}  {'n0_std':>6s}  {'n0_ex':>5s}  {'n_spur':>6s}  {'||d1d0||_std':>13s}  {'||d1d0||_ex':>12s}")

    for k_hat, label in zip(all_dirs, labels):
        k = k_scale * frac * k_hat

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)

        norm_std = np.linalg.norm(d1_std @ d0_k)
        norm_exact = np.linalg.norm(d1_exact @ d0_k)

        K_ex, _ = build_K_M(d1_exact, star1, star2)
        eigs_ex = np.sort(np.real(eigh(K_ex, np.diag(star1), eigvals_only=True)))
        thresh_ex = max(np.max(np.abs(eigs_ex)) * 1e-12, 1e-14)
        n_zero_ex = np.sum(np.abs(eigs_ex) < thresh_ex)

        K_st, _ = build_K_M(d1_std, star1, star2)
        eigs_st = np.sort(np.real(eigh(K_st, np.diag(star1), eigvals_only=True)))
        thresh_st = max(np.max(np.abs(eigs_st)) * 1e-12, 1e-14)
        n_zero_std = np.sum(np.abs(eigs_st) < thresh_st)

        print(f"  {label:>8s}  {n_zero_std:6d}  {n_zero_ex:5d}  {n_zero_ex - n_zero_std:6d}  {norm_std:13.4e}  {norm_exact:12.2e}")


# ── Part 3: Mesh distortion ─────────────────────────────────────────────

def test_mesh_distortion():
    """Exactness under geometric perturbation of cell centers."""
    print(f"\n{'=' * 70}")
    print(f"  MESH DISTORTION ROBUSTNESS")
    print(f"{'=' * 70}")

    N, L_cell = 2, 4.0
    L = N * L_cell
    k_scale = 2 * np.pi / L
    k = k_scale * 0.10 * np.array([1.0, 0.0, 0.0])

    print(f"\n  {'eps':>6s}  {'seed':>5s}  {'V':>4s}  {'||d1d0||_ex':>12s}  "
          f"{'n0_ex':>5s}  {'n_spur':>6s}  {'status':>8s}")

    for eps in [0.00, 0.01, 0.05, 0.10]:
        for seed in [42, 123, 456]:
            rng = np.random.RandomState(seed)
            points = get_bcc_points(N, L_cell)
            if eps > 0:
                points = (points + rng.randn(len(points), 3) * eps * L_cell) % L

            try:
                data = build_foam_with_dual_info(points, L)
            except Exception:
                print(f"  {eps:6.2f}  {seed:5d}  {'--':>4s}  {'--':>12s}  {'--':>5s}  {'--':>6s}  {'BUILD_ERR':>8s}")
                continue

            V, E, F = data['V'], data['E'], data['F']
            L_vec = data['L_vec']
            n_V = len(V)

            try:
                star1, star2 = build_hodge_stars_voronoi(data)
            except Exception:
                print(f"  {eps:6.2f}  {seed:5d}  {n_V:4d}  {'--':>12s}  {'--':>5s}  {'--':>6s}  {'HODGE_ERR':>8s}")
                continue

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

            d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)
            K_st, M_st = build_K_M(d1_std, star1, star2)
            eigs_st = np.sort(np.real(eigh(K_st, M_st, eigvals_only=True)))
            thresh_st = max(np.max(np.abs(eigs_st)) * 1e-12, 1e-14)
            n_zero_std = np.sum(np.abs(eigs_st) < thresh_st)

            status = "PASS" if (norm_exact < 1e-12 and n_zero_ex == n_V) else "FAIL"
            print(f"  {eps:6.2f}  {seed:5d}  {n_V:4d}  {norm_exact:12.2e}  "
                  f"{n_zero_ex:5d}  {n_zero_ex - n_zero_std:6d}  {status:>8s}")


# ── Part 4: BZ boundary ─────────────────────────────────────────────────

def test_bz_boundary():
    """Behavior at singular k-points (BZ boundary)."""
    print(f"\n{'=' * 70}")
    print(f"  BZ BOUNDARY BEHAVIOR")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, star1, star2, shifts, crossings, edge_lookup = setup_kelvin()
    n_V = len(V)
    k_bz = 2 * np.pi / L

    print(f"\n  {'frac':>8s}  {'rank_d0':>8s}  {'||d1d0||_ex':>12s}  {'n0_ex':>5s}  {'status':>10s}")

    for frac in [0.10, 0.50, 0.90, 0.999, 1.000]:
        k = k_bz * frac * np.array([1.0, 0.0, 0.0])
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        rank_d0 = np.linalg.matrix_rank(d0_k, tol=1e-10)

        d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        norm_exact = np.linalg.norm(d1_exact @ d0_k)

        K_ex, M_ex = build_K_M(d1_exact, star1, star2)
        eigs_ex = np.sort(np.real(eigh(K_ex, M_ex, eigvals_only=True)))
        thresh_ex = max(np.max(np.abs(eigs_ex)) * 1e-12, 1e-14)
        n_zero_ex = np.sum(np.abs(eigs_ex) < thresh_ex)

        status = "PASS" if (norm_exact < 1e-12 and n_zero_ex == n_V) else "DEGRADED"
        print(f"  {frac:8.3f}  {rank_d0:8d}  {norm_exact:12.2e}  {n_zero_ex:5d}  {status:>10s}")


# ── Part 5: Face ordering independence ───────────────────────────────────

def test_face_ordering():
    """K canonical under vertex permutation within faces (Prop 2)."""
    print(f"\n{'=' * 70}")
    print(f"  FACE ORDERING INDEPENDENCE (Proposition 2)")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, star1, star2, shifts, _, _ = setup_kelvin()
    k_scale = 2 * np.pi / L
    k = k_scale * 0.10 * np.array([1.0, 0.0, 0.0])

    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_ref = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    K_ref = d1_ref.conj().T @ np.diag(star2) @ d1_ref
    K_ref = 0.5 * (K_ref + K_ref.conj().T)
    eigs_ref = np.sort(np.real(eigh(K_ref, np.diag(star1), eigvals_only=True)))

    np.random.seed(42)
    print(f"\n  {'mode':>12s}  {'||K-K_ref||_F':>14s}  {'max|Δeig|':>12s}  {'status':>10s}")

    for mode in ['cycle1', 'cycle2', 'reverse', 'random', 'random', 'random']:
        F_perm = []
        for face in F:
            if mode == 'cycle1':
                F_perm.append(face[1:] + face[:1])
            elif mode == 'cycle2':
                F_perm.append(face[2:] + face[:2])
            elif mode == 'reverse':
                F_perm.append([face[0]] + face[1:][::-1])
            elif mode == 'random':
                n = len(face)
                shift = np.random.randint(0, n)
                F_perm.append(face[shift:] + face[:shift])

        d1_perm = build_d1_bloch_exact(V, E, F_perm, k, L_vec, d0_k)
        K_perm = d1_perm.conj().T @ np.diag(star2) @ d1_perm
        K_perm = 0.5 * (K_perm + K_perm.conj().T)
        eigs_perm = np.sort(np.real(eigh(K_perm, np.diag(star1), eigvals_only=True)))

        diff_K = np.linalg.norm(K_perm - K_ref, 'fro')
        diff_eig = np.max(np.abs(eigs_perm - eigs_ref))
        status = "IDENTICAL" if diff_K < 1e-10 else "DIFFERENT"

        print(f"  {mode:>12s}  {diff_K:14.2e}  {diff_eig:12.2e}  {status:>10s}")


# ── Part 6: Gauge transform invariance ─────────────────────────────────

def test_gauge_transform():
    """Eigenvalues invariant under vertex gauge transform u_v → e^{iθ(v)} u_v."""
    print(f"\n{'=' * 70}")
    print(f"  GAUGE TRANSFORM INVARIANCE (Table 6)")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, star1, star2, shifts, crossings, edge_lookup = setup_kelvin()
    n_V, n_E = len(V), len(E)
    k_scale = 2 * np.pi / L
    k = k_scale * 0.10 * np.array([1.0, 0.0, 0.0])

    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    K_ref = d1_exact.conj().T @ np.diag(star2) @ d1_exact
    K_ref = 0.5 * (K_ref + K_ref.conj().T)
    M = np.diag(star1)
    eigs_ref = np.sort(np.real(eigh(K_ref, M, eigvals_only=True)))

    print(f"\n  {'seed':>6s}  {'max|Δeig|':>12s}  {'||d1d0||':>12s}  {'status':>10s}")

    for seed in [42, 137, 999, 2024, 31415]:
        rng = np.random.RandomState(seed)
        theta = rng.uniform(0, 2 * np.pi, n_V)
        # Gauge transform: D = diag(e^{iθ(v)})
        # d0' = d0 @ D^{-1}, d1' = d1 (unchanged in curl-curl)
        # K' = d1'^† ⋆₂ d1' with d1' built from gauged d0'
        D_inv = np.diag(np.exp(-1j * theta))
        d0_gauged = d0_k @ D_inv
        d1_gauged = build_d1_bloch_exact(V, E, F, k, L_vec, d0_gauged)
        norm_exact = np.linalg.norm(d1_gauged @ d0_gauged)

        K_g = d1_gauged.conj().T @ np.diag(star2) @ d1_gauged
        K_g = 0.5 * (K_g + K_g.conj().T)
        eigs_g = np.sort(np.real(eigh(K_g, M, eigvals_only=True)))

        diff_eig = np.max(np.abs(eigs_g - eigs_ref))
        status = "IDENTICAL" if diff_eig < 1e-10 else "DIFFERENT"
        print(f"  {seed:6d}  {diff_eig:12.2e}  {norm_exact:12.2e}  {status:>10s}")


# ── Part 7: Operator norm ||K_std - K_exact||_F vs |k| ────────────────

def test_operator_norm():
    """Operator perturbation norm grows with |k|."""
    print(f"\n{'=' * 70}")
    print(f"  OPERATOR NORM ||K_std - K_exact||_F vs |k|")
    print(f"{'=' * 70}")

    V, E, F, L, L_vec, star1, star2, shifts, crossings, edge_lookup = setup_kelvin()
    k_scale = 2 * np.pi / L

    print(f"\n  {'k/BZ':>8s}  {'||K_std-K_ex||_F':>16s}  {'||K_ex||_F':>12s}  {'rel_diff':>10s}  {'||d1d0||_std':>13s}")

    for frac in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        k = k_scale * frac * np.array([1.0, 0.0, 0.0])

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_exact = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)

        K_ex = d1_exact.conj().T @ np.diag(star2) @ d1_exact
        K_ex = 0.5 * (K_ex + K_ex.conj().T)
        K_st = d1_std.conj().T @ np.diag(star2) @ d1_std
        K_st = 0.5 * (K_st + K_st.conj().T)

        diff_K = np.linalg.norm(K_st - K_ex, 'fro')
        norm_ex = np.linalg.norm(K_ex, 'fro')
        rel = diff_K / norm_ex if norm_ex > 0 else 0
        norm_std = np.linalg.norm(d1_std @ d0_k)
        print(f"  {frac:8.2f}  {diff_K:16.4e}  {norm_ex:12.4e}  {rel:10.4f}  {norm_std:13.4e}")


def main():
    print("=" * 70)
    print("ROBUSTNESS TESTS")
    print("Paper: Table 6, §6")
    print("=" * 70)

    test_asymptotics()
    test_random_directions()
    test_mesh_distortion()
    test_bz_boundary()
    test_face_ordering()
    test_gauge_transform()
    test_operator_norm()

    print(f"\n{'=' * 70}")
    print("ALL ROBUSTNESS TESTS COMPLETE.")


if __name__ == '__main__':
    main()
    print("\nDone.")
