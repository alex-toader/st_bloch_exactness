"""
Mesh refinement convergence: O(h²) acoustic wave speed.

Paper claims supported:
  - Table 3 (c² exact vs standard, N=2..6)
  - Figure 1 (convergence plot)
  - §5.2: exact converges at second order, standard oscillates

Usage:
    cd /path/to/st_bloch_exactness
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_convergence.py

Expected output (macOS, Python 3.9.6, SciPy 1.13):

    ======================================================================
    MESH REFINEMENT CONVERGENCE
    Paper: Table 3, Figure 1
    ======================================================================

    Kelvin, L_cell=4.0, k = 5% BZ [100]

      Building N=2... V=96, E=192, F=112
      Building N=3... V=324, E=648, F=378
      Building N=4... V=768, E=1536, F=896
      Building N=5... V=1500, E=3000, F=1750
      Building N=6... V=2592, E=5184, F=3024

    --- c² = ω²/|k|² for first 3 modes ---
      mode       N=2 exact       N=3 exact       N=4 exact       N=5 exact       N=6 exact         N=2 std         N=3 std         N=4 std         N=5 std         N=6 std
         0        0.999679        0.999857        0.999920        0.999949        0.999964        1.548661        2.476476        1.933329        1.028741        3.340576
         1        0.999679        0.999857        0.999920        0.999949        0.999964        2.324948        4.277027        3.072450        2.657859        4.166731
         2      321.538794      342.893465      350.694339      354.367666      356.380192        4.405924        4.467454        4.004709        4.871567        5.002041

    --- Convergence rate: |1 - c²| vs N ---
         N       V      c2_exact        |1-c2|     ratio   (N/N-1)^2
         2      96     0.9996788      3.21e-04       ---         ---
         3     324     0.9998572      1.43e-04     2.250      2.2500
         4     768     0.9999197      8.03e-05     1.778      1.7778
         5    1500     0.9999486      5.14e-05     1.562      1.5625
         6    2592     0.9999643      3.57e-05     1.440      1.4400

      Log-log fit: |1-c²| ~ C * (1/N)^p
      p = 2.00  (expected 2.0 for second-order)
      R² = 1.0000

      NOTE: Ratios match (N/(N-1))^2 exactly because the uniform BCC mesh
      produces a single-term h^2 error with no sub-leading corrections.
"""

import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import build_kelvin_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup


def build_K_M(d1_k, star1, star2):
    K = d1_k.conj().T @ np.diag(star2) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    return K, M


def analyze_N(N, L_cell, frac, k_hat):
    """Build Kelvin at given N, return physical eigenvalues for exact and standard."""
    data = build_kelvin_with_dual_info(N=N, L_cell=L_cell)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    n_V, n_E = len(V), len(E)

    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)
    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)

    k_scale = 2 * np.pi / L
    k = k_scale * frac * k_hat
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
    phys_st = eigs_st[np.abs(eigs_st) >= thresh_st]

    return {
        'N': N, 'V': n_V, 'E': n_E, 'F': len(F),
        'k2': k2,
        'phys_ex': phys_ex, 'phys_st': phys_st,
        'c2_ex': phys_ex[:3] / k2 if len(phys_ex) >= 3 else None,
        'c2_st': phys_st[:3] / k2 if len(phys_st) >= 3 else None,
    }


def main():
    print("=" * 70)
    print("MESH REFINEMENT CONVERGENCE")
    print("Paper: Table 3, Figure 1")
    print("=" * 70)

    L_cell = 4.0
    frac = 0.05
    k_hat = np.array([1.0, 0.0, 0.0])

    Ns = [2, 3, 4, 5, 6]

    print(f"\nKelvin, L_cell={L_cell}, k = {frac:.0%} BZ [100]")

    results = []
    for N in Ns:
        print(f"\n  Building N={N}...", end=" ", flush=True)
        try:
            r = analyze_N(N, L_cell, frac, k_hat)
            results.append(r)
            print(f"V={r['V']}, E={r['E']}, F={r['F']}")
        except Exception as e:
            print(f"FAILED: {e}")

    # c² table
    print(f"\n--- c² = ω²/|k|² for first 3 modes ---")
    header = f"  {'mode':>4s}"
    for r in results:
        header += f"  {'N=' + str(r['N']) + ' exact':>14s}"
    for r in results:
        header += f"  {'N=' + str(r['N']) + ' std':>14s}"
    print(header)

    for i in range(3):
        line = f"  {i:4d}"
        for r in results:
            val = r['c2_ex'][i] if r['c2_ex'] is not None and i < len(r['c2_ex']) else float('nan')
            line += f"  {val:14.6f}"
        for r in results:
            val = r['c2_st'][i] if r['c2_st'] is not None and i < len(r['c2_st']) else float('nan')
            line += f"  {val:14.6f}"
        print(line)

    # Convergence rate
    print(f"\n--- Convergence rate: |1 - c²| vs N ---")
    print(f"  {'N':>4s}  {'V':>6s}  {'c2_exact':>12s}  {'|1-c2|':>12s}  {'ratio':>8s}  {'(N/N-1)^2':>10s}")
    errors = []
    for r in results:
        c2 = r['c2_ex'][0]
        err = abs(1.0 - c2)
        errors.append(err)
        if len(errors) >= 2:
            ratio = errors[-2] / errors[-1]
            N = r['N']
            expected = (N / (N - 1)) ** 2
            print(f"  {N:4d}  {r['V']:6d}  {c2:12.7f}  {err:12.2e}  {ratio:8.3f}  {expected:10.4f}")
        else:
            print(f"  {r['N']:4d}  {r['V']:6d}  {c2:12.7f}  {err:12.2e}  {'---':>8s}  {'---':>10s}")

    if len(errors) >= 3:
        Ns_arr = np.array([r['N'] for r in results])
        log_h = np.log(1.0 / Ns_arr)
        log_err = np.log(np.array(errors))
        coeffs = np.polyfit(log_h, log_err, 1)
        R2 = 1 - np.sum((log_err - np.polyval(coeffs, log_h))**2) / np.sum((log_err - np.mean(log_err))**2)
        print(f"\n  Log-log fit: |1-c²| ~ C * (1/N)^p")
        print(f"  p = {coeffs[0]:.2f}  (expected 2.0 for second-order)")
        print(f"  R² = {R2:.4f}")
        print(f"\n  NOTE: Ratios match (N/(N-1))^2 exactly because the uniform BCC mesh")
        print(f"  produces a single-term h^2 error with no sub-leading corrections.")


if __name__ == '__main__':
    main()
    print("\nDone.")
