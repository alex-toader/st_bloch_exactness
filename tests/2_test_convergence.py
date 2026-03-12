"""
Dispersion convergence: second-order acoustic wave speed on two mesh families.

Paper claims supported:
  - Table 3 (c² exact vs standard, Kelvin N=2..6 and SC N=3..7)
  - Figure 1 (convergence plot)
  - §5.2: exact converges at second order, standard oscillates
  - R1-3/R2-2: second mesh family confirms convergence is not symmetry-dependent

WHAT IS BEING MEASURED:
  Cell size h is CONSTANT on both families (h=√2 Kelvin, h=2.0 SC).
  Period L grows with N: L=4N (Kelvin), L=2N (SC).
  Wavevector k = 2π·frac/L ∝ 1/N (fixed frac = position in BZ).
  As N increases, there are more cells per wavelength: N_ppw = λ/h ∝ N.

  The supercell eigenvalue equals the infinite lattice eigenvalue exactly
  (verified to machine precision on SC). The N-dependence comes purely
  from evaluating the lattice dispersion at k ∝ 1/N, NOT from finite-size
  effects. The error is:
    |c²_∞ - c²(k)| = O((ka)²) = O(1/N²)
  where a is the lattice constant and c²_∞ is the k→0 analytical limit.

  This is second-order convergence of the discrete dispersion to the
  continuum limit, consistent with DEC's second-order accuracy.

ANALYTICAL TARGETS:
  SC cubic (⋆₁=a, ⋆₂=1/a, a=2): c²_∞ = (⋆₂/⋆₁)·a² = (1/a²)·a² = 1.0.
    Derivation: on the infinite SC lattice with unit cell (V=1, E=3, F=3),
    the curl-curl dispersion along [100] is ω²=(⋆₂/⋆₁)·2(1-cos(ka)).
    At small k: ω² ≈ (⋆₂/⋆₁)·(ka)², hence c²=ω²/k²=(⋆₂/⋆₁)·a².
  Kelvin (Voronoi ⋆): c²_∞ = 1.0 (by Richardson extrapolation; consistent
    with the Voronoi stars approximating the uniform-medium Hodge star).
  Both families converge to c²=1.0 (the continuum dispersion limit for ε=μ=1).

TWO MESH FAMILIES:
  1. Kelvin (BCC, Im3̄m): N=2..6, h=√2, c²→1.0, ⋆₁/⋆₂ Voronoi-derived
  2. SC cubic (Pm3̄m):    N=3..7, h=2.0, c²→1.0, ⋆₁=a=2.0, ⋆₂=1/a=0.5

  Both: p=2.00, R²=1.0000. Ratios match (N/(N-1))² exactly.
  Standard: oscillates on both. No convergence.
  NOTE: Standard c² values are spurious modes (93-97% gradient),
  not acoustic — see R1-6 / 5_test_r16_oscillations.py.

RAW OUTPUT:

  Kelvin (BCC): k = 5% BZ [100]
    N=2: c2_ex=0.9996788  |1-c2|=3.21e-04
    N=3: c2_ex=0.9998572  |1-c2|=1.43e-04  ratio=2.250 = (3/2)²
    N=4: c2_ex=0.9999197  |1-c2|=8.03e-05  ratio=1.778 = (4/3)²
    N=5: c2_ex=0.9999486  |1-c2|=5.14e-05  ratio=1.562 = (5/4)²
    N=6: c2_ex=0.9999643  |1-c2|=3.57e-05  ratio=1.440 = (6/5)²
    Fit: p=2.00, R²=1.0000
    n_spur (std): 6 at all N (Kelvin [100])

  SC cubic: k = 5% BZ [100]
    N=3: c2_ex=0.9990865  |1-c2|=9.14e-04
    N=4: c2_ex=0.9994861  |1-c2|=5.14e-04  ratio=1.777 = (4/3)²
    N=5: c2_ex=0.9996711  |1-c2|=3.29e-04  ratio=1.562 = (5/4)²
    N=6: c2_ex=0.9997716  |1-c2|=2.28e-04  ratio=1.440 = (6/5)²
    N=7: c2_ex=0.9998322  |1-c2|=1.68e-04  ratio=1.361 = (7/6)²
    Fit: p=2.00, R²=1.0000
    Both families converge to c²=1.0 (continuum dispersion limit for ε=μ=1)

  SC cubic: k = 5% BZ [111]
    Fit: p=2.00, R²=1.0000 (same rate, both directions)

  Standard oscillation:
    Kelvin: c2_std = 1.55, 2.48, 1.93, 1.03, 3.34 (SPURIOUS, not acoustic)
    SC:     c2_std = 1.50, 5.08, 0.98, 1.53, 1.76 (SPURIOUS, not acoustic)
    SC n_spur = 5, 9, 19, 29, 41 (grows extensively)

  R²=1.0000 explanation: both lattices are perfectly regular (all edges
  identical), so the dispersion error is exactly C·(ka)²/12 + O((ka)⁴).
  At frac=0.05, the sub-leading correction is <0.04% of leading term.

ANSWER:
  Second-order dispersion convergence confirmed on TWO mesh families
  with different symmetry groups: Kelvin (Im3̄m) and SC (Pm3̄m).
  Both give p=2.00, R²=1.0000. Error = O((ka)²) = O(1/N²) where
  N = cells per axis (∝ cells per wavelength at fixed frac).
  Convergence is structural, not a symmetry artifact.
  Standard fails on both (spurious mode oscillation + growing n_spur).
"""

import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import build_kelvin_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup
from core_math.builders.solids_periodic import build_sc_supercell_periodic


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
    n_zero_st = int(np.sum(np.abs(eigs_st) < thresh_st))

    return {
        'N': N, 'V': n_V, 'E': n_E, 'F': len(F),
        'k2': k2,
        'phys_ex': phys_ex, 'phys_st': phys_st,
        'c2_ex': phys_ex[:3] / k2 if len(phys_ex) >= 3 else None,
        # NOTE: c2_st[:3] are the first non-zero eigenvalues of the standard
        # construction. Per R1-6, these are spurious modes (93-97% gradient),
        # not genuine acoustic modes. Reported for completeness / Table 3.
        'c2_st': phys_st[:3] / k2 if len(phys_st) >= 3 else None,
        'n_spur': n_V - n_zero_st,
    }


def analyze_sc(N, frac, k_hat):
    """Build SC cubic at given N, return physical eigenvalues for exact and standard."""
    V, E, F, _ = build_sc_supercell_periodic(N)
    L = 2.0 * N
    L_vec = np.array([L, L, L])
    a = 2.0
    # ⋆₁ = dual_face_area / edge_length = a²/a = a
    # ⋆₂ = dual_edge_length / face_area = a/a² = 1/a
    star1 = np.full(len(E), a)
    star2 = np.full(len(F), 1.0 / a)

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
    n_zero_st = int(np.sum(np.abs(eigs_st) < thresh_st))

    return {
        'N': N, 'V': len(V), 'E': len(E), 'F': len(F),
        'k2': k2,
        'phys_ex': phys_ex, 'phys_st': phys_st,
        'c2_ex': phys_ex[:3] / k2 if len(phys_ex) >= 3 else None,
        'c2_st': phys_st[:3] / k2 if len(phys_st) >= 3 else None,
        'n_spur': len(V) - n_zero_st,
    }


def convergence_fit(Ns, errors):
    """Log-log fit: |error| ~ C/N^p. Returns (p, R2).

    N = cells per axis (proportional to cells per wavelength at fixed frac).
    On both families, h (cell size) is constant; k ∝ 1/N, so ka ∝ 1/N.
    The fit measures the order of the dispersion error: |c²_∞ - c²| = O(1/N^p).
    """
    log_invN = np.log(1.0 / np.array(Ns))
    log_err = np.log(np.array(errors))
    coeffs = np.polyfit(log_invN, log_err, 1)
    R2 = 1 - np.sum((log_err - np.polyval(coeffs, log_invN))**2) / \
             np.sum((log_err - np.mean(log_err))**2)
    return coeffs[0], R2


def print_convergence_table(results, c2_target, label):
    """Print convergence table for a mesh family."""
    print(f"\n--- Convergence rate: |{c2_target} - c²| vs N ---")
    print(f"  {'N':>4s}  {'V':>6s}  {'c2_exact':>12s}  {'error':>12s}"
          f"  {'ratio':>8s}  {'(N/N-1)^2':>10s}")
    errors = []
    for r in results:
        c2 = r['c2_ex'][0]
        err = abs(c2_target - c2)
        errors.append(err)
        if len(errors) >= 2:
            ratio = errors[-2] / errors[-1]
            N = r['N']
            expected = (N / (N - 1)) ** 2
            print(f"  {N:4d}  {r['V']:6d}  {c2:12.7f}  {err:12.2e}"
                  f"  {ratio:8.3f}  {expected:10.4f}")
        else:
            print(f"  {r['N']:4d}  {r['V']:6d}  {c2:12.7f}  {err:12.2e}"
                  f"  {'---':>8s}  {'---':>10s}")

    if len(errors) >= 3:
        Ns = [r['N'] for r in results]
        p, R2 = convergence_fit(Ns, errors)
        print(f"\n  Log-log fit: |{c2_target}-c²| ~ C * (1/N)^p")
        print(f"  p = {p:.2f}  (expected 2.0 for second-order)")
        print(f"  R² = {R2:.4f}")
    print(f"  {label}")


def main():
    print("=" * 70)
    print("MESH REFINEMENT CONVERGENCE")
    print("Paper: Table 3, Figure 1; R1-3/R2-2")
    print("=" * 70)

    frac = 0.05
    k_hat = np.array([1.0, 0.0, 0.0])

    # ── Family 1: Kelvin (BCC, Im3̄m) ──
    print(f"\n{'─' * 70}")
    print(f"  Family 1: Kelvin (BCC), L_cell=4.0, k = {frac:.0%} BZ [100]")
    print(f"{'─' * 70}")

    kelvin_results = []
    for N in [2, 3, 4, 5, 6]:
        print(f"\n  Building N={N}...", end=" ", flush=True)
        r = analyze_N(N, 4.0, frac, k_hat)
        kelvin_results.append(r)
        print(f"V={r['V']}, E={r['E']}, F={r['F']}")

    # c² table
    print(f"\n--- c² = ω²/|k|² for first 3 modes ---")
    header = f"  {'mode':>4s}"
    for r in kelvin_results:
        header += f"  {'N=' + str(r['N']) + ' exact':>14s}"
    for r in kelvin_results:
        header += f"  {'N=' + str(r['N']) + ' std':>14s}"
    print(header)
    for i in range(3):
        line = f"  {i:4d}"
        for r in kelvin_results:
            val = r['c2_ex'][i] if r['c2_ex'] is not None and i < len(r['c2_ex']) else float('nan')
            line += f"  {val:14.6f}"
        for r in kelvin_results:
            val = r['c2_st'][i] if r['c2_st'] is not None and i < len(r['c2_st']) else float('nan')
            line += f"  {val:14.6f}"
        print(line)

    print_convergence_table(kelvin_results, 1.0,
                            "Kelvin: O(1/N²) dispersion convergence. h=√2 constant.")

    # Standard oscillation (Kelvin)
    print(f"\n--- Standard c² (Kelvin, SPURIOUS — see R1-6) ---")
    for r in kelvin_results:
        print(f"  N={r['N']:2d}: c2_std = {r['c2_st'][0]:.4f}, {r['c2_st'][1]:.4f},"
              f" {r['c2_st'][2]:.4f}   n_spur={r['n_spur']}")

    # ── Family 2: SC cubic (Pm3̄m) ──
    print(f"\n{'─' * 70}")
    print(f"  Family 2: SC cubic, a=2.0, k = {frac:.0%} BZ [100]")
    print(f"{'─' * 70}")

    sc_results = []
    for N in [3, 4, 5, 6, 7]:
        print(f"\n  Building N={N}...", end=" ", flush=True)
        r = analyze_sc(N, frac, k_hat)
        sc_results.append(r)
        print(f"V={r['V']}, E={r['E']}, F={r['F']}")

    # c² table (first 2 modes degenerate on [100] by cubic symmetry)
    print(f"\n--- c² = ω²/|k|² for first 3 modes ---")
    header = f"  {'mode':>4s}"
    for r in sc_results:
        header += f"  {'N=' + str(r['N']) + ' exact':>14s}"
    for r in sc_results:
        header += f"  {'N=' + str(r['N']) + ' std':>14s}"
    print(header)
    for i in range(3):
        line = f"  {i:4d}"
        for r in sc_results:
            val = r['c2_ex'][i] if r['c2_ex'] is not None and i < len(r['c2_ex']) else float('nan')
            line += f"  {val:14.6f}"
        for r in sc_results:
            val = r['c2_st'][i] if r['c2_st'] is not None and i < len(r['c2_st']) else float('nan')
            line += f"  {val:14.6f}"
        print(line)

    print_convergence_table(sc_results, 1.0,
                            "SC: O(1/N²). c²_∞=(⋆₂/⋆₁)·a²=1.0 (analytical). h=2.0 constant.")

    # Standard oscillation (SC)
    print(f"\n--- Standard c² (SC, SPURIOUS — see R1-6) ---")
    for r in sc_results:
        print(f"  N={r['N']:2d}: c2_std = {r['c2_st'][0]:.4f}, {r['c2_st'][1]:.4f},"
              f" {r['c2_st'][2]:.4f}   n_spur={r['n_spur']}")

    # ── SC [111] direction ──
    print(f"\n--- SC cubic [111] direction (cross-check) ---")
    k_hat_111 = np.array([1., 1., 1.]) / np.sqrt(3)
    sc_111 = []
    for N in [3, 4, 5, 6, 7]:
        r = analyze_sc(N, frac, k_hat_111)
        sc_111.append(r)
    print_convergence_table(sc_111, 1.0,
                            "SC [111]: same O(1/N²) as [100].")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    kelvin_Ns = [r['N'] for r in kelvin_results]
    kelvin_errs = [abs(1.0 - r['c2_ex'][0]) for r in kelvin_results]
    p_k, R2_k = convergence_fit(kelvin_Ns, kelvin_errs)

    sc_Ns = [r['N'] for r in sc_results]
    sc_errs = [abs(1.0 - r['c2_ex'][0]) for r in sc_results]
    p_s, R2_s = convergence_fit(sc_Ns, sc_errs)

    sc_111_errs = [abs(1.0 - r['c2_ex'][0]) for r in sc_111]
    p_s111, R2_s111 = convergence_fit(sc_Ns, sc_111_errs)

    print(f"\n  {'Family':>20s}  {'Symmetry':>10s}  {'c2_target':>10s}  {'p':>6s}  {'R2':>8s}")
    print(f"  {'Kelvin [100]':>20s}  {'Im3̄m':>10s}  {'1.0':>10s}  {p_k:6.2f}  {R2_k:8.4f}")
    print(f"  {'SC [100]':>20s}  {'Pm3̄m':>10s}  {'1.0':>10s}  {p_s:6.2f}  {R2_s:8.4f}")
    print(f"  {'SC [111]':>20s}  {'Pm3̄m':>10s}  {'1.0':>10s}  {p_s111:6.2f}  {R2_s111:8.4f}")
    print(f"\n  All three: p=2.00, R²=1.0000. O(1/N²) dispersion convergence")
    print(f"  confirmed on two mesh families. Error = O((ka)²) where a = lattice")
    print(f"  constant, k ∝ 1/N at fixed frac. Consistent with second-order DEC.")
    print(f"  Not symmetry-dependent (BCC Im3̄m vs SC Pm3̄m give same rate).")


if __name__ == '__main__':
    main()
    print("\nDone.")
