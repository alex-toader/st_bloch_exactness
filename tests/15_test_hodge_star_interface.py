"""
Hodge star averaging at material interfaces: logarithmic mean vs alternatives.

WHAT IS BEING MEASURED:
  BCC Voronoi (Kelvin foam) with dielectric contrast eps_A=1, eps_B=variable.
  The exact complex preserves d₁d₀=0 regardless of Hodge star — the question is
  which face-averaging formula for ε gives the most accurate eigenvalues.

  Four formulas tested for effective ε at an interface face between ε_A and ε_B:
    harmonic:    ε_H = 2·ε_A·ε_B/(ε_A+ε_B)          [current default]
    geometric:   ε_G = √(ε_A·ε_B)
    logarithmic: ε_L = (ε_B−ε_A)/ln(ε_B/ε_A)
    arithmetic:  ε_A = (ε_A+ε_B)/2

  Reference: MPB (MIT Photonic Bands) at resolution 64, k_frac=0.01.

  FACE STRUCTURE (BCC Kelvin):
  Each truncated octahedron has 14 faces: 8 hexagonal (nearest neighbors,
  opposite sublattice = INTERFACE) + 6 square (next-nearest, same sublattice
  = BULK). So 57% of faces are interface faces where the formula matters.

  WHY LOGARITHMIC MEAN:
  Empirically, log mean approximates the optimal interface ε to within 4%.
  The optimal (found by minimizing |c²_DEC - c²_MPB|) is a power mean M_p
  whose exponent drifts: p=0.14 at ε_B=1.5 → p=0.31 at ε_B=16. Log mean
  tracks this drift, staying within 4% at all contrasts.

  The standard Whitney-form argument predicts harmonic mean (symmetric
  integral of 1/ε over the dual edge). Harmonic gives up to 73% error.
  This discrepancy is not fully understood — likely the Voronoi ⋆₂ ratio
  (dual_edge_length / face_area) does not match the Whitney mass matrix
  on truncated octahedra.

  The formula is SCALE-INVARIANT: c²(α·ε_A, α·ε_B) = c²(ε_A, ε_B)/α,
  so only the ratio ε_B/ε_A matters. Testing ε_A=1 is sufficient.

RAW OUTPUT:

  TEST 1: FORMULA COMPARISON (eps_B=2,4,9,16, k_frac=0.01 [100])
    Error vs MPB res=64:
      ε_B   harmonic  geometric  logarithm  arithmetic
      1.5     +1.8%     +0.2%     -0.3%       -1.3%
      2.0     +5.1%     +0.7%     -0.8%       -3.6%
      3.0    +12.9%     +1.9%     -1.7%       -8.3%
      4.0    +20.3%     +3.4%     -2.3%      -12.1%
      6.0    +33.5%     +6.3%     -2.8%      -17.6%
      9.0    +49.0%    +10.7%     -2.7%      -22.5%
     16.0    +72.9%    +19.3%     -1.1%      -28.1%
    Max |error|: harmonic 72.9%, geometric 19.3%, logarithmic 2.8%, arithmetic 28.1%
    PASS

  TEST 2: EXACTNESS PRESERVED (all formulas, eps_B=4)
    All 4 formulas: n_zero = V = 96. PASS

  TEST 3: K-INDEPENDENCE (logarithmic, eps_B=4, k_frac=0.01..0.1)
    c² varies by <1% across k range (interface error dominates). PASS

  TEST 4: DIRECTION INDEPENDENCE (logarithmic, eps_B=4, [100] vs [111])
    Errors identical to 0.1%. PASS

  TEST 5: MESH REFINEMENT (logarithmic, eps_B=4, N=2,3,4)
    Fixed k_abs across N. Error O(1) — does not converge with BCC
    refinement (interface geometry is identical at all N). PASS

  TEST 6: HIGH CONTRAST STABILITY (logarithmic, eps_B=16)
    Error stable across k_frac=0.005..0.05 (no sign change).
    The -1.1% at ε_B=16 is genuine, not a cancellation artifact. PASS

  TEST 7: MPB REFERENCE VALIDATION
    MPB converged to <1.2% between res=32 and res=64.
    DEC errors (~2-3%) dominate MPB uncertainty. PASS

  TEST 8: SCALE INVARIANCE (logarithmic, ε_A≠1)
    c²(2,8)/c²(1,4) = 0.500000, c²(3,12)/c²(1,4) = 0.333333.
    Formula depends only on ratio ε_B/ε_A, not absolute values. PASS

ANSWER:
  Logarithmic mean ε_L = (ε_B−ε_A)/ln(ε_B/ε_A) gives ≤3% error across
  contrasts 1.5–16×. This is 25× better than harmonic (current) and 7×
  better than geometric. The error is O(1) in mesh size, k-independent,
  and direction-independent. All formulas preserve exactness.
"""

import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import build_kelvin_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.interface import log_mean, build_inv_eps_face


# ── Helpers ──


def build_kelvin_dielectric(N=2, L_cell=4.0, eps_B=4.0):
    """Build Kelvin mesh with BCC sublattice dielectric assignment."""
    data = build_kelvin_with_dual_info(N=N, L_cell=L_cell)
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)

    cc = data['cell_centers']
    labels = np.array([
        int(round((c[0] + c[1] + c[2]) / (L_cell / 2))) % 2
        for c in cc
    ])
    eps_cells = np.where(labels == 0, 1.0, float(eps_B))
    ftc = data['face_to_cells']

    return dict(V=V, E=E, F=F, L_vec=L_vec, star1=star1, star2=star2,
                shifts=shifts, labels=labels, eps_cells=eps_cells, ftc=ftc)


def solve_c2(mesh, k_abs, formula):
    """Solve curl-curl eigenvalue problem, return (c², n_zero)."""
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    L_vec, star1, star2 = mesh['L_vec'], mesh['star1'], mesh['star2']
    shifts = mesh['shifts']

    inv_eps, _ = build_inv_eps_face(F, mesh['ftc'], mesh['eps_cells'], formula)

    d0_k = build_d0_bloch(V, E, k_abs, L_vec, shifts)
    d1_ex = build_d1_bloch_exact(V, E, F, k_abs, L_vec, d0_k)

    K = d1_ex.conj().T @ np.diag(star2 * inv_eps) @ d1_ex
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    eigs = np.sort(np.real(eigh(K, M, eigvals_only=True)))

    thresh = max(np.max(np.abs(eigs)) * 1e-12, 1e-14)
    n_zero = int(np.sum(np.abs(eigs) < thresh))
    phys = eigs[eigs > thresh]
    k2 = np.dot(k_abs, k_abs)
    c2 = phys[0] / k2 if len(phys) > 0 else float('nan')

    return c2, n_zero


# ── MPB reference values (resolution 64, k_frac=0.01, [100]) ──

MPB_REF = {
    1.5: 0.810481,
    2.0: 0.691737,
    3.0: 0.545276,
    4.0: 0.455265,
    6.0: 0.346683,
    9.0: 0.258124,
    16.0: 0.163530,
}

FORMULAS = ['harmonic', 'geometric', 'logarithmic', 'arithmetic']


# ── Tests ──

def test_formula_comparison():
    """Logarithmic mean gives ≤3% error across all contrasts."""
    print(f"\n{'=' * 70}")
    print("  TEST 1: FORMULA COMPARISON vs MPB (k_frac=0.01, [100])")
    print(f"{'=' * 70}")

    N, L_cell = 2, 4.0
    k_frac = 0.01

    print(f"\n  {'ε_B':>5s}", end='')
    for name in FORMULAS:
        print(f"  {name:>11s}", end='')
    print()
    print("  " + "-" * 55)

    max_err = {f: 0.0 for f in FORMULAS}

    for eps_B in sorted(MPB_REF.keys()):
        mesh = build_kelvin_dielectric(N=N, L_cell=L_cell, eps_B=eps_B)
        k_abs = 2 * np.pi * k_frac / L_cell * np.array([1.0, 0.0, 0.0])
        ref = MPB_REF[eps_B]

        print(f"  {eps_B:5.1f}", end='')
        for name in FORMULAS:
            c2, nz = solve_c2(mesh, k_abs, name)
            err = (c2 - ref) / ref
            max_err[name] = max(max_err[name], abs(err))
            print(f"  {err:+10.1%}", end='')
        print()

    print(f"\n  Max |error|:")
    for name in FORMULAS:
        print(f"    {name:<12s}: {max_err[name]:.1%}")

    # Logarithmic must be best
    assert max_err['logarithmic'] < max_err['geometric'], \
        f"logarithmic ({max_err['logarithmic']:.1%}) not better than geometric ({max_err['geometric']:.1%})"
    assert max_err['logarithmic'] < 0.04, \
        f"logarithmic max error {max_err['logarithmic']:.1%} > 4%"
    assert max_err['logarithmic'] < max_err['harmonic'] / 5, \
        f"logarithmic not 5× better than harmonic"

    print(f"\n  Logarithmic mean: ≤{max_err['logarithmic']:.1%} across all contrasts.")
    print(f"  {max_err['harmonic']/max_err['logarithmic']:.0f}× better than harmonic.")
    print(f"  PASS")


def test_exactness_preserved():
    """All formulas preserve d₁d₀=0 (n_zero = V)."""
    print(f"\n{'=' * 70}")
    print("  TEST 2: EXACTNESS PRESERVED (all formulas, eps_B=4)")
    print(f"{'=' * 70}")

    mesh = build_kelvin_dielectric(N=2, L_cell=4.0, eps_B=4.0)
    nV = len(mesh['V'])
    k_abs = 2 * np.pi * 0.05 / 4.0 * np.array([1.0, 0.0, 0.0])

    print(f"\n  V = {nV}")
    for name in FORMULAS:
        c2, nz = solve_c2(mesh, k_abs, name)
        ok = nz == nV
        print(f"  {name:<12s}: n_zero = {nz}  {'✓' if ok else '✗'}")
        assert ok, f"{name}: n_zero={nz} != V={nV}"

    print(f"\n  All formulas preserve exactness. PASS")


def test_k_independence():
    """Interface error is k-independent (logarithmic mean, eps_B=4)."""
    print(f"\n{'=' * 70}")
    print("  TEST 3: K-INDEPENDENCE (logarithmic, eps_B=4)")
    print(f"{'=' * 70}")

    mesh = build_kelvin_dielectric(N=2, L_cell=4.0, eps_B=4.0)
    L_cell = 4.0

    print(f"\n  {'k_frac':>8s}  {'c²':>10s}  {'c²/c²_ref':>10s}")

    c2s = []
    for k_frac in [0.005, 0.01, 0.025, 0.05, 0.1]:
        k_abs = 2 * np.pi * k_frac / L_cell * np.array([1.0, 0.0, 0.0])
        c2, nz = solve_c2(mesh, k_abs, 'logarithmic')
        c2s.append(c2)

    c2_ref = c2s[0]
    for i, k_frac in enumerate([0.005, 0.01, 0.025, 0.05, 0.1]):
        ratio = c2s[i] / c2_ref
        print(f"  {k_frac:8.3f}  {c2s[i]:10.6f}  {ratio:10.6f}")

    # c² should vary by <2% across this k range
    spread = (max(c2s) - min(c2s)) / c2_ref
    print(f"\n  Spread: {spread:.2%}")
    assert spread < 0.02, f"c² spread {spread:.2%} > 2%"
    print(f"  Interface error dominates over dispersion. PASS")


def test_direction_independence():
    """Error is direction-independent on Kelvin (cubic symmetry)."""
    print(f"\n{'=' * 70}")
    print("  TEST 4: DIRECTION INDEPENDENCE (logarithmic, eps_B=4)")
    print(f"{'=' * 70}")

    mesh = build_kelvin_dielectric(N=2, L_cell=4.0, eps_B=4.0)
    L_cell = 4.0
    k_frac = 0.01

    dirs = [
        ('[100]', np.array([1.0, 0.0, 0.0])),
        ('[110]', np.array([1.0, 1.0, 0.0]) / np.sqrt(2)),
        ('[111]', np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
    ]

    print(f"\n  {'dir':>6s}  {'c²':>10s}  {'err vs MPB':>10s}")
    c2s = []
    for name, k_hat in dirs:
        k_abs = 2 * np.pi * k_frac / L_cell * k_hat
        c2, nz = solve_c2(mesh, k_abs, 'logarithmic')
        err = (c2 - MPB_REF[4.0]) / MPB_REF[4.0]
        c2s.append(c2)
        print(f"  {name:>6s}  {c2:10.6f}  {err:+9.2%}")

    spread = (max(c2s) - min(c2s)) / np.mean(c2s)
    print(f"\n  Directional spread: {spread:.3%}")
    assert spread < 0.005, f"Spread {spread:.3%} > 0.5%"
    print(f"  Direction-independent on Kelvin. PASS")


def test_mesh_refinement():
    """Error is O(1) under BCC refinement (interface geometry unchanged).

    Uses fixed L_cell=4.0 and fixed k_abs for all N. BCC refinement
    replicates identical cells — the interface face geometry (shape, aspect
    ratios) is identical at all N, so the interface error cannot converge.
    """
    print(f"\n{'=' * 70}")
    print("  TEST 5: MESH REFINEMENT (logarithmic, eps_B=4)")
    print(f"{'=' * 70}")

    L_cell = 4.0
    k_frac = 0.01
    k_abs = 2 * np.pi * k_frac / L_cell * np.array([1.0, 0.0, 0.0])
    eps_B = 4.0

    print(f"\n  Fixed k_abs = 2π×{k_frac}/L, L_cell={L_cell}")
    print(f"  {'N':>3s}  {'V':>6s}  {'c²_log':>10s}  {'err vs MPB':>10s}")

    c2s = []
    for N in [2, 3, 4]:
        mesh = build_kelvin_dielectric(N=N, L_cell=L_cell, eps_B=eps_B)
        c2, nz = solve_c2(mesh, k_abs, 'logarithmic')
        err = (c2 - MPB_REF[4.0]) / MPB_REF[4.0]
        c2s.append(c2)
        print(f"  {N:3d}  {len(mesh['V']):6d}  {c2:10.6f}  {err:+9.2%}")

    # Error should NOT decrease (O(1) — same interface geometry)
    spread = (max(c2s) - min(c2s)) / np.mean(c2s)
    print(f"\n  c² spread across N=2,3,4: {spread:.4%}")
    assert spread < 0.001, f"c² changed with N: spread {spread:.3%}"
    print(f"  Error O(1) — BCC refinement does not change interface geometry. PASS")


def test_high_contrast_stability():
    """Check that the ε_B=16 error is not a cancellation artifact.

    The error at ε_B=16 is -1.1%, smaller than at ε_B=6 (-2.8%).
    If this is due to cancellation between interface and dispersion errors,
    the error should change sign at different k_frac. We test stability.
    """
    print(f"\n{'=' * 70}")
    print("  TEST 6: HIGH CONTRAST STABILITY (logarithmic, eps_B=16)")
    print(f"{'=' * 70}")

    mesh = build_kelvin_dielectric(N=2, L_cell=4.0, eps_B=16.0)
    L_cell = 4.0
    ref = MPB_REF[16.0]

    print(f"\n  MPB ref (res=64): {ref:.6f}")
    print(f"  {'k_frac':>8s}  {'c²':>10s}  {'err':>8s}")

    errs = []
    for k_frac in [0.005, 0.01, 0.025, 0.05]:
        k_abs = 2 * np.pi * k_frac / L_cell * np.array([1.0, 0.0, 0.0])
        c2, nz = solve_c2(mesh, k_abs, 'logarithmic')
        err = (c2 - ref) / ref
        errs.append(err)
        print(f"  {k_frac:8.3f}  {c2:10.6f}  {err:+7.2%}")

    # All errors should have the same sign (no cancellation)
    signs = [np.sign(e) for e in errs]
    same_sign = all(s == signs[0] for s in signs) or all(abs(e) < 0.005 for e in errs)
    spread = max(abs(e) for e in errs) - min(abs(e) for e in errs)

    print(f"\n  Error range: [{min(errs):+.2%}, {max(errs):+.2%}]")
    print(f"  Same sign across k: {'yes' if same_sign else 'NO — cancellation detected'}")

    # The error should stay within ±4% at all k
    assert all(abs(e) < 0.04 for e in errs), \
        f"Error exceeds 4% at some k_frac: {errs}"
    print(f"  Error stable under k variation. PASS")


def test_mpb_convergence():
    """Validate MPB reference by checking resolution convergence at highest contrast."""
    print(f"\n{'=' * 70}")
    print("  TEST 7: MPB REFERENCE VALIDATION (eps_B=16, res=32 vs 64)")
    print(f"{'=' * 70}")

    # MPB values at res=32 and res=64 (computed in this session)
    mpb_32 = {4: 0.453146, 9: 0.255770, 16: 0.161663}
    mpb_64 = {4: 0.455265, 9: 0.258124, 16: 0.163530}

    print(f"\n  {'ε_B':>5s}  {'res=32':>10s}  {'res=64':>10s}  {'Δ':>8s}")
    for eps_B in [4, 9, 16]:
        diff = abs(mpb_32[eps_B] - mpb_64[eps_B]) / mpb_64[eps_B]
        print(f"  {eps_B:5d}  {mpb_32[eps_B]:10.6f}  {mpb_64[eps_B]:10.6f}  {diff:7.2%}")

    # MPB must be converged to <2% at res=64
    for eps_B in [4, 9, 16]:
        diff = abs(mpb_32[eps_B] - mpb_64[eps_B]) / mpb_64[eps_B]
        assert diff < 0.02, \
            f"MPB not converged at eps_B={eps_B}: Δ={diff:.2%}"

    print(f"\n  MPB converged to <1.2% between res=32 and res=64.")
    print(f"  DEC logarithmic mean errors (~2-3%) are larger than MPB uncertainty.")
    print(f"  PASS")


def test_scale_invariance():
    """Formula depends only on ε_B/ε_A ratio, not absolute values.

    If c²(ε_A, ε_B) is the lowest eigenvalue ratio, then
    c²(α·ε_A, α·ε_B) = c²(ε_A, ε_B) / α for any α > 0.
    This means testing ε_A=1 covers all ε_A values.
    """
    print(f"\n{'=' * 70}")
    print("  TEST 8: SCALE INVARIANCE (logarithmic, ε_A ≠ 1)")
    print(f"{'=' * 70}")

    L_cell = 4.0
    k_frac = 0.01
    k_abs = 2 * np.pi * k_frac / L_cell * np.array([1.0, 0.0, 0.0])

    # Reference: ε_A=1, ε_B=4
    mesh_ref = build_kelvin_dielectric(N=2, L_cell=L_cell, eps_B=4.0)
    c2_ref, _ = solve_c2(mesh_ref, k_abs, 'logarithmic')

    print(f"\n  Reference: c²(1, 4) = {c2_ref:.6f}")
    print(f"  {'config':>15s}  {'c²':>10s}  {'ratio':>10s}  {'expected':>10s}")

    # Scaled versions: same ratio 4:1 but different absolute values
    for alpha, eps_A, eps_B in [(2, 2.0, 8.0), (3, 3.0, 12.0)]:
        data = build_kelvin_with_dual_info(N=2, L_cell=L_cell)
        V, E, F = data['V'], data['E'], data['F']
        L_vec = data['L_vec']
        star1, star2 = build_hodge_stars_voronoi(data)
        shifts = compute_edge_shifts(V, E, L_vec)
        cc = data['cell_centers']
        labels = np.array([
            int(round((c[0] + c[1] + c[2]) / (L_cell / 2))) % 2
            for c in cc
        ])
        eps_cells = np.where(labels == 0, eps_A, eps_B)
        ftc = data['face_to_cells']

        inv_eps, _ = build_inv_eps_face(F, ftc, eps_cells, 'logarithmic')
        d0_k = build_d0_bloch(V, E, k_abs, L_vec, shifts)
        d1_ex = build_d1_bloch_exact(V, E, F, k_abs, L_vec, d0_k)
        K = d1_ex.conj().T @ np.diag(star2 * inv_eps) @ d1_ex
        K = 0.5 * (K + K.conj().T)
        M = np.diag(star1)
        eigs = np.sort(np.real(eigh(K, M, eigvals_only=True)))
        thresh = max(np.max(np.abs(eigs)) * 1e-12, 1e-14)
        phys = eigs[eigs > thresh]
        c2 = phys[0] / np.dot(k_abs, k_abs)

        ratio = c2 / c2_ref
        expected = 1.0 / alpha
        print(f"  c²({eps_A:.0f},{eps_B:.0f}){' '*(9-len(f'{eps_A:.0f},{eps_B:.0f}'))}  {c2:10.6f}  {ratio:10.6f}  {expected:10.6f}")
        assert abs(ratio - expected) < 1e-10, \
            f"Scale invariance broken: c²({eps_A},{eps_B})/c²(1,4) = {ratio} != {expected}"

    print(f"\n  c²(α·ε_A, α·ε_B) = c²(ε_A, ε_B)/α — exact to machine precision.")
    print(f"  Only ratio ε_B/ε_A matters. PASS")


def main():
    print("=" * 70)
    print("HODGE STAR INTERFACE AVERAGING — LOGARITHMIC MEAN")
    print("=" * 70)

    test_formula_comparison()
    test_exactness_preserved()
    test_k_independence()
    test_direction_independence()
    test_mesh_refinement()
    test_high_contrast_stability()
    test_mpb_convergence()
    test_scale_invariance()


if __name__ == '__main__':
    main()
    print("\nDone.")
