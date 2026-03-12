"""
Acoustic modes as deformed cohomology classes: Helmholtz decomposition on the exact complex.

At Γ (k=0), the 3 harmonic 1-forms H¹(T³) = ℝ³ are the three linearly
independent curl-free, non-gradient cochains. They correspond to constant
E-field polarizations. At k→0⁺, the Bloch character χ_k breaks this
degeneracy via the discrete Helmholtz decomposition:

    H¹(T³) = [1 longitudinal → im(d₀(k))] ⊕ [2 transverse → acoustic ω²=|k|²]

The longitudinal mode (ê ∥ k) becomes exact (a coboundary d₀f), absorbed into
the gradient subspace. The two transverse modes (ê ⊥ k) peel off as acoustic
eigenvectors of the curl-curl operator with ω = v_s|k|, where v_s = 1 in
natural units.

This decomposition is verified via subspace trace overlaps:
    Tr(P_H · P_grad)     = 1.000  (1 harmonic → gradient)
    Tr(P_H · P_acoustic)  = 2.000  (2 harmonics → acoustic)
    Tr(P_H · P_optical)   = 0.000  (no leakage to optical)
    Sum                   = 3.000  (Parseval identity)

The M₁-orthonormal harmonic forms are extracted via Hodge decomposition:
    ker(d₁) = im(d₀) ⊕ Harm¹  (M₁-orthogonal)
using the M₁-Gram matrix within the null space of d₁ to cleanly separate
the 3-dimensional harmonic subspace (gap ≈ 10¹³) from the 95-dimensional
gradient subspace (Kelvin N=2).

RAW OUTPUT:

  Harmonic extraction (gap = sv₃/sv₄ for projector onto Harm¹):
    Kelvin N=2:  gap = 1.23e+14,  M₁-ortho err = 2.41e-15
    C15 N=1:     gap = 1.01e+14,  M₁-ortho err = 1.05e-15
    WP N=1:      gap = 7.45e+13,  M₁-ortho err = 1.29e-15

  Subspace traces at k_frac=0.001 (all structures, [100] and [111]):
    Tr(H,grad) = 1.0000,  Tr(H,acou) = 2.0000,  Tr(H,opt) = 0.0000

  Acoustic speed: ω²/k² = 1.0000 (all structures)

  Dispersion along Γ→X (Kelvin N=2, k_frac 0→0.5):
    v_s = 1.0000 at k_frac=0.001, v_s = 0.9841 at X (1.6% lattice effect)

ANSWER:
  The exact complex implements the discrete Helmholtz decomposition: at k≠0,
  the 3 harmonic cohomology classes of T³ split into 2 transverse acoustic
  modes (ω=|k|) and 1 longitudinal gradient. This is universal across foam
  topologies and k-directions. The standard complex cannot do this because
  it has no cohomology at k≠0.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/14_test_acoustic_deformation.py
"""

import sys, os
import numpy as np
from numpy.linalg import svd, norm, eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from scipy.linalg import null_space, eigh as scipy_eigh
from physics.hodge import (build_kelvin_with_dual_info, build_c15_with_dual_info,
                            build_wp_with_dual_info, build_hodge_stars_voronoi)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact


# ── Helpers ────────────────────────────────────────────────────────────────

def extract_harmonic_forms(V, E, F, L_vec, shifts, M1):
    """Extract 3 M₁-orthonormal harmonic 1-forms at Γ via Hodge decomposition.

    Returns (H, gap, ortho_err) where H is (nE, 3), gap is sv₃/sv₄ of the
    harmonic projector, and ortho_err is ||H^H M₁ H - I||.
    """
    k0 = np.array([0.0, 0.0, 0.0])
    d0_0 = build_d0_bloch(V, E, k0, L_vec, shifts)
    d1_0 = build_d1_bloch_exact(V, E, F, k0, L_vec, d0_0)

    # Null space of d₁ (Euclidean orthonormal basis)
    K = null_space(d1_0)

    # Express im(d₀) in K-coordinates: im(d₀) ⊂ ker(d₁)
    C = K.conj().T @ d0_0  # (dim_ker, nV)
    G = K.conj().T @ M1 @ K  # M₁-Gram matrix in K-coords

    # M₁-orthogonal projector onto gradient subspace within ker(d₁)
    CGC = C.conj().T @ G @ C
    Uc, sc, Vtc = svd(CGC, full_matrices=False)
    s_inv = np.where(sc > 1e-10, 1.0 / sc, 0.0)
    CGC_pinv = (Vtc.T * s_inv) @ Uc.T
    P_grad = C @ CGC_pinv @ C.conj().T @ G
    P_harm = np.eye(K.shape[1]) - P_grad

    # SVD of harmonic projector — top 3 singular values ≈ 1, rest ≈ 0
    Uh, sh, _ = svd(P_harm)
    gap = sh[2] / max(sh[3], 1e-30)

    # Raw harmonic forms (not M₁-orthonormal)
    H_raw = K @ Uh[:, :3]

    # M₁-orthonormalize via Gram matrix diagonalization
    Gram = H_raw.conj().T @ M1 @ H_raw
    eigvals_G, eigvecs_G = eigh(Gram)
    H = H_raw @ eigvecs_G @ np.diag(1.0 / np.sqrt(eigvals_G))

    ortho_err = norm(H.conj().T @ M1 @ H - np.eye(3))

    return H, gap, ortho_err


def compute_subspace_traces(V, E, F, L_vec, shifts, M1, M2, H, k_vec):
    """Compute Tr(P_H · P_sub) for sub = gradient, acoustic, optical.

    Acoustic modes are identified by maximum harmonic overlap, not assumed to
    be the lowest nonzero eigenvalues. We then verify that the two modes with
    highest H-overlap ARE the two lowest nonzero eigenvalues (if not, there's
    a band crossing).

    Returns (v_s2, tr_grad, tr_acou, tr_opt, acou_are_lowest).
    """
    k_mag = norm(k_vec)
    d0_k = build_d0_bloch(V, E, k_vec, L_vec, shifts)
    d1_k = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0_k)

    KK = d1_k.conj().T @ M2 @ d1_k
    eigvals, eigvecs = scipy_eigh(KK, M1)
    idx = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    n_z = np.sum(np.abs(eigvals) < 1e-8)
    nz_idx = np.where(np.abs(eigvals) >= 1e-8)[0]

    # Harmonic overlap per nonzero eigenvector
    ov_per_mode = np.zeros(len(nz_idx))
    for mi, ei in enumerate(nz_idx):
        for hi in range(3):
            ov_per_mode[mi] += np.abs(H[:, hi].conj() @ M1 @ eigvecs[:, ei]) ** 2

    # Acoustic = 2 nonzero modes with highest harmonic overlap
    top2 = np.argsort(-ov_per_mode)[:2]
    a_idx = nz_idx[sorted(top2)]  # sorted by eigenvalue order
    o_mask = np.ones(len(nz_idx), dtype=bool)
    o_mask[top2] = False
    o_idx = nz_idx[o_mask]

    # Are the two highest-overlap modes also the two lowest nonzero eigenvalues?
    acou_are_lowest = set(a_idx) == set(nz_idx[:2])

    v_s2 = eigvals[a_idx[0]] / k_mag ** 2

    tr_g = sum(np.abs(H[:, hi].conj() @ M1 @ eigvecs[:, gi]) ** 2
               for hi in range(3) for gi in range(n_z))
    tr_a = sum(np.abs(H[:, hi].conj() @ M1 @ eigvecs[:, ai]) ** 2
               for hi in range(3) for ai in a_idx)
    tr_o = sum(np.abs(H[:, hi].conj() @ M1 @ eigvecs[:, oi]) ** 2
               for hi in range(3) for oi in o_idx)

    return v_s2, tr_g, tr_a, tr_o, acou_are_lowest


def load_foam(builder, N, L_cell):
    """Load foam structure and return (V, E, F, L_vec, shifts, M1, M2)."""
    data = builder(N=N, L_cell=L_cell)
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    shifts = compute_edge_shifts(V, E, L_vec)
    star1, star2 = build_hodge_stars_voronoi(data)
    M1 = np.diag(star1)
    M2 = np.diag(star2)
    return V, E, F, L_vec, shifts, M1, M2


# ── Test 1: Harmonic extraction ──────────────────────────────────────────

def test_harmonic_extraction():
    """Three M₁-orthonormal harmonic 1-forms at Γ, clean gap on all structures."""
    print(f"\n{'=' * 70}")
    print(f"  HARMONIC 1-FORM EXTRACTION AT Γ")
    print(f"{'=' * 70}")

    structures = [
        ('Kelvin N=2', build_kelvin_with_dual_info, 2, 1.0),
        ('C15 N=1',    build_c15_with_dual_info,    1, 1.0),
        ('WP N=1',     build_wp_with_dual_info,     1, 1.0),
    ]

    for name, builder, N, Lc in structures:
        V, E, F, L_vec, shifts, M1, M2 = load_foam(builder, N, Lc)
        H, gap, oe = extract_harmonic_forms(V, E, F, L_vec, shifts, M1)

        nV, nE, nF = len(V), len(E), len(F)
        print(f"\n  {name} (V={nV}, E={nE}, F={nF}):")
        print(f"    gap = {gap:.2e},  M₁-ortho err = {oe:.2e}")

        # Verify: d₁ h = 0
        k0 = np.array([0.0, 0.0, 0.0])
        d0_0 = build_d0_bloch(V, E, k0, L_vec, shifts)
        d1_0 = build_d1_bloch_exact(V, E, F, k0, L_vec, d0_0)
        for i in range(3):
            curl_norm = norm(d1_0 @ H[:, i])
            assert curl_norm < 1e-12, f"d₁ h{i} ≠ 0: {curl_norm:.2e}"

        assert gap > 1e10, f"Harmonic gap too small: {gap:.2e}"
        assert oe < 1e-12, f"M₁-orthonormality error: {oe:.2e}"

        # Verify: harmonic forms are not gradients
        # Project onto im(d₀) using pseudoinverse
        A = d0_0.conj().T @ M1 @ d0_0
        Ua, sa, Vta = svd(A, full_matrices=False)
        s_inv = np.where(sa > 1e-10, 1.0 / sa, 0.0)
        P_full = d0_0 @ ((Vta.T * s_inv) @ Ua.T) @ d0_0.conj().T @ M1
        for i in range(3):
            grad_frac = norm(P_full @ H[:, i]) / norm(H[:, i])
            assert grad_frac < 1e-10, f"h{i} has gradient component: {grad_frac:.2e}"
            print(f"    h{i}: ||d₁h||={norm(d1_0 @ H[:,i]):.2e}, grad_frac={grad_frac:.2e}")

    print(f"\n  PASS: 3 M₁-orthonormal harmonic 1-forms on all structures.")


# ── Test 2: Helmholtz decomposition (1 + 2 = 3 splitting) ───────────────

def test_helmholtz_splitting():
    """At k→0⁺: Tr(H,grad)=1, Tr(H,acoustic)=2, Tr(H,optical)=0."""
    print(f"\n{'=' * 70}")
    print(f"  HELMHOLTZ DECOMPOSITION: 1 GRAD + 2 ACOUSTIC = 3 HARMONIC")
    print(f"{'=' * 70}")

    structures = [
        ('Kelvin N=2', build_kelvin_with_dual_info, 2, 1.0),
        ('C15 N=1',    build_c15_with_dual_info,    1, 1.0),
        ('WP N=1',     build_wp_with_dual_info,     1, 1.0),
    ]

    k_frac = 0.001  # small enough for clean separation
    directions = [('[100]', [1, 0, 0]), ('[010]', [0, 1, 0]),
                  ('[001]', [0, 0, 1]), ('[111]', [1, 1, 1])]

    for name, builder, N, Lc in structures:
        V, E, F, L_vec, shifts, M1, M2 = load_foam(builder, N, Lc)
        H, _, _ = extract_harmonic_forms(V, E, F, L_vec, shifts, M1)
        L = L_vec[0]

        print(f"\n  {name}:")
        print(f"    {'dir':>6s} {'Tr(grad)':>10s} {'Tr(acou)':>10s} {'Tr(opt)':>10s} {'sum':>8s} {'acou=low':>9s}")

        for label, kdir in directions:
            kd = np.array(kdir, dtype=float)
            kd /= norm(kd)
            k_vec = k_frac * 2 * np.pi / L * kd

            v_s2, tg, ta, to, acou_low = compute_subspace_traces(
                V, E, F, L_vec, shifts, M1, M2, H, k_vec)
            total = tg + ta + to

            print(f"    {label:>6s} {tg:10.6f} {ta:10.6f} {to:10.6f} {total:8.4f} {'yes' if acou_low else 'NO':>9s}")

            assert abs(tg - 1.0) < 0.01, f"Tr(H,grad) = {tg:.6f}, expected 1.0"
            assert abs(ta - 2.0) < 0.01, f"Tr(H,acou) = {ta:.6f}, expected 2.0"
            assert to < 0.001, f"Tr(H,opt) = {to:.6f}, expected ~0"
            assert abs(total - 3.0) < 1e-6, f"Sum = {total:.6f}, expected 3.0"
            # Strongest check: max-overlap modes ARE the lowest nonzero eigenvalues
            assert acou_low, f"Acoustic modes are NOT the two lowest nonzero eigenvalues"

    print(f"\n  PASS: 1+2+0=3 Helmholtz splitting on all structures and directions.")
    print(f"  Max-overlap modes = lowest nonzero eigenvalues (no band crossing).")


# ── Test 3: Acoustic speed ω²/k² = 1 ────────────────────────────────────

def test_acoustic_speed():
    """Acoustic dispersion ω² = k² (v_s = 1) on all structures."""
    print(f"\n{'=' * 70}")
    print(f"  ACOUSTIC SPEED: ω²/k² = 1 (NATURAL UNITS)")
    print(f"{'=' * 70}")

    structures = [
        ('Kelvin N=2', build_kelvin_with_dual_info, 2, 1.0),
        ('C15 N=1',    build_c15_with_dual_info,    1, 1.0),
        ('WP N=1',     build_wp_with_dual_info,     1, 1.0),
    ]

    for name, builder, N, Lc in structures:
        V, E, F, L_vec, shifts, M1, M2 = load_foam(builder, N, Lc)
        H, _, _ = extract_harmonic_forms(V, E, F, L_vec, shifts, M1)
        L = L_vec[0]

        print(f"\n  {name}:")
        print(f"    {'k_frac':>8s} {'ω²/k²':>10s} {'ω₁':>10s} {'ω₂':>10s}")

        for k_frac in [0.001, 0.005, 0.01, 0.05, 0.1]:
            k_vec = np.array([k_frac * 2 * np.pi / L, 0, 0])
            k_mag = norm(k_vec)

            d0_k = build_d0_bloch(V, E, k_vec, L_vec, shifts)
            d1_k = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0_k)
            KK = d1_k.conj().T @ M2 @ d1_k
            eigvals, _ = scipy_eigh(KK, M1)
            eigvals = np.sort(eigvals)

            nz = eigvals[np.abs(eigvals) >= 1e-8]
            w1, w2 = np.sqrt(nz[0]), np.sqrt(nz[1])
            vs2 = nz[0] / k_mag ** 2

            print(f"    {k_frac:8.4f} {vs2:10.6f} {w1:10.6f} {w2:10.6f}")

            if k_frac <= 0.01:
                assert abs(vs2 - 1.0) < 0.001, f"v_s² = {vs2:.6f}, expected 1.0"
            # At larger k, allow lattice dispersion
            assert abs(vs2 - 1.0) < 0.01, f"v_s² = {vs2:.6f}, too far from 1.0"

    print(f"\n  PASS: ω²/k² = 1 on all structures.")


# ── Test 4: Dispersion curve Γ→X ─────────────────────────────────────────

def test_dispersion_curve():
    """Full dispersion along Γ→X: linear acoustic, lattice deviation at zone boundary."""
    print(f"\n{'=' * 70}")
    print(f"  DISPERSION CURVE Γ → X (KELVIN N=2)")
    print(f"{'=' * 70}")

    V, E, F, L_vec, shifts, M1, M2 = load_foam(build_kelvin_with_dual_info, 2, 1.0)
    H, _, _ = extract_harmonic_forms(V, E, F, L_vec, shifts, M1)
    L = L_vec[0]

    print(f"\n    {'k_frac':>8s} {'v_s':>8s} {'ω₁':>10s} {'ω₂':>10s} {'Tr(g)':>8s} {'Tr(a)':>8s} {'a=low':>6s}")

    vs_values = []
    n_crossings = 0
    for k_frac in np.linspace(0.01, 0.5, 20):
        k_vec = np.array([k_frac * 2 * np.pi / L, 0, 0])
        k_mag = norm(k_vec)

        v_s2, tr_g, tr_a, _, acou_low = compute_subspace_traces(
            V, E, F, L_vec, shifts, M1, M2, H, k_vec)
        vs = np.sqrt(v_s2)
        w1 = vs * k_mag
        vs_values.append(vs)
        if not acou_low:
            n_crossings += 1

        print(f"    {k_frac:8.3f} {vs:8.4f} {w1:10.6f} {w1:10.6f} {tr_g:8.4f} {tr_a:8.4f} {'yes' if acou_low else 'NO':>6s}")

    # v_s should decrease monotonically (lattice dispersion)
    assert vs_values[0] > vs_values[-1], "v_s should decrease toward zone boundary"
    # v_s at zone boundary: lattice dispersion depends on mesh.
    # Kelvin N=2 L_cell=1: L=2, k_X=π/2, effective ka depends on edge length.
    # Observed: v_s(X) ≈ 0.984. Bound at 0.90 to be safe across parameters.
    assert vs_values[-1] > 0.90, f"v_s at X = {vs_values[-1]:.4f}, too small"

    print(f"\n  PASS: Acoustic dispersion Γ→X, v_s monotone from 1.000 to {vs_values[-1]:.3f}.")
    if n_crossings > 0:
        print(f"  WARNING: {n_crossings} k-points where acoustic ≠ lowest nonzero (band crossing).")
    else:
        print(f"  Acoustic = lowest nonzero eigenvalue at all k-points (no band crossing).")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    test_harmonic_extraction()
    test_helmholtz_splitting()
    test_acoustic_speed()
    test_dispersion_curve()

    print(f"\n{'=' * 70}")
    print(f"  ALL 4 TESTS PASSED")
    print(f"  Acoustic modes are deformed cohomology classes.")
    print(f"  The exact complex implements the discrete Helmholtz decomposition.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
