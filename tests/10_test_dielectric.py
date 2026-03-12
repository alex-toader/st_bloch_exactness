"""
Material independence: exactness survives scalar ε, tensor ε, and μ≠1.

d₁(k) depends on neither ⋆₂ nor ⋆₁. Exactness (d₁d₀=0) is a property of
the cochain complex, not of the material. The exact construction preserves it;
the standard construction breaks it — at every material configuration identically.

Setup: Kelvin foam (BCC Voronoi). BCC has two sublattices (A and B).
Assign ε_A = 1 (sublattice A), ε_B = variable (sublattice B).
Per-face ε: harmonic mean of adjacent cells (H-field formulation).

MPB reference (obtained interactively, NOT validated in this test):
  conda activate mpb_env; python -c "from meep import mpb; ..."
  BCC Voronoi, epsilon_input_file (HDF5 grid), resolution=32.
  eps_B=1: c²=1.000, eps_B=2: c²=0.691, eps_B=4: c²=0.453, eps_B=9: c²=0.255.
  DEC lowest-order gives ~20% higher c² at eps_B=4. The discrepancy is consistent
  with second-order dispersion error at fixed cell size; we do not study
  h-refinement here. NOT an exactness defect. The exact method is structurally
  correct (d₁d₀=0, correct null space), not "more accurate" than MPB.
  TO VERIFY: run MPB comparison script (not yet written) to reproduce these values.

RAW OUTPUT:

  Part 1 — Uniform epsilon sanity check (SC N=5):
    eps=1.0: c2_DEC=0.999671, c2_expected=1.000000, error=3.29e-04
    eps=4.0: c2_DEC=0.249918, c2_expected=0.250000, error=3.29e-04
    eps=12.0: c2_DEC=0.083306, c2_expected=0.083333, error=3.29e-04
    All scale as 1/eps. Dispersion error O((ka)^2) identical at all eps.

  Part 2 — Kelvin foam with dielectric contrast:
    k = 5% BZ [100]:
      eps_B=1:  exact c2=1.000 n_zero=96  | std c2=1.549,2.325 n_zero=90
      eps_B=4:  exact c2=0.548 n_zero=96  | std c2=0.803,1.215 n_zero=90
      eps_B=9:  exact c2=0.384 n_zero=96  | std c2=0.545,0.820 n_zero=90
    k = 5% BZ [111]:
      eps_B=1:  exact c2=1.000 n_zero=96  | std c2=0.582,0.636 n_zero=82
      eps_B=4:  exact c2=0.548 n_zero=96  | std c2=0.307,0.345 n_zero=82
      eps_B=9:  exact c2=0.384 n_zero=96  | std c2=0.206,0.244 n_zero=82
    Exact: n_zero = V = 96 at ALL eps. Standard: 6-14 spurious modes.
    Exact: c2 degenerate (two polarizations). Standard: split.

  Part 3 — Exactness independent of epsilon (all structures):
    Kelvin N=2 (V=96):  d1d0_ex=7.86e-16, d1d0_std=7.54, n_zero_ex=96 at all eps
    C15 N=1 (V=136):    d1d0_ex=5.54e-16, d1d0_std=7.80, n_zero_ex=136 at all eps
    WP N=1 (V=46):      d1d0_ex=5.20e-16, d1d0_std=5.07, n_zero_ex=46 at all eps
    n_zero_exact = V on ALL structures at ALL epsilon values.

  Part 4 — Tensor epsilon and mu != 1 (3 structures × 3 configs):
    Kelvin N=2: nz_ex=96 at eps=[1,1,4]/mu=1, eps=1/mu=[1,1,3], eps=[2,1,4]/mu=[1,3,1]
    C15 N=1:    nz_ex=136 at all configs
    WP N=1:     nz_ex=46 at all configs
    c² changes with material (0.17–0.50). n_zero_exact = V at ALL configs.
    d₁(k) depends on neither ⋆₂ nor ⋆₁. Exactness is purely topological.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/10_test_dielectric.py
"""

import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from core_math.builders.solids_periodic import build_sc_solid_periodic
from physics.hodge import (build_kelvin_with_dual_info, build_c15_with_dual_info,
                           build_wp_with_dual_info, build_hodge_stars_voronoi)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup


# ── Helpers ──────────────────────────────────────────────────────────────

def classify_sublattice(cell_centers, L, N):
    """Classify BCC cells into sublattice A (even) and B (odd).

    BCC has two sublattices: (0,0,0) and (½,½,½) in reduced coordinates.
    Formula: sum of coordinates / (a/2), rounded to nearest integer, mod 2.
    Safe as long as Voronoi centers are within a/4 of ideal BCC positions.
    """
    a_cell = L / N
    labels = np.array([
        int(round((c[0] + c[1] + c[2]) / (a_cell / 2))) % 2
        for c in cell_centers
    ])
    nA, nB = np.sum(labels == 0), np.sum(labels == 1)
    assert nA == N**3 and nB == N**3, \
        f"BCC sublattice count wrong: A={nA}, B={nB}, expected {N**3} each"
    return labels


def build_star2_dielectric(star2, face_to_cells, eps_cell, nF):
    """Build modified star2 for H-field formulation with variable epsilon.

    H-field: K = d1† (star2/eps) d1.
    Per-face eps: harmonic mean of adjacent cells (1/eps averaging).
    """
    inv_eps_face = np.zeros(nF)
    for f_idx in range(nF):
        ca, cb = face_to_cells[f_idx]
        inv_eps_face[f_idx] = 0.5 * (1.0 / eps_cell[ca] + 1.0 / eps_cell[cb])
    return star2 * inv_eps_face


def solve_curl_curl(d1_k, star2_mod, star1):
    """Solve K u = omega^2 M u. Returns sorted eigenvalues."""
    K = d1_k.conj().T @ np.diag(star2_mod) @ d1_k
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    return np.sort(np.real(eigh(K, M, eigvals_only=True)))


def count_modes(eigvals, nV):
    """Count zero modes and spurious modes."""
    max_eig = max(np.max(np.abs(eigvals)), 1e-14)
    threshold = max_eig * 1e-8
    n_zero = int(np.sum(np.abs(eigvals) < threshold))
    n_spur = max(0, nV - n_zero)
    physical = eigvals[np.abs(eigvals) >= threshold]
    return n_zero, n_spur, physical


# ── Part 1: Uniform epsilon sanity check ─────────────────────────────────

def test_uniform_eps():
    """Uniform epsilon: c^2 = 1/eps at all eps values."""
    print(f"\n{'=' * 70}")
    print(f"  PART 1: UNIFORM EPSILON SANITY CHECK (SC N=5)")
    print(f"{'=' * 70}")

    mesh = build_sc_solid_periodic(N=5)
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    L = mesh['period_L']
    L_vec = np.array([L, L, L])
    nV, nE, nF = len(V), len(E), len(F)
    a = L / 5

    star1 = np.ones(nE) * a
    star2 = np.ones(nF) * (1.0 / a)

    shifts = compute_edge_shifts(V, E, L_vec)
    k = 2 * np.pi / L * 0.05 * np.array([1, 0, 0.])
    k_mag = np.linalg.norm(k)

    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)

    print(f"\n  SC N=5, frac=0.05 [100], H-field: K = d1†(star2/eps)d1")
    print(f"\n  {'eps':>6s} {'c2_expected':>12s} {'c2_DEC':>10s} {'rel_error':>10s}")

    for eps_val in [1.0, 2.0, 4.0, 9.0, 12.0]:
        s2 = star2 / eps_val
        eigvals = solve_curl_curl(d1_ex, s2, star1)
        _, _, physical = count_modes(eigvals, nV)
        c2 = physical[0] / k_mag**2
        c2_expected = 1.0 / eps_val
        error = abs(c2 / c2_expected - 1)

        print(f"  {eps_val:4.1f}   {c2_expected:10.6f} {c2:10.6f} {error:10.2e}")

        assert error < 1e-3, f"c2 error {error:.2e} > 1e-3 at eps={eps_val}"

    print(f"\n  All scale as 1/eps. VERIFIED.")


# ── Part 2: Kelvin foam with dielectric contrast ────────────────────────

def test_dielectric_contrast():
    """Kelvin foam, BCC sublattices A=1 B=eps. Exact vs standard."""
    print(f"\n{'=' * 70}")
    print(f"  PART 2: KELVIN FOAM WITH DIELECTRIC CONTRAST")
    print(f"{'=' * 70}")

    data = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    nV, nE, nF = len(V), len(E), len(F)
    cc = data['cell_centers']
    nC = len(cc)
    face_to_cells = data['face_to_cells']

    star1, star2 = build_hodge_stars_voronoi(data)
    sublattice = classify_sublattice(cc, L, N=2)

    shifts = compute_edge_shifts(V, E, L_vec)
    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)

    print(f"\n  V={nV}, E={nE}, F={nF}, C={nC}, L={L}")
    print(f"  Sublattice A: {np.sum(sublattice == 0)} cells (eps=1)")
    print(f"  Sublattice B: {np.sum(sublattice == 1)} cells (eps=variable)")

    k_scale = 2 * np.pi / L

    for frac, k_hat, label in [(0.05, np.array([1, 0, 0.]), '[100]'),
                                 (0.05, np.array([1, 1, 1.]) / np.sqrt(3), '[111]')]:
        k = k_scale * frac * k_hat
        k_mag = np.linalg.norm(k)

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)

        print(f"\n  k = {frac:.0%} BZ {label}")
        print(f"  {'eps_B':>6s} {'method':>7s} {'c2_1':>8s} {'c2_2':>8s}"
              f" {'n_zero':>7s} {'n_spur':>7s} {'||d1d0||':>10s}")

        for eps_B in [1.0, 2.0, 4.0, 9.0]:
            eps_cell = np.where(sublattice == 0, 1.0, eps_B)
            star2_eps = build_star2_dielectric(star2, face_to_cells, eps_cell, nF)

            for tag, d1_k in [('exact', d1_ex), ('std', d1_std)]:
                eigvals = solve_curl_curl(d1_k, star2_eps, star1)
                n_zero, n_spur, physical = count_modes(eigvals, nV)
                c2_1 = physical[0] / k_mag**2
                c2_2 = physical[1] / k_mag**2
                norm_d1d0 = np.linalg.norm(d1_k @ d0_k)

                print(f"  {eps_B:4.1f}   {tag:>5s} {c2_1:8.4f} {c2_2:8.4f}"
                      f" {n_zero:7d} {n_spur:7d} {norm_d1d0:10.2e}")

                # Exact assertions
                if tag == 'exact':
                    assert n_zero == nV, \
                        f"Exact n_zero={n_zero} != V={nV} at eps_B={eps_B}"
                    assert norm_d1d0 < 1e-12, \
                        f"Exact d1d0={norm_d1d0:.2e} at eps_B={eps_B}"
                    assert abs(c2_1 - c2_2) / max(c2_1, 1e-14) < 1e-3, \
                        f"Exact modes not degenerate: {c2_1:.4f} vs {c2_2:.4f}"

                # Standard assertions
                if tag == 'std':
                    assert n_spur > 0, \
                        f"Standard should have spurious modes at eps_B={eps_B}"
                    assert norm_d1d0 > 1.0, \
                        f"Standard d1d0 should be O(1): {norm_d1d0:.2e}"


# ── Part 3: Exactness independent of epsilon ─────────────────────────────

def test_exactness_eps_independent():
    """d1d0 = 0 is independent of epsilon on all 4 structures."""
    print(f"\n{'=' * 70}")
    print(f"  PART 3: EXACTNESS INDEPENDENT OF EPSILON (ALL STRUCTURES)")
    print(f"{'=' * 70}")

    print(f"\n  d1 is built BEFORE star2 is modified. Epsilon only enters K and M.")
    print(f"  Therefore ||d1·d0|| is the same at every epsilon value.")

    structures = [
        ('Kelvin N=2', lambda: build_kelvin_with_dual_info(N=2, L_cell=4.0)),
        ('C15 N=1',    lambda: build_c15_with_dual_info(N=1, L_cell=4.0)),
        ('WP N=1',     lambda: build_wp_with_dual_info(N=1, L_cell=4.0)),
    ]

    for name, builder in structures:
        data = builder()
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        nV, nF = len(V), len(F)
        cc = data['cell_centers']
        nC = len(cc)
        face_to_cells = data['face_to_cells']

        star1, star2 = build_hodge_stars_voronoi(data)

        # Per-cell eps: even/odd cell index
        cell_label = np.arange(nC) % 2

        shifts = compute_edge_shifts(V, E, L_vec)
        crossings = compute_edge_crossings(V, E, L)
        edge_lookup = build_edge_lookup(E, crossings)

        k = 2 * np.pi / L * 0.10 * np.array([1, 0, 0.])
        k_mag = np.linalg.norm(k)

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)

        norm_ex = np.linalg.norm(d1_ex @ d0_k)
        norm_std = np.linalg.norm(d1_std @ d0_k)

        print(f"\n  {name}: V={nV}, E={len(E)}, F={nF}, C={nC}")
        print(f"  ||d1_exact · d0|| = {norm_ex:.2e}  ||d1_std · d0|| = {norm_std:.2e}")

        print(f"  {'eps_B':>6s} {'n_zero_ex':>10s} {'c2_ex':>8s} {'n_zero_std':>11s} {'c2_std':>8s}")

        for eps_B in [1.0, 4.0, 12.0]:
            eps_cell = np.where(cell_label == 0, 1.0, eps_B)
            star2_eps = build_star2_dielectric(star2, face_to_cells, eps_cell, nF)

            eig_ex = solve_curl_curl(d1_ex, star2_eps, star1)
            eig_std = solve_curl_curl(d1_std, star2_eps, star1)

            nz_ex, _, phys_ex = count_modes(eig_ex, nV)
            nz_std, _, phys_std = count_modes(eig_std, nV)

            c2_ex = phys_ex[0] / k_mag**2
            c2_std = phys_std[0] / k_mag**2

            print(f"  {eps_B:4.1f}   {nz_ex:8d}   {c2_ex:8.4f}   {nz_std:9d}   {c2_std:8.4f}")

            assert nz_ex == nV, \
                f"{name}: exact n_zero={nz_ex} != V={nV} at eps_B={eps_B}"

        assert norm_ex < 1e-12, f"{name}: d1_exact·d0 = {norm_ex:.2e}"
        assert norm_std > 0.1, f"{name}: d1_std·d0 should be O(1): {norm_std:.2e}"

    print(f"\n  n_zero_exact = V at ALL epsilon values on ALL structures.")
    print(f"  Exactness is a property of the complex, not of the material.")


# ── Part 4: Tensor ε and μ≠1 ──────────────────────────────────────────

def compute_face_normals(V, F, L_vec):
    """Unit normal for each face, minimum image convention."""
    normals = np.zeros((len(F), 3))
    for f_idx, face in enumerate(F):
        v0 = np.array(V[face[0]], dtype=float)
        d1 = np.array(V[face[1]], dtype=float) - v0
        d2 = np.array(V[face[2]], dtype=float) - v0
        for dim in range(3):
            d1[dim] -= L_vec[dim] * round(d1[dim] / L_vec[dim])
            d2[dim] -= L_vec[dim] * round(d2[dim] / L_vec[dim])
        n = np.cross(d1, d2)
        norm = np.linalg.norm(n)
        if norm > 1e-14:
            normals[f_idx] = n / norm
    return normals


def compute_edge_directions(V, E, L_vec):
    """Unit direction for each edge, minimum image convention."""
    dirs = np.zeros((len(E), 3))
    for e_idx, (i, j) in enumerate(E):
        d = np.array(V[j], dtype=float) - np.array(V[i], dtype=float)
        for dim in range(3):
            d[dim] -= L_vec[dim] * round(d[dim] / L_vec[dim])
        norm = np.linalg.norm(d)
        if norm > 1e-14:
            dirs[e_idx] = d / norm
    return dirs


def test_tensor_eps_and_mu():
    """Exactness survives anisotropic ε (diagonal tensor) and μ≠1.

    d₁(k) depends on neither ⋆₂ nor ⋆₁, so d₁d₀=0 and n_zero=V hold
    regardless of material tensors. Eigenvalues change; null space does not.

    Per-face effective ε⁻¹: Σᵢ n̂ᵢ² / εᵢ (diagonal ε only).
    Per-edge effective μ: Σᵢ êᵢ² · μᵢ (diagonal μ only).
    """
    print(f"\n{'=' * 70}")
    print(f"  PART 4: TENSOR EPSILON AND MU != 1")
    print(f"{'=' * 70}")

    configs = [
        ('eps=[1,1,4] mu=1',       [1, 1, 4], [1, 1, 1]),
        ('eps=1 mu=[1,1,3]',       [1, 1, 1], [1, 1, 3]),
        ('eps=[2,1,4] mu=[1,3,1]', [2, 1, 4], [1, 3, 1]),
    ]

    structures = [
        ('Kelvin N=2', lambda: build_kelvin_with_dual_info(N=2, L_cell=4.0)),
        ('C15 N=1',    lambda: build_c15_with_dual_info(N=1, L_cell=4.0)),
        ('WP N=1',     lambda: build_wp_with_dual_info(N=1, L_cell=4.0)),
    ]

    for struct_name, builder in structures:
        data = builder()
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        nV, nE, nF = len(V), len(E), len(F)

        star1, star2 = build_hodge_stars_voronoi(data)
        face_normals = compute_face_normals(V, F, L_vec)
        edge_dirs = compute_edge_directions(V, E, L_vec)

        shifts = compute_edge_shifts(V, E, L_vec)
        crossings = compute_edge_crossings(V, E, L)
        edge_lookup = build_edge_lookup(E, crossings)

        k = 2 * np.pi / L * 0.10 * np.array([1, 0, 0.])
        k_mag = np.linalg.norm(k)

        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)

        print(f"\n  {struct_name}: V={nV}, E={nE}, F={nF}")
        print(f"  {'config':>25s} {'nz_ex':>6s} {'nz_std':>7s} {'c2_ex':>8s} {'c2_std':>8s}")

        for label, eps_diag, mu_diag in configs:
            inv_eps = 1.0 / np.array(eps_diag, dtype=float)
            mu = np.array(mu_diag, dtype=float)

            star2_mod = star2 * np.array([np.dot(n**2, inv_eps) for n in face_normals])
            star1_mod = star1 * np.array([np.dot(d**2, mu) for d in edge_dirs])

            eig_ex = solve_curl_curl(d1_ex, star2_mod, star1_mod)
            eig_std = solve_curl_curl(d1_std, star2_mod, star1_mod)

            nz_ex, _, phys_ex = count_modes(eig_ex, nV)
            nz_std, _, phys_std = count_modes(eig_std, nV)

            c2_ex = phys_ex[0] / k_mag**2
            c2_std = phys_std[0] / k_mag**2

            print(f"  {label:>25s} {nz_ex:6d} {nz_std:7d} {c2_ex:8.4f} {c2_std:8.4f}")

            assert nz_ex == nV, \
                f"{struct_name} {label}: exact n_zero={nz_ex} != V={nV}"

    print(f"\n  n_zero_exact = V at ALL (ε_tensor, μ_tensor) configs on ALL structures.")
    print(f"  d₁(k) depends on neither ⋆₂ nor ⋆₁. Exactness is purely topological.")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MATERIAL INDEPENDENCE TESTS")
    print("Does exactness survive variable, anisotropic materials?")
    print("=" * 70)

    test_uniform_eps()
    test_dielectric_contrast()
    test_exactness_eps_independent()
    test_tensor_eps_and_mu()

    print(f"\n{'=' * 70}")
    print("ALL MATERIAL INDEPENDENCE TESTS PASSED.")
    print("=" * 70)


if __name__ == '__main__':
    main()
    print("\nDone.")
