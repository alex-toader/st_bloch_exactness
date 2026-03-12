"""
d₂(k) exactness test: the recurrence pattern generalizes to level 2→3.

On periodic complexes, the standard Bloch extension breaks d₂d₁ = 0
at k≠0, just as it breaks d₁d₀ = 0. The same recurrence construction
(propagate phases along shared (p-1)-cells) restores exactness.

Key finding: this is NOT specific to level 1→2. At level 2→3:
  - d₂_top · d₁_exact(k) ~ O(k)  (topological d₂ breaks exactness)
  - d₂_exact · d₁_exact(k) ~ 10⁻¹⁵  (recurrence restores it)
  - Per-face standard phase fails (faces need cell-dependent phases)

Recurrence formula (same pattern as d₁):
  d₂[c, f_next] = -d₂[c, f_curr] · d₁[f_curr, e] / d₁[f_next, e]
where e is the edge shared by f_curr and f_next in cell c's boundary.

Tested on: SC N=3, Kelvin N=2, C15 N=1, WP N=1.

RAW OUTPUT:

  SC N=3 (V=27, E=81, F=81, C=27):
    frac=0.10 [100]: d2d1_ex=1.18e-16, d2top·d1ex=3.71, n_incon=9/81
    frac=0.10 [111]: d2d1_ex=7.95e-16, d2top·d1ex=3.42, n_incon=27/81
    All unimodular. All asserts pass.

  Kelvin N=2 (V=96, E=192, F=112, C=16):
    frac=0.10 [100]: d2d1_ex=1.40e-15, d2top·d1ex=4.94, n_incon=28/112
    frac=0.10 [111]: d2d1_ex=2.00e-15, d2top·d1ex=5.02, n_incon=66/112
    All unimodular. All asserts pass.

  Universality (frac=0.10, [100]):
    SC N=3:    d1d0=8.98e-17, d2d1_ex=1.18e-16, d2top·d1=3.71
    Kelvin N=2: d1d0=7.86e-16, d2d1_ex=1.40e-15, d2top·d1=4.94
    C15 N=1:   d1d0=5.54e-16, d2d1_ex=1.53e-15, d2top·d1=5.93
    WP N=1:    d1d0=5.20e-16, d2d1_ex=1.13e-15, d2top·d1=4.46

  All 4 structures: d₂d₁_exact ~ 10⁻¹⁵, d₂_top·d₁ ~ O(1).
  Same recurrence, same fix, same failure mode on all structures.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/9_test_d2_exactness.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from core_math.builders.solids_periodic import build_sc_solid_periodic
from core_math.operators.incidence import build_d2
from physics.hodge import build_kelvin_with_dual_info, build_c15_with_dual_info, build_wp_with_dual_info
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch_complex import build_face_edge_map, build_cell_face_incidence


def build_d2_exact(cell_face_inc, face_edges, d1_ex, n_cells, n_faces):
    """Build exactness-preserving d₂(k) via recurrence on cell boundaries.

    For each cell c, traverse its boundary faces. At each edge shared by
    two faces of c, the constraint d₂d₁=0 determines the phase ratio:
        d₂[c, f_next] = -d₂[c, f_curr] · d₁[f_curr, e] / d₁[f_next, e]

    This is the level 2→3 analogue of the d₁ recurrence at level 1→2.

    Cell holonomy: BFS uses n_faces-1 edges. The remaining adjacency edges
    give closure conditions (cell holonomy = trivial). These are verified
    explicitly — if d₁ satisfies d₁d₀=0, all closure conditions hold.
    """
    d2 = np.zeros((n_cells, n_faces), dtype=complex)

    for c_idx in range(n_cells):
        faces_of_c = cell_face_inc[c_idx]
        if not faces_of_c:
            continue

        # Build face adjacency within this cell
        adj = {f: [] for f, _ in faces_of_c}
        for i, (f1, _) in enumerate(faces_of_c):
            for j, (f2, _) in enumerate(faces_of_c):
                if i >= j:
                    continue
                shared = face_edges[f1] & face_edges[f2]
                for e in shared:
                    adj[f1].append((f2, e))
                    adj[f2].append((f1, e))

        # BFS recurrence from seed face
        f0, o0 = faces_of_c[0]
        d2[c_idx, f0] = o0
        visited = {f0}
        queue = [f0]
        bfs_edges = set()  # track which (f_curr, f_next) pairs BFS used

        while queue:
            fc = queue.pop(0)
            for fn, es in adj[fc]:
                if fn in visited:
                    continue
                d1c = d1_ex[fc, es]
                d1n = d1_ex[fn, es]
                if abs(d1n) < 1e-14:
                    raise ValueError(f"d1[{fn},{es}] = 0, cannot propagate")
                d2[c_idx, fn] = -d2[c_idx, fc] * d1c / d1n
                visited.add(fn)
                queue.append(fn)
                bfs_edges.add((fc, fn, es))
                bfs_edges.add((fn, fc, es))

        assert len(visited) == len(faces_of_c), \
            f"Cell {c_idx}: visited {len(visited)}/{len(faces_of_c)} faces"

        # Cell holonomy check: verify non-BFS edges are consistent.
        # Each such edge gives a closure condition that must be satisfied.
        for fc in adj:
            for fn, es in adj[fc]:
                if (fc, fn, es) in bfs_edges:
                    continue
                if fc >= fn:
                    continue  # check each pair once
                d1c = d1_ex[fc, es]
                d1n = d1_ex[fn, es]
                if abs(d1n) < 1e-14:
                    continue
                expected = -d2[c_idx, fc] * d1c / d1n
                actual = d2[c_idx, fn]
                assert abs(expected - actual) < 1e-10, \
                    f"Cell {c_idx}: holonomy failure on non-BFS edge " \
                    f"({fc},{fn},e={es}): expected {expected:.6f}, got {actual:.6f}"

    return d2


def build_d2_top_from_foam(data):
    """Build topological d₂ from foam data (face_to_cells).

    Determines face orientations relative to cells using face normals
    and cell center positions. Validates at k=0 that d₂·d₁ = 0.

    Returns:
        d2_top: (C, F) real matrix
        cfi: cell_face_incidence list
    """
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    face_to_cells = data['face_to_cells']
    cell_centers = data['cell_centers']
    nF = len(F)
    nC = len(cell_centers)

    # Initial guess: d₂[cell_a, f] = +1, d₂[cell_b, f] = -1
    d2_top = np.zeros((nC, nF))
    for f_idx in range(nF):
        ca, cb = face_to_cells[f_idx]
        d2_top[ca, f_idx] = +1
        d2_top[cb, f_idx] = -1

    # Check at k=0 (imports already at top level)
    shifts = compute_edge_shifts(V, E, L_vec)
    k0 = np.array([0., 0., 0.])
    d0_k0 = build_d0_bloch(V, E, k0, L_vec, shifts)
    d1_k0 = build_d1_bloch_exact(V, E, F, k0, L_vec, d0_k0)

    if np.linalg.norm(d2_top @ d1_k0) > 1e-10:
        # Fix orientations using face normals
        for f_idx in range(nF):
            face = F[f_idx]
            verts = np.array([V[v] for v in face[:3]])
            normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal = normal / np.linalg.norm(normal)

            ca, cb = face_to_cells[f_idx]
            centroid = np.mean(np.array([V[v] for v in face]), axis=0)
            dir_a = cell_centers[ca] - centroid

            if np.dot(normal, dir_a) > 0:
                d2_top[ca, f_idx] = -1
                d2_top[cb, f_idx] = +1
            else:
                d2_top[ca, f_idx] = +1
                d2_top[cb, f_idx] = -1

    assert np.linalg.norm(d2_top @ d1_k0) < 1e-10, \
        f"d2_top @ d1 != 0 at k=0 after orientation fix"

    # Build cell_face_incidence
    cfi = [[] for _ in range(nC)]
    for c in range(nC):
        for f in range(nF):
            if abs(d2_top[c, f]) > 0.5:
                cfi[c].append((f, int(d2_top[c, f])))

    return d2_top, cfi


# ── Part 1: SC N=3 ──────────────────────────────────────────────────────

def test_sc():
    """d₂ exactness on SC cubic N=3."""
    print(f"\n{'=' * 70}")
    print(f"  d₂ EXACTNESS — SC CUBIC N=3")
    print(f"{'=' * 70}")

    mesh = build_sc_solid_periodic(N=3)
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    L = mesh['period_L']
    L_vec = np.array([L, L, L])
    cfi = mesh['cell_face_incidence']
    nV, nE, nF, nC = len(V), len(E), len(F), mesh['n_cells']

    print(f"\n  V={nV}, E={nE}, F={nF}, C={nC}, L={L}")

    # Topological d2
    d2_top = build_d2(cfi, nF)

    # Edge infrastructure
    shifts = compute_edge_shifts(V, E, L_vec)
    face_edges = build_face_edge_map(F, E)
    k_scale = 2 * np.pi / L

    # Verify at k=0
    k0 = np.array([0., 0., 0.])
    d0_k0 = build_d0_bloch(V, E, k0, L_vec, shifts)
    d1_k0 = build_d1_bloch_exact(V, E, F, k0, L_vec, d0_k0)
    assert np.linalg.norm(d2_top @ d1_k0) < 1e-12, "d2_top @ d1 != 0 at k=0"

    print(f"\n  {'frac':>6s} {'dir':>6s} {'||d2d1_ex||':>12s} {'||d2top·d1ex||':>15s}"
          f" {'n_incon':>8s} {'unimod':>7s}")

    for frac in [0.05, 0.10, 0.20]:
        for k_hat, label in [(np.array([1, 0, 0.]), '[100]'),
                              (np.array([1, 1, 1.]) / np.sqrt(3), '[111]')]:
            k = k_scale * frac * k_hat
            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)

            # d2 exact via recurrence
            d2_ex = build_d2_exact(cfi, face_edges, d1_ex, nC, nF)

            norm_d2d1 = np.linalg.norm(d2_ex @ d1_ex)
            norm_top = np.linalg.norm(d2_top @ d1_ex)
            unimod = np.allclose(np.abs(d2_ex[d2_ex != 0]), 1.0)

            # Per-face consistency check
            ratio = np.zeros((nC, nF), dtype=complex)
            for c in range(nC):
                for f in range(nF):
                    if abs(d2_top[c, f]) > 0.5:
                        ratio[c, f] = d2_ex[c, f] / d2_top[c, f]
            n_incon = 0
            for f in range(nF):
                nz = ratio[:, f][np.abs(ratio[:, f]) > 0.5]
                if len(nz) == 2 and not np.allclose(nz[0], nz[1]):
                    n_incon += 1

            print(f"  {frac:6.2f} {label:>6s} {norm_d2d1:12.2e} {norm_top:15.2e}"
                  f" {n_incon:8d} {str(unimod):>7s}")

            assert norm_d2d1 < 1e-12, f"d2_exact @ d1_exact != 0: {norm_d2d1:.2e}"
            assert norm_top > 0.1, f"d2_top should fail at k!=0: {norm_top:.2e}"
            assert unimod, "d2_exact entries should be unimodular"
            if label == '[100]':
                assert n_incon > 0, "Should have inconsistent faces for [100]"


# ── Part 2: Kelvin N=2 ──────────────────────────────────────────────────

def test_kelvin():
    """d₂ exactness on Kelvin foam N=2."""
    print(f"\n{'=' * 70}")
    print(f"  d₂ EXACTNESS — KELVIN FOAM N=2")
    print(f"{'=' * 70}")

    data = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    nV, nE, nF = len(V), len(E), len(F)
    nC = len(data['cell_centers'])

    print(f"\n  V={nV}, E={nE}, F={nF}, C={nC}, L={L}")

    # Build d2_top with orientation fix
    d2_top, cfi = build_d2_top_from_foam(data)

    faces_per_cell = [len(cfi[c]) for c in range(nC)]
    print(f"  Faces per cell: {faces_per_cell[0]} (truncated octahedra: 14)")

    # Edge infrastructure
    shifts = compute_edge_shifts(V, E, L_vec)
    face_edges = build_face_edge_map(F, E)
    k_scale = 2 * np.pi / L

    print(f"\n  {'frac':>6s} {'dir':>6s} {'||d2d1_ex||':>12s} {'||d2top·d1ex||':>15s}"
          f" {'n_incon':>8s} {'unimod':>7s}")

    for frac in [0.05, 0.10, 0.20]:
        for k_hat, label in [(np.array([1, 0, 0.]), '[100]'),
                              (np.array([1, 1, 1.]) / np.sqrt(3), '[111]')]:
            k = k_scale * frac * k_hat
            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)

            d2_ex = build_d2_exact(cfi, face_edges, d1_ex, nC, nF)

            norm_d2d1 = np.linalg.norm(d2_ex @ d1_ex)
            norm_top = np.linalg.norm(d2_top @ d1_ex)
            unimod = np.allclose(np.abs(d2_ex[d2_ex != 0]), 1.0)

            ratio = np.zeros((nC, nF), dtype=complex)
            for c in range(nC):
                for f in range(nF):
                    if abs(d2_top[c, f]) > 0.5:
                        ratio[c, f] = d2_ex[c, f] / d2_top[c, f]
            n_incon = 0
            for f in range(nF):
                nz = ratio[:, f][np.abs(ratio[:, f]) > 0.5]
                if len(nz) == 2 and not np.allclose(nz[0], nz[1]):
                    n_incon += 1

            print(f"  {frac:6.2f} {label:>6s} {norm_d2d1:12.2e} {norm_top:15.2e}"
                  f" {n_incon:8d} {str(unimod):>7s}")

            assert norm_d2d1 < 1e-12, f"d2_exact @ d1_exact != 0: {norm_d2d1:.2e}"
            assert norm_top > 0.1, f"d2_top should fail at k!=0: {norm_top:.2e}"
            assert unimod, "d2_exact entries should be unimodular"


# ── Part 3: Summary ─────────────────────────────────────────────────────

def test_universality():
    """Verify the universal recurrence pattern: d_{p+1} from d_p."""
    print(f"\n{'=' * 70}")
    print(f"  UNIVERSAL RECURRENCE PATTERN")
    print(f"{'=' * 70}")

    print(f"\n  The recurrence d₂[c, f'] = -d₂[c, f] · d₁[f, e] / d₁[f', e]")
    print(f"  mirrors d₁[f, e'] = -d₁[f, e] · d₀[e, v] / d₀[e', v]")
    print(f"\n  General pattern for d_{{p+1}} on periodic complexes:")
    print(f"    d_{{p+1}}[σ, τ'] = -d_{{p+1}}[σ, τ] · d_p[τ, ρ] / d_p[τ', ρ]")
    print(f"  where ρ is the (p-1)-cell shared by τ, τ' in ∂σ.")

    # Quick test: both structures, one k, verify pattern
    results = []

    # SC
    mesh = build_sc_solid_periodic(N=3)
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    L = mesh['period_L']
    L_vec = np.array([L, L, L])
    cfi = mesh['cell_face_incidence']
    nC = mesh['n_cells']
    d2_top = build_d2(cfi, len(F))

    shifts = compute_edge_shifts(V, E, L_vec)
    face_edges = build_face_edge_map(F, E)
    k = 2 * np.pi / L * 0.10 * np.array([1, 0, 0.])
    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    d2_ex = build_d2_exact(cfi, face_edges, d1_ex, nC, len(F))

    norm_d1d0 = np.linalg.norm(d1_ex @ d0_k)
    norm_d2d1 = np.linalg.norm(d2_ex @ d1_ex)
    norm_d2top = np.linalg.norm(d2_top @ d1_ex)
    results.append(('SC N=3', norm_d1d0, norm_d2d1, norm_d2top))

    # Kelvin — use build_d2_top_from_foam helper
    data = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    nF = len(F)
    nC = len(data['cell_centers'])

    d2_top_k, cfi = build_d2_top_from_foam(data)

    shifts = compute_edge_shifts(V, E, L_vec)
    face_edges = build_face_edge_map(F, E)

    k = 2 * np.pi / L * 0.10 * np.array([1, 0, 0.])
    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
    d2_ex = build_d2_exact(cfi, face_edges, d1_ex, nC, nF)

    norm_d1d0 = np.linalg.norm(d1_ex @ d0_k)
    norm_d2d1 = np.linalg.norm(d2_ex @ d1_ex)
    norm_d2top = np.linalg.norm(d2_top_k @ d1_ex)
    results.append(('Kelvin N=2', norm_d1d0, norm_d2d1, norm_d2top))

    # C15 and WP foams
    for builder, name in [(build_c15_with_dual_info, 'C15 N=1'),
                           (build_wp_with_dual_info, 'WP N=1')]:
        data = builder(N=1, L_cell=4.0)
        V, E, F = data['V'], data['E'], data['F']
        L, L_vec = data['L'], data['L_vec']
        nF = len(F)
        nC = len(data['cell_centers'])

        d2_top_f, cfi = build_d2_top_from_foam(data)

        shifts = compute_edge_shifts(V, E, L_vec)
        face_edges = build_face_edge_map(F, E)

        k = 2 * np.pi / L * 0.10 * np.array([1, 0, 0.])
        d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
        d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)
        d2_ex = build_d2_exact(cfi, face_edges, d1_ex, nC, nF)

        norm_d1d0 = np.linalg.norm(d1_ex @ d0_k)
        norm_d2d1 = np.linalg.norm(d2_ex @ d1_ex)
        norm_d2top = np.linalg.norm(d2_top_f @ d1_ex)
        results.append((name, norm_d1d0, norm_d2d1, norm_d2top))

    print(f"\n  {'structure':>12s} {'||d1d0||':>12s} {'||d2d1_ex||':>12s} {'||d2top·d1||':>13s}")
    for name, n1, n2, n3 in results:
        print(f"  {name:>12s} {n1:12.2e} {n2:12.2e} {n3:13.2e}")

    for name, n1, n2, n3 in results:
        assert n1 < 1e-12, f"{name}: d1d0 should be zero"
        assert n2 < 1e-12, f"{name}: d2d1_exact should be zero"
        assert n3 > 0.1, f"{name}: d2_top should fail"

    print(f"\n  Level 1→2: d₁d₀ = 0 via face-boundary recurrence. VERIFIED.")
    print(f"  Level 2→3: d₂d₁ = 0 via cell-boundary recurrence. VERIFIED.")
    print(f"  Same pattern, same fix, same failure mode.")
    print(f"  Standard (topological) d₂ breaks exactness at k≠0 — O(k) error.")


def main():
    print("=" * 70)
    print("d₂ EXACTNESS TESTS")
    print("Does the recurrence pattern generalize to level 2→3?")
    print("=" * 70)

    test_sc()
    test_kelvin()
    test_universality()

    print(f"\n{'=' * 70}")
    print("ALL d₂ TESTS PASSED.")
    print("=" * 70)


if __name__ == '__main__':
    main()
    print("\nDone.")
