"""
Corollary 1 proof: no per-edge phase assignment preserves exactness.

Paper claim (Cor 1): if φ_e = φ(n_e) depends only on lattice shift,
then d₁d₀ ≠ 0 for generic k.

Condition: ∃ face f and positions i < j such that n_{e_i} = n_{e_j}
and the partial recurrence shift S_{i→j} ≠ 0.

S_{i→j} is computed from d₀ entries along the face boundary. At each
vertex v_m (m = i+1,...,j), the Bloch phase exp(ik·n_e·L) appears iff
v_m is the head of edge e. Explicitly:
  S_{i→j} = Σ_{m=i+1}^{j} [δ(v_m=head(e_{m-1}))·n_{m-1} − δ(v_m=head(e_m))·n_m]
This is NOT the naive sum of edge shifts — it depends on edge orientations.

Note: exp(ik·S·L) ≠ 1 for all nonzero S ∈ Z³ except on a measure-zero
set of k-vectors. The contradiction holds for almost all Bloch vectors.

Proof chain (all steps verified numerically):
  0. Uniqueness: per-face constraint matrix from d₁d₀=0 has 1D null space.
     Any d₁ with d₁d₀=0 satisfies d₁[f,:] = λ_f · d₁ᵉˣ[f,:].
  1. Per-edge hypothesis: φ(n_{e_a}) = φ(n_{e_b}) when n_{e_a} = n_{e_b}
  2. Uniqueness forces d₁[f,:] = λ_f · d₁ᵉˣ[f,:], so ψ_a = ψ_b
  3. But recurrence gives ψ_b/ψ_a = exp(ik·S·L) with S ∈ Z³
  4. When S ≠ 0: exp ≠ 1 for generic k → contradiction
  5. Therefore: per-edge phase → cannot satisfy d₁d₀=0.

RAW OUTPUT:

  Part 0 — UNIQUENESS:
    Kelvin N=2: 112/112 faces, 1D null space. SC N=3: 81/81 faces, 1D null space.
  Part 1 — INTRA-FACE CONTRADICTIONS:
    Kelvin: 37/112 faces.  SC: 45/81 faces.
  Part 2 — EXPLICIT EXAMPLE (auto-found):
    Kelvin face 1: 6 edges, shift-0 edges at pos [0,2,4].
    ψ₂/ψ₀ = 0.809-0.588j,  |ψ₂/ψ₀ - 1| = 0.618.
    S = (-1,+0,+0) by analytic formula, cross-checked by axis probing.
    exp(ik·S·L) = 0.809-0.588j.  Match: 0.00e+00.
  Part 3 — STANDARD VERIFICATION:
    ||d₁ˢᵗᵈd₀||: 8.37 (Kelvin), 7.74 (SC).  ||d₁ᵉˣd₀||: 8.46e-16, 5.58e-16.
  Logic chain: uniqueness + intra-face inconsistency → Cor 1. QED.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/8_test_cor1_proof.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import build_kelvin_with_dual_info
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup
from core_math.builders.solids_periodic import build_sc_supercell_periodic


def build_face_edges(E, F):
    """Build ordered edge info per face: [(edge_idx, orient), ...].

    Requires F[f] to be a cyclically ordered vertex list.
    """
    edge_map = {}
    for idx, (i, j) in enumerate(E):
        edge_map[(i, j)] = (idx, +1)
        edge_map[(j, i)] = (idx, -1)

    face_edges = []
    for f_idx, face in enumerate(F):
        n = len(face)
        edges_info = []
        for v_pos in range(n):
            i = face[v_pos]
            j = face[(v_pos + 1) % n]
            assert (i, j) in edge_map, (
                f"Face {f_idx}: consecutive vertices ({i},{j}) don't form an edge")
            e_idx, orient = edge_map[(i, j)]
            edges_info.append((e_idx, orient))
        face_edges.append(edges_info)
    return face_edges


def get_recurrence_phases(face, edges_info, d0_k):
    """Compute recurrence phases ψ₀=1, ψ₁, ..., ψ_{n-1} for a face."""
    n = len(face)
    phases = [1.0 + 0j]
    for i in range(1, n):
        e_prev, orient_prev = edges_info[i - 1]
        e_curr, orient_curr = edges_info[i]
        v = face[i]
        phase_curr = -orient_prev * phases[i - 1] * d0_k[e_prev, v] / (orient_curr * d0_k[e_curr, v])
        phases.append(phase_curr)
    return phases


def build_face_constraint_matrix(face, edges_info, d0_k):
    """Build n×n constraint matrix A_f from d₁d₀=0 restricted to face f.

    Unknowns: ψ_0, ..., ψ_{n-1} (edge phases).
    Constraint from vertex face[i]: the two face edges meeting at face[i]
    must satisfy orient_{i-1}·d₀[e_{i-1}, v_i]·ψ_{i-1} + orient_i·d₀[e_i, v_i]·ψ_i = 0.
    Uniqueness iff null(A_f) = 1.
    """
    n = len(face)
    A = np.zeros((n, n), dtype=complex)
    for i in range(n):
        v = face[i]
        e_prev, orient_prev = edges_info[(i - 1) % n]
        e_curr, orient_curr = edges_info[i]
        A[i, (i - 1) % n] = orient_prev * d0_k[e_prev, v]
        A[i, i] = orient_curr * d0_k[e_curr, v]
    return A


def test_uniqueness(name, V, E, F, L_vec, shifts, k):
    """Part 0: Verify per-face null space is exactly 1D on all faces."""
    n_V = len(V)
    assert set(range(n_V)) == {v for e in E for v in e}, \
        "Vertex indices not contiguous 0..V-1"

    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    face_edges = build_face_edges(E, F)

    n_pass = 0
    for f_idx in range(len(F)):
        A = build_face_constraint_matrix(F[f_idx], face_edges[f_idx], d0_k)
        n = A.shape[0]
        rank = np.linalg.matrix_rank(A, tol=1e-10)
        null_dim = n - rank
        assert null_dim == 1, (
            f"Face {f_idx}: null_dim={null_dim}, expected 1 (n={n}, rank={rank})")
        n_pass += 1

    print(f"    {name}: all {n_pass}/{len(F)} faces have 1D null space. "
          f"Uniqueness verified.")
    return n_pass


def count_intra_face_contradictions(F, face_edges, shifts, d0_k):
    """Count faces where same-shift edges get different recurrence phases."""
    contradictions = 0
    for f_idx in range(len(F)):
        phases = get_recurrence_phases(F[f_idx], face_edges[f_idx], d0_k)
        n = len(face_edges[f_idx])

        # Group edge positions by shift
        shift_groups = {}
        for pos in range(n):
            e_idx, _ = face_edges[f_idx][pos]
            key = tuple(shifts[e_idx])
            if key not in shift_groups:
                shift_groups[key] = []
            shift_groups[key].append((pos, phases[pos]))

        for key, group in shift_groups.items():
            if len(group) > 1:
                ref_phase = group[0][1]
                if any(abs(ph - ref_phase) > 1e-10 for _, ph in group[1:]):
                    contradictions += 1
                    break  # one contradiction per face is enough

    return contradictions


def test_intra_face(name, V, E, F, L_vec, shifts, k):
    """Part 1: Count intra-face contradictions on a mesh."""
    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    face_edges = build_face_edges(E, F)
    n_contra = count_intra_face_contradictions(F, face_edges, shifts, d0_k)
    print(f"    {name} (V={len(V)}, E={len(E)}, F={len(F)}):")
    print(f"      Faces with same-shift edges having different recurrence phases: "
          f"{n_contra}/{len(F)}")
    assert n_contra > 0, f"Expected contradictions on {name}, got 0"
    return n_contra


def find_example_face(F, face_edges, shifts, d0_k):
    """Find a face with ≥2 same-shift edges having different recurrence phases.

    Returns (f_idx, shift_key, positions) or None.
    Prefers shift=(0,0,0) for clarity.
    """
    best = None
    for f_idx in range(len(F)):
        phases = get_recurrence_phases(F[f_idx], face_edges[f_idx], d0_k)
        n = len(face_edges[f_idx])

        shift_groups = {}
        for pos in range(n):
            e_idx, _ = face_edges[f_idx][pos]
            key = tuple(shifts[e_idx])
            if key not in shift_groups:
                shift_groups[key] = []
            shift_groups[key].append((pos, phases[pos]))

        for key, group in shift_groups.items():
            if len(group) >= 2:
                ref_phase = group[0][1]
                if any(abs(ph - ref_phase) > 1e-3 for _, ph in group[1:]):
                    positions = [pos for pos, _ in group]
                    # Prefer shift=(0,0,0) — clearest for paper
                    if key == (0, 0, 0):
                        return f_idx, key, positions
                    if best is None:
                        best = (f_idx, key, positions)

    return best


def analytic_net_shift(edges_info, E, shifts, p0, p1):
    """Compute net lattice shift S from pos p0 to pos p1 analytically.

    Formula: S = Σ_{m=p0+1}^{p1} [δ(v_m=head(e_{m-1}))·n_{m-1} − δ(v_m=head(e_m))·n_m]
    where v_m = head(e_{m-1}) iff orient_{m-1} = +1, and v_m = head(e_m) iff orient_m = -1.
    """
    S = np.zeros(3, dtype=float)
    for m in range(p0 + 1, p1 + 1):
        e_prev, o_prev = edges_info[m - 1]
        e_curr, o_curr = edges_info[m]
        n_prev = shifts[e_prev].astype(float)
        n_curr = shifts[e_curr].astype(float)
        # v_m is head of e_{m-1} iff orient_{m-1} = +1
        # v_m is head of e_m iff orient_m = -1
        S += (1 if o_prev == +1 else 0) * n_prev - (1 if o_curr == -1 else 0) * n_curr
    return np.round(S).astype(int)


def compute_net_shift(face, edges_info, V, E, L_vec, shifts, p0, p1):
    """Compute net lattice shift S from pos p0 to pos p1.

    Primary: analytic formula from edge shifts and orientations.
    Cross-check: 3 axis-aligned k-probes (independent verification).
    """
    # Analytic
    S = analytic_net_shift(edges_info, E, shifts, p0, p1)

    # Cross-check via axis probing
    S_probe = np.zeros(3, dtype=int)
    for axis in range(3):
        k_probe = np.zeros(3)
        k_probe[axis] = 0.01
        d0_probe = build_d0_bloch(V, E, k_probe, L_vec, shifts)
        phases_probe = get_recurrence_phases(face, edges_info, d0_probe)
        ratio = phases_probe[p1] / phases_probe[p0]
        raw = np.angle(ratio) / (k_probe[axis] * L_vec[axis])
        S_probe[axis] = int(round(raw))
        assert abs(raw - S_probe[axis]) < 0.01, (
            f"axis {axis}: S={raw:.4f} not close to integer")

    assert np.array_equal(S, S_probe), (
        f"Analytic S={S} != probe S={S_probe}")
    return S


def test_explicit_example(V, E, F, L_vec, L, shifts, name):
    """Part 2: Show explicit contradiction on an auto-found face."""
    k = 2 * np.pi / L * 0.10 * np.array([1.0, 0.3, 0.7])
    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    face_edges = build_face_edges(E, F)

    result = find_example_face(F, face_edges, shifts, d0_k)
    assert result is not None, f"No example face found on {name}"
    f_idx, shift_key, positions = result

    face = F[f_idx]
    edges_info = face_edges[f_idx]
    phases = get_recurrence_phases(face, edges_info, d0_k)
    n = len(face)

    print(f"\n  Face {f_idx}: vertices {list(face)}, {n} edges")
    for pos in range(n):
        e_idx, orient = edges_info[pos]
        i_v, j_v = E[e_idx]
        sh = shifts[e_idx]
        ph_angle = np.angle(phases[pos]) * 180 / np.pi
        print(f"    pos {pos}: edge {e_idx:2d} ({i_v}→{j_v}), "
              f"shift=({sh[0]:+d},{sh[1]:+d},{sh[2]:+d}), "
              f"ψ = ∠{ph_angle:+.1f}°")

    shift_str = f"({int(shift_key[0]):+d},{int(shift_key[1]):+d},{int(shift_key[2]):+d})"
    print(f"\n  Same-shift edges (shift={shift_str}) at positions {positions}:")
    for pos in positions:
        print(f"    ψ_{pos} = {phases[pos]:.6f}  (|ψ|={abs(phases[pos]):.6f})")

    # Check contradiction between first two
    p0, p1 = positions[0], positions[1]
    ratio = phases[p1] / phases[p0]
    print(f"\n  ψ_{p1}/ψ_{p0} = {ratio:.6f}")
    print(f"  |ψ_{p1}/ψ_{p0} - 1| = {abs(ratio - 1):.6e}")
    assert abs(ratio - 1) > 1e-3, "Expected ψ ratio ≠ 1"

    # Compute net shift from data (Bug 3 fix: not hardcoded)
    S = compute_net_shift(face, edges_info, V, E, L_vec, shifts, p0, p1)
    expected = np.exp(1j * np.dot(k, S * L_vec))
    print(f"\n  Net shift S = ({S[0]:+d},{S[1]:+d},{S[2]:+d})  (computed from data)")
    print(f"  exp(ik·S·L) = {expected:.6f}")
    print(f"  Match |ratio - exp(ik·S·L)|: {abs(ratio - expected):.2e}")
    assert abs(ratio - expected) < 1e-12, "Ratio should equal exp(ik·S·L)"
    assert np.any(S != 0), "Net shift S should be nonzero"

    # Trace recurrence path
    print(f"\n  Recurrence path pos {p0} → pos {p1}:")
    for i in range(p0 + 1, p1 + 1):
        e_prev, orient_prev = edges_info[i - 1]
        e_curr, orient_curr = edges_info[i]
        v = face[i]
        sh_prev = shifts[e_prev]
        sh_curr = shifts[e_curr]
        step_ratio = -orient_prev * d0_k[e_prev, v] / (orient_curr * d0_k[e_curr, v])
        v_is_head_prev = (v == E[e_prev][1])
        v_is_head_curr = (v == E[e_curr][1])
        print(f"    Step {i}: vertex {v}")
        print(f"      e_prev={e_prev}(n=({sh_prev[0]:+d},{sh_prev[1]:+d},{sh_prev[2]:+d}), "
              f"{'head' if v_is_head_prev else 'tail'}), "
              f"e_curr={e_curr}(n=({sh_curr[0]:+d},{sh_curr[1]:+d},{sh_curr[2]:+d}), "
              f"{'head' if v_is_head_curr else 'tail'})")
        print(f"      ratio = {abs(step_ratio):.4f}∠{np.angle(step_ratio)*180/np.pi:+.1f}°")

    S_str = f"({S[0]:+d},{S[1]:+d},{S[2]:+d})"
    print(f"\n  S = {S_str} ≠ 0 → ψ_{p1} ≠ ψ_{p0} → per-edge phase impossible. QED.")


def test_standard_confirms():
    """Part 3: Verify that standard d₁ (which IS a per-edge function) fails."""
    # Kelvin
    data = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    shifts = compute_edge_shifts(V, E, L_vec)
    crossings = compute_edge_crossings(V, E, L)
    edge_lookup = build_edge_lookup(E, crossings)

    k = 2 * np.pi / L * 0.10 * np.array([1.0, 0.3, 0.7])
    d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
    d1_std = build_d1_bloch_standard(V, E, F, L, k, edge_lookup, crossings)
    d1_ex = build_d1_bloch_exact(V, E, F, k, L_vec, d0_k)

    norm_std = np.linalg.norm(d1_std @ d0_k)
    norm_ex = np.linalg.norm(d1_ex @ d0_k)

    print(f"\n  Standard uses φ(n_e) = exp(ik·n_e·L) — a per-edge function of n_e.")
    print(f"  Kelvin N=2: ||d₁ˢᵗᵈd₀|| = {norm_std:.2f},  ||d₁ᵉˣd₀|| = {norm_ex:.2e}")
    assert norm_std > 1.0, "Standard should fail"
    assert norm_ex < 1e-12, "Exact should pass"

    # SC N=3
    V_sc, E_sc, F_sc, _ = build_sc_supercell_periodic(3)
    L_sc = 2.0 * 3
    L_vec_sc = np.array([L_sc, L_sc, L_sc])
    shifts_sc = compute_edge_shifts(V_sc, E_sc, L_vec_sc)
    crossings_sc = compute_edge_crossings(V_sc, E_sc, L_sc)
    edge_lookup_sc = build_edge_lookup(E_sc, crossings_sc)

    k_sc = 2 * np.pi / L_sc * 0.10 * np.array([1.0, 0.3, 0.7])
    d0_sc = build_d0_bloch(V_sc, E_sc, k_sc, L_vec_sc, shifts_sc)
    d1_std_sc = build_d1_bloch_standard(V_sc, E_sc, F_sc, L_sc, k_sc, edge_lookup_sc, crossings_sc)
    d1_ex_sc = build_d1_bloch_exact(V_sc, E_sc, F_sc, k_sc, L_vec_sc, d0_sc)

    norm_std_sc = np.linalg.norm(d1_std_sc @ d0_sc)
    norm_ex_sc = np.linalg.norm(d1_ex_sc @ d0_sc)

    print(f"  SC N=3:    ||d₁ˢᵗᵈd₀|| = {norm_std_sc:.2f},  ||d₁ᵉˣd₀|| = {norm_ex_sc:.2e}")
    assert norm_std_sc > 1.0, "Standard should fail on SC"
    assert norm_ex_sc < 1e-12, "Exact should pass on SC"

    print(f"\n  Per-edge phases → exactness failure confirmed on both structures.")


def main():
    print("=" * 70)
    print("COROLLARY 1 PROOF: no per-edge phase preserves exactness")
    print("Paper: §3.3 Cor 1 (R1-1)")
    print("=" * 70)

    # --- Build meshes ---
    data = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    shifts = compute_edge_shifts(V, E, L_vec)
    k = 2 * np.pi / L * 0.10 * np.array([1.0, 0.3, 0.7])

    V_sc, E_sc, F_sc, _ = build_sc_supercell_periodic(3)
    L_sc = 2.0 * 3
    L_vec_sc = np.array([L_sc, L_sc, L_sc])
    shifts_sc = compute_edge_shifts(V_sc, E_sc, L_vec_sc)
    k_sc = 2 * np.pi / L_sc * 0.10 * np.array([1.0, 0.3, 0.7])

    # --- Part 0: Uniqueness (per-face 1D null space) ---
    print(f"\n{'=' * 70}")
    print(f"  UNIQUENESS — per-face constraint null space is 1D")
    print(f"{'=' * 70}")

    test_uniqueness("Kelvin N=2", V, E, F, L_vec, shifts, k)
    test_uniqueness("SC cubic N=3", V_sc, E_sc, F_sc, L_vec_sc, shifts_sc, k_sc)

    # --- Part 1: Count intra-face contradictions ---
    print(f"\n{'=' * 70}")
    print(f"  INTRA-FACE CONTRADICTIONS — per-edge phase impossibility")
    print(f"{'=' * 70}")

    n1 = test_intra_face("Kelvin N=2", V, E, F, L_vec, shifts, k)
    n2 = test_intra_face("SC cubic N=3", V_sc, E_sc, F_sc, L_vec_sc, shifts_sc, k_sc)

    # --- Part 2: Explicit example (auto-found) ---
    print(f"\n{'=' * 70}")
    print(f"  EXPLICIT EXAMPLE — auto-found face with contradiction")
    print(f"{'=' * 70}")

    test_explicit_example(V, E, F, L_vec, L, shifts, "Kelvin N=2")

    # --- Part 3: Standard confirms ---
    print(f"\n{'=' * 70}")
    print(f"  VERIFICATION: standard d₁ fails with per-edge phases")
    print(f"{'=' * 70}")

    test_standard_confirms()

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"ALL COR 1 PROOF TESTS COMPLETE.")
    print(f"  Part 0: Uniqueness — 1D null space on all faces (both structures)")
    print(f"  Part 1: Intra-face contradictions: {n1}/{len(F)} (Kelvin), {n2}/{len(F_sc)} (SC)")
    print(f"  Part 2: Explicit mechanism: ψ_b/ψ_a = exp(ik·S·L), S ≠ 0")
    print(f"  Part 3: Standard d₁ (per-edge) fails on both structures")
    print(f"\n  Logic chain: uniqueness + intra-face inconsistency → Cor 1. QED.")


if __name__ == '__main__':
    main()
    print("\nDone.")
