"""
Bloch-periodic cochain complex infrastructure.

Shared helpers for building the full exact complex 0 → C⁰ → C¹ → C² → C³ → 0
at arbitrary k-point.  Used by tests 13, 18, 19 and any future test that needs
the complete complex or Hodge structure.

Functions:
  build_face_edge_map     — face → set of edge indices (pure topology)
  build_cell_face_incidence — foam data → oriented cell-face incidence list
  build_d2_bloch_exact    — k-dependent d₂ via cell-boundary recurrence
  load_foam               — unified foam loader with optional Hodge stars
"""

import numpy as np
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact


def build_face_edge_map(faces, edges):
    """Map face index -> set of edge indices.

    Args:
        faces: list of face vertex lists
        edges: list of (v1, v2) edge tuples

    Returns:
        dict[int, set[int]]: face_idx -> set of edge indices
    """
    E_canonical = [(min(a, b), max(a, b)) for a, b in edges]
    edge_to_idx = {e: i for i, e in enumerate(E_canonical)}
    result = {}
    for f_idx, face in enumerate(faces):
        n = len(face)
        fe = set()
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            fe.add(edge_to_idx[(min(v1, v2), max(v1, v2))])
        result[f_idx] = fe
    return result


def build_cell_face_incidence(data):
    """Build oriented cell-face incidence from foam data.

    The foam builders provide face_to_cells (face → two adjacent cells).
    This function orients the signs so that d₂_top @ d₁(k=0) = 0, producing
    the cell_face_incidence list needed by build_d2_bloch_exact.

    Args:
        data: foam dict with keys V, E, F, L_vec, face_to_cells, cell_centers

    Returns:
        (cfi, n_flipped) where:
          cfi: list of lists, cfi[c] = [(face_idx, ±1), ...]
          n_flipped: number of faces whose orientation was corrected
    """
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    face_to_cells = data['face_to_cells']
    cell_centers = data['cell_centers']
    nF = len(F)
    nC = len(cell_centers)

    d2_top = np.zeros((nC, nF))
    for f_idx in range(nF):
        ca, cb = face_to_cells[f_idx]
        d2_top[ca, f_idx] = +1
        d2_top[cb, f_idx] = -1

    shifts = compute_edge_shifts(V, E, L_vec)
    d0_k0 = build_d0_bloch(V, E, np.zeros(3), L_vec, shifts)
    d1_k0 = build_d1_bloch_exact(V, E, F, np.zeros(3), L_vec, d0_k0)

    n_flipped = 0
    if np.linalg.norm(d2_top @ d1_k0) > 1e-10:
        for f_idx in range(nF):
            face = F[f_idx]
            verts = np.array([V[v] for v in face[:3]])
            normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            ca, cb = face_to_cells[f_idx]
            centroid = np.mean(np.array([V[v] for v in face]), axis=0)
            old_sign = d2_top[ca, f_idx]
            if np.dot(normal, cell_centers[ca] - centroid) > 0:
                d2_top[ca, f_idx] = -1
                d2_top[cb, f_idx] = +1
            else:
                d2_top[ca, f_idx] = +1
                d2_top[cb, f_idx] = -1
            if d2_top[ca, f_idx] != old_sign:
                n_flipped += 1

    assert np.linalg.norm(d2_top @ d1_k0) < 1e-10, \
        f"orientation fix failed: ||d2·d1|| = {np.linalg.norm(d2_top @ d1_k0):.2e}"

    cfi = [[] for _ in range(nC)]
    for c in range(nC):
        for f in range(nF):
            if abs(d2_top[c, f]) > 0.5:
                cfi[c].append((f, int(d2_top[c, f])))

    return cfi, n_flipped


def build_d2_bloch_exact(cfi, face_edges, d1_ex, nC, nF):
    """Build exactness-preserving d₂(k) via cell-boundary recurrence.

    Same principle as build_d1_bloch_exact: propagate phases along a spanning
    tree of the face adjacency graph within each cell, using d₁ ratios to
    determine the relative phases.

    Args:
        cfi: cell-face incidence, cfi[c] = [(face_idx, ±1), ...]
        face_edges: dict[int, set[int]], from build_face_edge_map
        d1_ex: (nF, nE) complex array, the exact d₁(k)
        nC: number of cells
        nF: number of faces

    Returns:
        d2: (nC, nF) complex array with d₂(k) @ d₁(k) = 0
    """
    d2 = np.zeros((nC, nF), dtype=complex)
    for c_idx in range(nC):
        faces_of_c = cfi[c_idx]
        if not faces_of_c:
            continue
        adj = {f: [] for f, _ in faces_of_c}
        for i, (f1, _) in enumerate(faces_of_c):
            for j, (f2, _) in enumerate(faces_of_c):
                if i >= j:
                    continue
                shared = face_edges[f1] & face_edges[f2]
                for e in shared:
                    adj[f1].append((f2, e))
                    adj[f2].append((f1, e))
        f0, o0 = faces_of_c[0]
        d2[c_idx, f0] = o0
        visited = {f0}
        queue = [f0]
        while queue:
            fc = queue.pop(0)
            for fn, es in adj[fc]:
                if fn in visited:
                    continue
                d1c, d1n = d1_ex[fc, es], d1_ex[fn, es]
                if abs(d1n) < 1e-14:
                    raise ValueError(f'd1[{fn},{es}] = 0')
                d2[c_idx, fn] = -d2[c_idx, fc] * d1c / d1n
                visited.add(fn)
                queue.append(fn)
        assert len(visited) == len(faces_of_c)
    return d2


def load_foam(builder, N, L_cell, with_stars=False):
    """Load foam and prepare complex infrastructure.

    Args:
        builder: foam builder function (e.g. build_kelvin_with_dual_info)
        N: supercell size
        L_cell: cell size
        with_stars: if True, also compute Hodge stars M0, M1, M2, M3

    Returns:
        dict with keys: V, E, F, L_vec, L, nC, cfi, n_flipped,
        face_edges, shifts.  If with_stars=True, also M0, M1, M2, M3.
    """
    data = builder(N=N, L_cell=L_cell)
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    L = data['L']
    nC = len(data['cell_centers'])

    cfi, n_flipped = build_cell_face_incidence(data)
    face_edges = build_face_edge_map(F, E)
    shifts = compute_edge_shifts(V, E, L_vec)

    mesh = dict(V=V, E=E, F=F, L_vec=L_vec, L=L, nC=nC,
                cfi=cfi, n_flipped=n_flipped,
                face_edges=face_edges, shifts=shifts)

    if with_stars:
        from physics.hodge import build_hodge_stars_voronoi
        star1, star2 = build_hodge_stars_voronoi(data)
        # M₀, M₃: uniform approximations. Spectral structure (pairing,
        # Hodge decomposition) depends on d₀, d₁ and M₁, M₂.
        # M₀ cancels in P_E = U₀U₀ᴴ (only column space matters).
        mesh['M0'] = np.ones(len(V)) * (L_vec[0]**3 / len(V))
        mesh['M1'] = star1
        mesh['M2'] = star2
        mesh['M3'] = np.ones(nC) / (L_vec[0]**3 / nC)

    return mesh
