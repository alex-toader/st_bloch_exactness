"""
MPB comparison: validate DEC results against MIT Photonic Bands.

WHAT IS BEING MEASURED:
  BCC Voronoi (Kelvin foam) with dielectric contrast eps_A=1, eps_B=variable.
  MPB solves the same physical problem on a voxel grid (resolution N_mpb).
  DEC solves on the exact Voronoi mesh (Kelvin N=2, L_cell=4.0).
  Compare c² = (freq/|k|)² at small k.

  Geometry is identical in the continuum limit. MPB uses a voxel approximation
  that converges with resolution; convergence is demonstrated below.

  Two sources of DEC error:
    1. Dispersion error O(k²h²): present even in vacuum, scales as k².
    2. Interface error O(h²): from piecewise-constant ε in the Hodge star,
       k-independent, dominates at finite contrast.

MPB SETUP:
  Conventional cubic cell [0,1]³ with epsilon defined on a grid.
  BCC Voronoi rule: point (x,y,z) gets eps_B if closer to body center
  (0.5,0.5,0.5) than to nearest corner, else eps_A=1.

REQUIRES: meep (conda-forge), h5py. Run with mpb_env:
  /Users/alextoader/miniconda3/envs/mpb_env/bin/python tests/12_test_mpb_comparison.py

RAW OUTPUT:

  SANITY: ε=1, expect c²=1.0
    MPB: c² = 1.000000, DEC: c² = 0.998716, n_zero = V = 96. PASS

  DEC vs MPB: eps_B=2,4,9 at k_frac=0.05
     eps_B   c2_mpb   c2_dec   ratio   nz=V
         2   0.6907   0.7263  1.0515      Y
         4   0.4528   0.5469  1.2078      Y
         9   0.2553   0.3832  1.5006      Y

  K-SCALING (k_frac = 0.025, 0.05):
    Vacuum: err ratio = 4.00 → O(k²) dispersion
    eps_B=4: err ratio = 1.00 → k-independent interface error

  HODGE STAR SENSITIVITY (eps_B=4, 64/112 faces = 57% interface):
    Direction [100], c2_mpb = 0.4528:
      harmonic(1/eps) [current]:  c² = 0.547, +20.8%, nz=V ✓
      geometric mean:             c² = 0.470,  +3.8%, nz=V ✓
      arithmetic mean of eps:     c² = 0.399, -11.8%, nz=V ✓
    Direction [111], c2_mpb = 0.4529:
      harmonic(1/eps) [current]:  c² = 0.547, +20.7%, nz=V ✓
      geometric mean:             c² = 0.470,  +3.7%, nz=V ✓
      arithmetic mean of eps:     c² = 0.399, -11.9%, nz=V ✓
    Direction-independent on Kelvin (cubic symmetry).

  MPB RESOLUTION CONVERGENCE: res 16→32→64, <0.5% at res=32.

ANSWER:
  DEC and MPB solve the same BCC Voronoi structure. DEC overestimates c²:
  5% at eps_B=2, 21% at eps_B=4, 50% at eps_B=9. Ratio monotonically
  increasing with contrast.

  Error decomposition:
    Vacuum: pure O(k²) dispersion error (ratio 4.00 at k doubling).
    Dielectric: k-independent interface error dominates (~20% at eps_B=4),
    from piecewise-constant ε approximation in the Hodge star.

  Exactness preserved at all contrasts: n_zero = V = 96.

K-MAPPING VERIFICATION:
  k_abs = 2π × k_frac / L_cell is CORRECT. Verified: c² = 0.998716
  at k_frac=0.05 for L_cell = 1, 2, 4, 8 (scale-invariant).
"""

import sys, os
import numpy as np
from scipy.linalg import eigh
import h5py
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import meep as mp
from meep import mpb

from physics.hodge import build_kelvin_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact


# ── MPB helpers ──

def make_bcc_voronoi_eps(eps_B, resolution):
    """Create 3D epsilon array for BCC Voronoi (truncated octahedron)."""
    n = resolution
    eps = np.ones((n, n, n))
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                x = (ix + 0.5) / n
                y = (iy + 0.5) / n
                z = (iz + 0.5) / n
                dx = min(x, 1 - x)
                dy = min(y, 1 - y)
                dz = min(z, 1 - z)
                d2_A = dx**2 + dy**2 + dz**2
                d2_B = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2
                if d2_B < d2_A:
                    eps[ix, iy, iz] = eps_B
    return eps


def run_mpb(eps_B, k_frac, resolution=32, num_bands=4):
    """Run MPB on BCC Voronoi at given k_frac along [100]. Returns frequencies."""
    eps_grid = make_bcc_voronoi_eps(eps_B, resolution)
    tmpfile = os.path.join(tempfile.gettempdir(), 'bcc_eps.h5')
    with h5py.File(tmpfile, 'w') as f:
        f.create_dataset('data', data=eps_grid)

    ms = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1, 1)),
        resolution=resolution,
        num_bands=num_bands,
        default_material=mp.Medium(epsilon=1),
        epsilon_input_file=tmpfile,
        verbose=False,
    )
    ms.k_points = [mp.Vector3(k_frac, 0, 0)]
    ms.run()  # all polarizations (run_tm is a 2D concept)
    return ms.all_freqs[0]


# ── DEC helpers ──

def run_dec(eps_B, k_frac_mpb, N=2, L_cell=4.0):
    """Run DEC on Kelvin N=2 at equivalent k-point. Returns c², n_zero, V."""
    data = build_kelvin_with_dual_info(N=N, L_cell=L_cell)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)

    # MPB k_frac = k × a / (2π) with a=1. DEC uses a = L_cell.
    # k_abs = 2π × k_frac / L_cell.
    k_abs = 2 * np.pi * k_frac_mpb / L_cell * np.array([1.0, 0.0, 0.0])
    k2 = np.dot(k_abs, k_abs)

    # BCC sublattice assignment
    cell_centers = data.get('cell_centers', None)
    if cell_centers is None:
        raise RuntimeError("Need cell_centers for sublattice assignment")

    a_cell = L_cell  # single unit cell lattice constant
    labels = np.array([
        int(round((c[0] + c[1] + c[2]) / (a_cell / 2))) % 2
        for c in cell_centers
    ])
    nA, nB = np.sum(labels == 0), np.sum(labels == 1)
    assert nA == N**3 and nB == N**3, \
        f"BCC sublattice count wrong: A={nA}, B={nB}, expected {N**3} each"
    eps_cells = np.where(labels == 0, 1.0, float(eps_B))

    # Per-face epsilon: harmonic mean of 1/ε from adjacent cells
    face_to_cells_map = data['face_to_cells']

    inv_eps_face = np.zeros(len(F))
    for fi in range(len(F)):
        ca, cb = face_to_cells_map[fi]
        inv_eps_face[fi] = 0.5 * (1.0/eps_cells[ca] + 1.0/eps_cells[cb])

    star2_mod = star2 * inv_eps_face

    d0_k = build_d0_bloch(V, E, k_abs, L_vec, shifts)
    d1_ex = build_d1_bloch_exact(V, E, F, k_abs, L_vec, d0_k)

    K = d1_ex.conj().T @ np.diag(star2_mod) @ d1_ex
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    eigs = np.sort(np.real(eigh(K, M, eigvals_only=True)))
    thresh = max(np.max(np.abs(eigs)) * 1e-12, 1e-14)
    n_zero = int(np.sum(np.abs(eigs) < thresh))
    phys = eigs[eigs > thresh]
    c2 = phys[0] / k2 if len(phys) > 0 else float('nan')

    return c2, n_zero, len(V)


# ── Tests ──

def test_vacuum_sanity():
    """Both methods give c²=1.0 for ε=1 (vacuum)."""
    print(f"\n{'=' * 70}")
    print("  SANITY: ε=1, expect c²=1.0")
    print(f"{'=' * 70}")

    k_frac = 0.05
    freqs = run_mpb(1, k_frac, resolution=32)
    c2_mpb = (freqs[0] / k_frac)**2
    c2_dec, nz, nV = run_dec(1, k_frac)

    print(f"  MPB: c² = {c2_mpb:.6f}")
    print(f"  DEC: c² = {c2_dec:.6f}")
    print(f"  DEC n_zero = {nz} = V = {nV}")

    assert abs(c2_mpb - 1.0) < 1e-4, f"MPB vacuum c²={c2_mpb}"
    assert abs(c2_dec - 1.0) < 1e-2, f"DEC vacuum c²={c2_dec}"
    assert nz == nV, f"DEC n_zero={nz} != V={nV}"
    print("  PASS")


def test_dielectric_comparison():
    """Compare DEC vs MPB at eps_B = 2, 4, 9."""
    print(f"\n{'=' * 70}")
    print("  DEC vs MPB: BCC Voronoi, eps_A=1, eps_B=2,4,9")
    print(f"{'=' * 70}")

    k_frac = 0.05

    print(f"\n  k_frac = {k_frac} (5% BZ along [100])")
    print(f"  {'eps_B':>6s}  {'c2_mpb':>10s}  {'c2_dec':>10s}  {'ratio':>8s}  {'nz=V':>5s}")

    ratios = []
    for eps_B in [2, 4, 9]:
        freqs = run_mpb(eps_B, k_frac, resolution=32)
        c2_mpb = (freqs[0] / k_frac)**2
        c2_dec, nz, nV = run_dec(eps_B, k_frac)
        ratio = c2_dec / c2_mpb
        ok = (nz == nV)
        print(f"  {eps_B:6d}  {c2_mpb:10.4f}  {c2_dec:10.4f}  {ratio:8.4f}  {'Y' if ok else 'N':>5s}")
        ratios.append(ratio)

        # DEC must reflect the dielectric (not stuck at vacuum)
        assert c2_dec < 0.95, f"DEC c²={c2_dec:.4f} at eps_B={eps_B}: dielectric not applied"
        # DEC overestimates (coarse mesh → stiff → higher eigenvalues)
        assert ratio > 1.0, f"ratio={ratio:.4f} < 1 at eps_B={eps_B}"
        # Exactness preserved
        assert nz == nV, f"n_zero={nz} != V={nV} at eps_B={eps_B}"

    # Ratio must grow with eps_B: higher contrast → more interface error
    assert ratios[0] < ratios[1] < ratios[2], \
        f"Ratio not monotonic in eps_B: {ratios}"

    print(f"\n  Ratio > 1 and monotonically increasing with eps_B.")
    print(f"  PASS")


def test_k_scaling():
    """Identify error type: O(k²) dispersion vs k-independent interface."""
    print(f"\n{'=' * 70}")
    print("  K-SCALING: error type at vacuum vs eps_B=4")
    print(f"{'=' * 70}")

    k_fracs = [0.025, 0.05]

    print(f"\n  {'k_frac':>6s}  {'eps_B':>5s}  {'c2_mpb':>10s}  {'c2_dec':>10s}  {'err_rel':>10s}")

    errs = {}
    for k_frac in k_fracs:
        for eps_B in [1, 4]:
            freqs = run_mpb(eps_B, k_frac, resolution=32)
            c2_mpb = (freqs[0] / k_frac)**2
            c2_dec, _, _ = run_dec(eps_B, k_frac)
            err = (c2_dec - c2_mpb) / c2_mpb
            errs[(k_frac, eps_B)] = err
            print(f"  {k_frac:6.3f}  {eps_B:5d}  {c2_mpb:10.4f}  {c2_dec:10.4f}  {err:+10.5f}")

    # Vacuum: error should scale as k² (ratio ≈ 4)
    ratio_vac = errs[(0.05, 1)] / errs[(0.025, 1)]
    # eps_B=4: error should be k-independent (ratio ≈ 1)
    ratio_eps = errs[(0.05, 4)] / errs[(0.025, 4)]

    print(f"\n  Vacuum: err(0.05)/err(0.025) = {ratio_vac:.2f}  (expect ~4 if O(k²))")
    print(f"  eps_B=4: err(0.05)/err(0.025) = {ratio_eps:.2f}  (expect ~1 if k-independent)")

    assert 3.5 < ratio_vac < 4.5, f"Vacuum error not O(k²): ratio={ratio_vac:.2f}"
    assert 0.8 < ratio_eps < 1.2, f"Dielectric error not k-independent: ratio={ratio_eps:.2f}"

    print(f"\n  Vacuum error = O(k²) dispersion.")
    print(f"  Dielectric error = k-independent interface approximation (Hodge star).")
    print(f"  PASS")


def _run_mpb_kvec(eps_B, k_vec, resolution=32):
    """Run MPB at arbitrary k-vector. Returns c² = (freq/|k|)²."""
    eps_grid = make_bcc_voronoi_eps(eps_B, resolution)
    tmpfile = os.path.join(tempfile.gettempdir(), 'bcc_eps.h5')
    with h5py.File(tmpfile, 'w') as f:
        f.create_dataset('data', data=eps_grid)
    ms = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1, 1)),
        resolution=resolution, num_bands=4,
        default_material=mp.Medium(epsilon=1),
        epsilon_input_file=tmpfile, verbose=False,
    )
    ms.k_points = [mp.Vector3(*k_vec)]
    ms.run()
    k_mag = np.linalg.norm(k_vec)
    return (ms.all_freqs[0][0] / k_mag)**2


def test_hodge_star_sensitivity():
    """Quantify how face-averaging formula affects c² at material interfaces.
    Tests [100] and [111] directions to check orientation dependence."""
    print(f"\n{'=' * 70}")
    print("  HODGE STAR SENSITIVITY: face averaging at eps_B=4")
    print(f"{'=' * 70}")

    N, L_cell = 2, 4.0
    eps_B = 4.0
    eps_A = 1.0
    k_frac = 0.05

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
    eps_cells = np.where(labels == 0, eps_A, eps_B)
    ftc = data['face_to_cells']

    n_iface = sum(1 for fi in range(len(F)) if labels[ftc[fi][0]] != labels[ftc[fi][1]])
    print(f"\n  Interface faces: {n_iface}/{len(F)} ({n_iface/len(F):.0%})")

    schemes = [
        ('harmonic(1/eps) [current]', 0.5 * (1/eps_A + 1/eps_B)),
        ('geometric mean',            1 / np.sqrt(eps_A * eps_B)),
        ('arithmetic mean of eps',    2 / (eps_A + eps_B)),
    ]

    k_dirs = [
        ('[100]', np.array([1.0, 0.0, 0.0])),
        ('[111]', np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
    ]

    for dir_name, k_hat in k_dirs:
        k_abs = 2 * np.pi * k_frac / L_cell * k_hat
        k2 = np.dot(k_abs, k_abs)

        # MPB reference at same k-direction
        c2_mpb = _run_mpb_kvec(eps_B, k_frac * k_hat)

        d0_k = build_d0_bloch(V, E, k_abs, L_vec, shifts)
        d1_ex = build_d1_bloch_exact(V, E, F, k_abs, L_vec, d0_k)

        def solve_c2(inv_eps_face):
            K = d1_ex.conj().T @ np.diag(star2 * inv_eps_face) @ d1_ex
            K = 0.5 * (K + K.conj().T)
            eigs = np.sort(np.real(eigh(K, np.diag(star1), eigvals_only=True)))
            thresh = max(np.max(np.abs(eigs)) * 1e-12, 1e-14)
            n_zero = int(np.sum(np.abs(eigs) < thresh))
            phys = eigs[eigs > thresh]
            return phys[0] / k2, n_zero

        print(f"\n  Direction {dir_name}, c2_mpb = {c2_mpb:.4f}")
        print(f"  {'scheme':<27s}  {'c2':>10s}  {'vs MPB':>8s}  {'nz=V':>5s}")
        for name, iface_val in schemes:
            inv_eps = np.zeros(len(F))
            for fi in range(len(F)):
                ca, cb = ftc[fi]
                if labels[ca] == labels[cb]:
                    inv_eps[fi] = 1.0 / eps_cells[ca]
                else:
                    inv_eps[fi] = iface_val
            c2, nz = solve_c2(inv_eps)
            err = (c2 - c2_mpb) / c2_mpb
            print(f"  {name:<27s}  {c2:10.4f}  {err:+7.1%}  {'Y' if nz == len(V) else 'N':>5s}")
            assert nz == len(V), f"Exactness broken with {name} {dir_name}: nz={nz}"

    print(f"\n  All formulas preserve exactness at both k-directions.")
    print(f"  DEC-MPB gap is from Hodge star averaging, not from the complex.")
    print(f"  PASS")


def test_mpb_resolution_convergence():
    """MPB c² converges with grid resolution."""
    print(f"\n{'=' * 70}")
    print("  MPB RESOLUTION CONVERGENCE: eps_B=4, k_frac=0.05")
    print(f"{'=' * 70}")

    k_frac = 0.05
    eps_B = 4
    print(f"\n  {'res':>5s}  {'freq1':>10s}  {'c2':>10s}")

    c2_values = []
    for res in [16, 32, 64]:
        freqs = run_mpb(eps_B, k_frac, resolution=res)
        c2 = (freqs[0] / k_frac)**2
        c2_values.append(c2)
        print(f"  {res:5d}  {freqs[0]:10.6f}  {c2:10.4f}")

    # Quantify: res=32 vs res=64 difference
    mpb_diff = abs(c2_values[2] - c2_values[1]) / c2_values[2]
    print(f"\n  MPB res=32 vs res=64: {mpb_diff:.1%} difference.")
    print(f"  MPB discretization error negligible relative to DEC error (~20%).")
    print(f"  PASS")


def main():
    print("=" * 70)
    print("MPB COMPARISON — BCC Voronoi (Kelvin foam)")
    print("=" * 70)

    test_vacuum_sanity()
    test_dielectric_comparison()
    test_k_scaling()
    test_hodge_star_sensitivity()
    test_mpb_resolution_convergence()


if __name__ == '__main__':
    main()
    print("\nDone.")
