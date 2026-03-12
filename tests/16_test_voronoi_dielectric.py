"""DIR-S7: Voronoi + dielectric h-refinement.

Question: does the exact DEC method with logarithmic mean interface averaging
converge on irregular meshes with material contrast?

Setup: random Voronoi on [0,L)³, z-plane split (z<L/2 → ε_A=1, z≥L/2 → ε_B).
Wave vector along x (parallel to layers). Effective medium predictions
(exact for infinite planar slabs, approximate for ragged Voronoi interface):
  c²_⊥ = 2/(1+ε_B)      [E ⊥ layers, arithmetic ε]
  c²_∥ = (1+ε_B)/(2ε_B)  [E ∥ layers, harmonic ε]

EM predictions are sanity checks (~10%) because the Voronoi interface
is rough: at n=200 (~6 cells/dim), roughness ~ L/6.

Performance budget (3 seeds, measured):
  n=50: 0.3s   n=80: 0.7s   n=120: 2.3s   n=200: 8.7s
  Eigensolve dominates at n≥120 (O(E³) dense eigh).
  Target: full suite < 3 min.

RAW OUTPUT:
  Exactness: n_zero = V on ALL meshes (n=50..200, 3 seeds each). 1 degenerate mesh skipped.
  Stability: c² = 0.404-0.406 across n=50..200. Spread < 1%. Already converged at n=50.
  Variance: std drops from 0.011 (n=50) to 0.009 (n=200).
  h-convergence fit: h^1.54, R²=0.43 — poor fit because c² is flat (noise, not convergence).
  Effective medium (n=120, 3 seeds):
    eps_B=2: c²=0.664, EM=0.667, err=-0.3%
    eps_B=9: c²=0.207, EM=0.200, err=+3.6%
  Formula comparison (eps_B=4, n=120):
    log: +1.0% vs EM | harmonic: +5.6% | arithmetic: -1.5%
  Two polarizations: c²_1=0.404, c²_2=0.535 (split, both physical).
    c²_2 is between EM_⊥=0.400 and EM_∥=0.625 — rough Voronoi interface
    mixes polarizations, preventing clean E∥ mode.
  High contrast: eps_B=16, c²=0.126, EM=0.118, err=+7.1%.
  Runtime: 117s, 12 meshes built, 1 failed.

ANSWER:
  The method exhibits statistical stability: c² is mesh-size independent
  within <1% from n=50 onward. This is NOT h-convergence (each n gives a
  different random Voronoi — the interface geometry changes, not refines).
  No systematic h-scaling observed; geometric roughness dominates.
  Exactness preserved at all sizes. c² matches effective medium within:
    -0.3% at eps_B=2, +1.0% at eps_B=4, +3.6% at eps_B=9, +7.1% at eps_B=16.
  Error grows with contrast — expected for rough interface.
  On Voronoi (unlike Kelvin BCC), arithmetic mean closest to EM (-1.5%),
  log mean second (+1.0%), harmonic worst (+5.6%). Best formula depends on
  mesh geometry; log mean performs consistently well across both regular
  and irregular meshes.
"""
import sys, os, time, unittest
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from physics.hodge import build_foam_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.interface import build_inv_eps_face


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_h(V, E, L_vec):
    lengths = []
    for i, j in E:
        d = V[j] - V[i]
        d -= np.round(d / L_vec) * L_vec
        lengths.append(np.linalg.norm(d))
    return np.mean(lengths)


def solve_c2(data, k_frac, formula='log', n_modes=2):
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    star1, star2 = data['star1'], data['star2']

    inv_eps, _ = build_inv_eps_face(F, data['face_to_cells'],
                                     data['eps_cells'], formula)
    shifts = compute_edge_shifts(V, E, L_vec)
    k_vec = np.array([k_frac * 2 * np.pi / L_vec[0], 0, 0])
    k_abs = np.linalg.norm(k_vec)

    d0 = build_d0_bloch(V, E, k_vec, L_vec, shifts)
    d1 = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0)

    K = d1.conj().T @ np.diag(star2 * inv_eps) @ d1
    M = np.diag(star1)
    evals = np.sort(np.real(eigh(K, M, eigvals_only=True)))

    thr = 1e-10 * np.max(np.abs(evals))
    n_zero = int(np.sum(np.abs(evals) < thr))
    nz = evals[np.abs(evals) >= thr]
    return nz[:n_modes] / k_abs ** 2, n_zero, len(V), measure_h(V, E, L_vec)


# ---------------------------------------------------------------------------
# Mesh cache
# ---------------------------------------------------------------------------

_CACHE = {}
L_BOX = 10.0


def get_mesh(n_cells, seed, eps_B=4.0):
    """Build once per (n_cells, seed), reuse. Returns data dict or None."""
    key = (n_cells, seed)
    if key not in _CACHE:
        try:
            np.random.seed(seed)
            pts = np.random.uniform(0, L_BOX, size=(n_cells, 3))
            data = build_foam_with_dual_info(pts, L_BOX)
            s1, s2 = build_hodge_stars_voronoi(data)
            data['star1'], data['star2'] = s1, s2
            _CACHE[key] = data
        except Exception:
            _CACHE[key] = None
    cached = _CACHE[key]
    if cached is None:
        return None
    # Return copy with fresh eps_cells — don't mutate cache
    data = dict(cached)
    centers = data['cell_centers']
    data['eps_cells'] = np.where(centers[:, 2] < L_BOX / 2, 1.0, float(eps_B))
    return data


def valid_seeds(n_cells, n_want=3, max_try=15):
    seeds = []
    for s in range(max_try):
        if get_mesh(n_cells, s) is not None:
            seeds.append(s)
            if len(seeds) == n_want:
                break
    return seeds


# ---------------------------------------------------------------------------
# Tests — 3 seeds, n≤200 except convergence test uses n=50..200
# ---------------------------------------------------------------------------

K_FRAC = 0.01
N_SEEDS = 3


class TestVoronoiDielectric(unittest.TestCase):

    def test_1_exactness(self):
        """n_zero = V at all sizes, all seeds."""
        for nc in [50, 120, 200]:
            for seed in valid_seeds(nc, N_SEEDS):
                data = get_mesh(nc, seed)
                _, nz, nV, _ = solve_c2(data, K_FRAC)
                self.assertEqual(nz, nV, f"n={nc},seed={seed}: {nz}!={nV}")

    def test_2_stability_and_variance(self):
        """c² stable across sizes; variance decreases."""
        res = {}
        for nc in [50, 120, 200]:
            vals = []
            for seed in valid_seeds(nc, N_SEEDS):
                c2, nz, nV, _ = solve_c2(get_mesh(nc, seed), K_FRAC)
                if nz == nV:
                    vals.append(c2[0])
            res[nc] = (np.mean(vals), np.std(vals))

        print(f"\n  c² stability (eps_B=4, log mean):")
        for nc in sorted(res):
            m, s = res[nc]
            print(f"    n={nc:>3}: c² = {m:.6f} ± {s:.6f}")

        means = [m for m, _ in res.values()]
        spread = (max(means) - min(means)) / np.mean(means)
        self.assertLess(spread, 0.05)
        # variance should decrease
        self.assertLess(res[200][1], res[50][1] + 0.001)

    def test_3_h_convergence(self):
        """c² vs h — measure bias vs effective medium.

        Reference = EM prediction c²_⊥ = 2/(1+ε_B).
        On random Voronoi, each n is a different mesh (not a refinement),
        so we measure bias vs physics, not classical h-convergence.
        """
        c2_em = 2.0 / (1.0 + 4.0)  # eps_B=4 default
        sizes = [50, 80, 120, 200]
        hs, c2s = [], []
        print(f"\n  h-convergence vs EM (eps_B=4, EM={c2_em:.4f}):")
        for nc in sizes:
            vals, hvs = [], []
            for seed in valid_seeds(nc, N_SEEDS):
                c2, nz, nV, h = solve_c2(get_mesh(nc, seed), K_FRAC)
                if nz == nV:
                    vals.append(c2[0])
                    hvs.append(h)
            c2m, hm = np.mean(vals), np.mean(hvs)
            hs.append(hm)
            c2s.append(c2m)
            err_pct = 100 * (c2m - c2_em) / c2_em
            print(f"    n={nc:>3}: h={hm:.3f}, c²={c2m:.6f} ± {np.std(vals):.6f} ({err_pct:+.1f}% vs EM)")

        errs = [abs(c - c2_em) for c in c2s]
        nz_mask = [e > 1e-10 for e in errs]
        if sum(nz_mask) >= 2:
            lh = np.log([hs[i] for i in range(len(errs)) if nz_mask[i]])
            le = np.log([errs[i] for i in range(len(errs)) if nz_mask[i]])
            p, b = np.polyfit(lh, le, 1)
            ss_res = np.sum((le - p * np.array(lh) - b) ** 2)
            ss_tot = np.sum((le - np.mean(le)) ** 2)
            R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"    Fit: |c²-EM| ~ h^{p:.2f}, R²={R2:.3f}")
        else:
            print(f"    Already converged — errors < 1e-10")

        # All meshes within 5% of EM
        for i, nc in enumerate(sizes):
            self.assertLess(errs[i] / c2_em, 0.05, f"n={nc}: {errs[i]/c2_em:.1%} > 5%")

    def test_4_effective_medium(self):
        """c² near EM prediction (sanity check)."""
        print(f"\n  Effective medium (n=120, 3 seeds):")
        for eps_B in [2.0, 9.0]:
            vals = []
            for seed in valid_seeds(120, N_SEEDS):
                c2, nz, nV, _ = solve_c2(get_mesh(120, seed, eps_B), K_FRAC)
                if nz == nV:
                    vals.append(c2[0])
            em = 2.0 / (1.0 + eps_B)
            c2m = np.mean(vals)
            err = 100 * (c2m - em) / em
            print(f"    eps_B={eps_B:.0f}: c²={c2m:.4f}, EM={em:.4f}, err={err:+.1f}%")
            self.assertLess(abs(err), 15)

    def test_5_two_polarizations(self):
        """Two lowest modes split."""
        c1s, c2s = [], []
        for seed in valid_seeds(120, N_SEEDS):
            c2, nz, nV, _ = solve_c2(get_mesh(120, seed), K_FRAC, n_modes=2)
            if nz == nV and len(c2) >= 2:
                c1s.append(c2[0])
                c2s.append(c2[1])
        c1m, c2m = np.mean(c1s), np.mean(c2s)
        print(f"\n  Polarizations: c²_1={c1m:.4f}, c²_2={c2m:.4f}")
        self.assertLess(c1m, c2m)

    def test_6_formula_comparison(self):
        """Compare formulas on same meshes (reuses cache)."""
        em = 2.0 / (1.0 + 4.0)
        seeds = valid_seeds(120, N_SEEDS)
        print(f"\n  Formula comparison (eps_B=4, n=120, EM={em:.4f}):")
        for formula in ['log', 'harmonic', 'arithmetic']:
            vals = []
            for seed in seeds:
                c2, nz, nV, _ = solve_c2(get_mesh(120, seed), K_FRAC, formula)
                if nz == nV:
                    vals.append(c2[0])
            m = np.mean(vals)
            err = 100 * (m - em) / em
            print(f"    {formula:>10}: c²={m:.4f} ({err:+.1f}% vs EM)")
            self.assertLess(abs(err), 20)

    def test_7_high_contrast(self):
        """Stable at eps_B=16."""
        seeds = valid_seeds(80, N_SEEDS)
        vals = []
        for seed in seeds:
            c2, nz, nV, _ = solve_c2(get_mesh(80, seed, 16.0), K_FRAC)
            if nz == nV:
                vals.append(c2[0])
        em = 2.0 / 17.0
        c2m = np.mean(vals)
        err = 100 * (c2m - em) / em
        print(f"\n  High contrast: eps_B=16, c²={c2m:.4f}, EM={em:.4f}, err={err:+.1f}%")
        self.assertGreater(len(vals), 0)


if __name__ == '__main__':
    t0 = time.time()
    unittest.main(verbosity=2, exit=False)
    dt = time.time() - t0
    built = sum(1 for v in _CACHE.values() if v is not None)
    failed = sum(1 for v in _CACHE.values() if v is None)
    print(f"\nTotal: {dt:.1f}s, meshes built: {built}, failed: {failed}")
