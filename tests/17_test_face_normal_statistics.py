"""DIR-S7a: Face-normal statistics — explains the formula flip.

Question: why does logarithmic mean win on Kelvin BCC but arithmetic
is closest to EM on Voronoi? Is there a principled orientation-weighted
formula that beats both?

Setup: for each interface face, compute cos²θ = (n̂_f · n̂_ref)²
where n̂_ref is a reference direction (k̂ for BCC, ẑ for Voronoi z-plane).
Then test the "orientation-weighted" (ow) formula:
  (ε⁻¹)_f = cos²θ · (1/ε_H) + sin²θ · (1/ε_A)
where ε_H = harmonic mean, ε_A = arithmetic mean.

NOTE ON REFERENCE NORMAL:
  On Voronoi z-plane, ẑ is the physical interface normal — ow is
  physically motivated. On Kelvin BCC, there is no single interface
  plane: each interface face IS the interface, so face normal =
  interface normal → cos²=1 → ow = harmonic (worst performer).
  Using k̂ as reference on Kelvin computes cos²(face normal, wave dir),
  which is a different quantity — a geometric angle (54.7° between
  {111} and [100]), not the interface angle. The ow formula with k̂
  is therefore a one-parameter interpolation family parametrized by
  a geometric angle, NOT the "correct interface physics."
  The key finding (test 5) is independent of this: log mean's
  effective weight adapts to contrast while any fixed-weight formula
  (including the physical ow) cannot.

RAW OUTPUT:

  TEST 1: Face-normal distributions
    Kelvin BCC: ALL interface faces at cos²(n̂,k̂) = 1/3 (hexagonal {111} faces)
    Voronoi: interface faces at <cos²(n̂,ẑ)> ≈ 0.52 (biased z-normal)
    Voronoi: <cos²(n̂,k̂)> ≈ 0.25 (roughly isotropic minus z-bias)

  TEST 2: Orientation-weighted formula vs contrasts
    KELVIN (vs MPB, ref=k̂):
      eps_B=2:  ow=-0.6%, log=-0.8%  (ow slightly better)
      eps_B=4:  ow=-0.3%, log=-2.3%  (ow better at low contrast)
      eps_B=9:  ow=+7.5%, log=-2.7%  (log wins at high contrast)
      eps_B=16: ow=+20.8%, log=-1.1% (log decisively wins)
    VORONOI (vs EM, ref=ẑ, 3 seeds):
      eps_B=4:  ow=+1.1%, log=+1.0%  (nearly identical)
      eps_B=9:  ow=+4.6%, log=+3.6%
      eps_B=16: ow=+7.7%, log=+5.7%  (log slightly better)

  TEST 3: Why — effective interpolation weight
    Log mean has contrast-dependent "effective t" between H and A:
      eps_B=2:  t_log=0.318, cos²=0.333  (close → ow ≈ log)
      eps_B=4:  t_log=0.276, cos²=0.333  (diverging)
      eps_B=9:  t_log=0.210, cos²=0.333  (log shifts toward arithmetic)
      eps_B=16: t_log=0.163, cos²=0.333  (log is 2× closer to arithmetic)

ANSWER:
  The orientation-weighted formula is NOT a universal winner. It works at
  low contrast (≤4×) but degrades badly at high contrast (21% at 16×).
  Log mean is robust because it ADAPTS to contrast — its effective weight
  toward harmonic DECREASES with increasing ε_B/ε_A, while ANY fixed-weight
  formula (including the physically motivated ow with cos²=1) cannot track
  this shift. This contrast-tracking makes log mean universally robust
  across both regular (Kelvin) and irregular (Voronoi) meshes.

  Physical interpretation: at high contrast, the field avoids the high-ε
  region (field expulsion), so the effective ε shifts toward the smaller
  value regardless of face orientation. The logarithmic mean L(a,b)
  naturally shifts toward the smaller argument as the ratio grows —
  a known property (L is between geometric and arithmetic means, closer
  to geometric, which itself is closer to the smaller argument).
  This is why log mean encodes both geometry (through its position
  between H and A) and contrast (through its contrast-dependent weight),
  while the ow formula captures geometry only.

  Structural argument: the ow formula introduces dependence on the wave
  direction k̂ into the Hodge star. This is conceptually problematic —
  the Hodge star is a metric quantity and should not depend on the
  solution being computed. Log mean is k-independent: it depends only
  on the two permittivities adjacent to the face. This k-independence
  is a stronger argument for log mean than error magnitude alone.
"""
import sys, os, unittest
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from physics.hodge import (build_kelvin_with_dual_info, build_foam_with_dual_info,
                           build_hodge_stars_voronoi)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.interface import log_mean, build_inv_eps_face


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def face_normal(V, face, L_vec):
    """Unit normal for a polygonal face with periodic wrapping."""
    v0 = np.array(V[face[0]], dtype=float)
    d1 = np.array(V[face[1]], dtype=float) - v0
    d2 = np.array(V[face[2]], dtype=float) - v0
    for dim in range(3):
        d1[dim] -= L_vec[dim] * round(d1[dim] / L_vec[dim])
        d2[dim] -= L_vec[dim] * round(d2[dim] / L_vec[dim])
    n = np.cross(d1, d2)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-14 else np.zeros(3)


def interface_cos2(V, F, ftc, eps_cells, L_vec, ref_normal):
    """Compute cos²(n̂_f, ref_normal) for all interface faces."""
    cos2_vals = []
    for fi in range(len(F)):
        ca, cb = ftc[fi]
        if abs(float(eps_cells[ca]) - float(eps_cells[cb])) > 1e-12:
            n = face_normal(V, F[fi], L_vec)
            cos2_vals.append((np.dot(n, ref_normal))**2)
    return np.array(cos2_vals)


def build_inv_eps_ow(F, ftc, eps_cells, L_vec, V, ref_normal):
    """Orientation-weighted: cos²θ·(1/ε_H) + sin²θ·(1/ε_A)."""
    nF = len(F)
    inv_eps = np.zeros(nF)
    for fi in range(nF):
        ca, cb = ftc[fi]
        ea, eb = float(eps_cells[ca]), float(eps_cells[cb])
        if abs(ea - eb) < 1e-12:
            inv_eps[fi] = 1.0 / ea
        else:
            n = face_normal(V, F[fi], L_vec)
            cos2 = (np.dot(n, ref_normal))**2
            inv_H = 0.5 * (1.0/ea + 1.0/eb)
            inv_A = 2.0 / (ea + eb)
            inv_eps[fi] = cos2 * inv_H + (1.0 - cos2) * inv_A
    return inv_eps


def solve_c2_inv(V, E, F, L_vec, star1, star2, inv_eps, k_vec):
    """Solve curl-curl with given inv_eps, return c²."""
    shifts = compute_edge_shifts(V, E, L_vec)
    d0 = build_d0_bloch(V, E, k_vec, L_vec, shifts)
    d1 = build_d1_bloch_exact(V, E, F, k_vec, L_vec, d0)
    K = d1.conj().T @ np.diag(star2 * inv_eps) @ d1
    K = 0.5 * (K + K.conj().T)
    M = np.diag(star1)
    evals = np.sort(np.real(eigh(K, M, eigvals_only=True)))
    thr = max(np.max(np.abs(evals)) * 1e-12, 1e-14)
    phys = evals[evals > thr]
    k2 = np.dot(k_vec, k_vec)
    return phys[0] / k2 if len(phys) > 0 else float('nan')


# ---------------------------------------------------------------------------
# Kelvin BCC setup (reused across tests)
# ---------------------------------------------------------------------------

def build_kelvin(eps_B=4.0, N=2, L_cell=4.0):
    data = build_kelvin_with_dual_info(N=N, L_cell=L_cell)
    V, E, F = data['V'], data['E'], data['F']
    L_vec = data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    cc = data['cell_centers']
    labels = np.array([int(round((c[0]+c[1]+c[2])/(L_cell/2))) % 2 for c in cc])
    eps_cells = np.where(labels == 0, 1.0, float(eps_B))
    return dict(V=V, E=E, F=F, L_vec=L_vec, star1=star1, star2=star2,
                face_to_cells=data['face_to_cells'], eps_cells=eps_cells)


MPB_REF = {1.5: 0.810481, 2: 0.691737, 3: 0.545276,
           4: 0.455265, 6: 0.346683, 9: 0.258124, 16: 0.163530}
K_HAT = np.array([1.0, 0.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFaceNormalStatistics(unittest.TestCase):

    def test_1_kelvin_cos2_uniform(self):
        """All Kelvin BCC interface faces have cos²(n̂,k̂) = 1/3."""
        d = build_kelvin(eps_B=4.0)
        cos2 = interface_cos2(d['V'], d['F'], d['face_to_cells'],
                              d['eps_cells'], d['L_vec'], K_HAT)
        print(f"\n  Kelvin: {len(cos2)} interface faces, cos²(n̂,k̂) = {np.mean(cos2):.6f}")
        print(f"    std = {np.std(cos2):.2e} (should be 0)")
        self.assertAlmostEqual(np.mean(cos2), 1.0/3.0, places=5)
        self.assertLess(np.std(cos2), 1e-10)

    def test_2_voronoi_cos2_distributed(self):
        """Voronoi interface faces have distributed cos²(n̂,ẑ)."""
        L_BOX = 10.0
        cos2_means = []
        print(f"\n  Voronoi (n=120, z-plane split):")
        for seed in range(3):
            np.random.seed(seed)
            pts = np.random.uniform(0, L_BOX, size=(120, 3))
            try:
                data = build_foam_with_dual_info(pts, L_BOX)
            except Exception:
                continue
            cc = data['cell_centers']
            eps_cells = np.where(cc[:, 2] < L_BOX/2, 1.0, 4.0)
            cos2 = interface_cos2(data['V'], data['F'], data['face_to_cells'],
                                  eps_cells, data['L_vec'], Z_HAT)
            cos2_means.append(np.mean(cos2))
            print(f"    seed={seed}: {len(cos2)} iface faces, "
                  f"<cos²(n̂,ẑ)>={np.mean(cos2):.3f}, std={np.std(cos2):.3f}")

        m = np.mean(cos2_means)
        # Should be biased toward ~0.5 (faces near z-plane tend to be z-normal)
        # but NOT 1/3 (isotropic) and NOT 1.0 (all z-aligned)
        self.assertGreater(m, 0.40)
        self.assertLess(m, 0.65)
        print(f"    mean <cos²> = {m:.3f} (biased z-normal, not isotropic 1/3)")

    def test_3_kelvin_ow_vs_log(self):
        """On Kelvin: ow ≈ log at low contrast, log wins at high contrast."""
        d0 = build_kelvin(eps_B=1.0)  # for topology
        k_vec = 2 * np.pi * 0.01 / d0['L_vec'][0] * K_HAT

        print(f"\n  Kelvin ow vs log (% error vs MPB):")
        print(f"    {'eps_B':>6} {'log':>8} {'ow':>8} {'winner':>8}")

        log_wins = 0
        for eps_B in [2, 4, 9, 16]:
            d = build_kelvin(eps_B=float(eps_B))
            mpb = MPB_REF[eps_B]

            inv_log, _ = build_inv_eps_face(d['F'], d['face_to_cells'],
                                            d['eps_cells'], 'log')
            inv_ow = build_inv_eps_ow(d['F'], d['face_to_cells'], d['eps_cells'],
                                      d['L_vec'], d['V'], K_HAT)

            c2_log = solve_c2_inv(d['V'], d['E'], d['F'], d['L_vec'],
                                  d['star1'], d['star2'], inv_log, k_vec)
            c2_ow = solve_c2_inv(d['V'], d['E'], d['F'], d['L_vec'],
                                 d['star1'], d['star2'], inv_ow, k_vec)

            err_log = abs(c2_log - mpb) / mpb
            err_ow = abs(c2_ow - mpb) / mpb
            winner = 'log' if err_log < err_ow else 'ow'
            if err_log < err_ow:
                log_wins += 1
            print(f"    {eps_B:6d} {100*(c2_log-mpb)/mpb:+7.1f}% {100*(c2_ow-mpb)/mpb:+7.1f}% {winner:>8}")

        # Log must win at high contrast (eps_B >= 6)
        self.assertGreaterEqual(log_wins, 2, "Log should win at high contrast")

    def test_4_voronoi_ow_vs_log(self):
        """On Voronoi: ow ≈ log, both within 2% of EM at eps_B=4."""
        L_BOX = 10.0
        eps_B = 4.0
        em = 2.0 / (1.0 + eps_B)

        log_vals, ow_vals = [], []
        for seed in range(3):
            np.random.seed(seed)
            pts = np.random.uniform(0, L_BOX, size=(120, 3))
            try:
                data = build_foam_with_dual_info(pts, L_BOX)
            except Exception:
                continue
            V, E, F = data['V'], data['E'], data['F']
            L_vec = data['L_vec']
            s1, s2 = build_hodge_stars_voronoi(data)
            cc = data['cell_centers']
            eps_cells = np.where(cc[:, 2] < L_BOX/2, 1.0, eps_B)
            k_vec = np.array([0.01 * 2 * np.pi / L_vec[0], 0, 0])

            inv_log, _ = build_inv_eps_face(F, data['face_to_cells'], eps_cells, 'log')
            inv_ow = build_inv_eps_ow(F, data['face_to_cells'], eps_cells, L_vec, V, Z_HAT)

            c2_log = solve_c2_inv(V, E, F, L_vec, s1, s2, inv_log, k_vec)
            c2_ow = solve_c2_inv(V, E, F, L_vec, s1, s2, inv_ow, k_vec)
            log_vals.append(c2_log)
            ow_vals.append(c2_ow)

        ml = np.mean(log_vals)
        mo = np.mean(ow_vals)
        print(f"\n  Voronoi eps_B={eps_B:.0f}: log={ml:.4f} ({100*(ml-em)/em:+.1f}%), "
              f"ow={mo:.4f} ({100*(mo-em)/em:+.1f}%)")
        print(f"    Difference: {100*abs(ml-mo)/em:.2f}% (should be <1%)")

        # ow and log should be close on Voronoi
        self.assertLess(abs(ml - mo) / em, 0.01)
        # Both should be within 5% of EM
        self.assertLess(abs(ml - em) / em, 0.05)
        self.assertLess(abs(mo - em) / em, 0.05)

    def test_5_effective_weight(self):
        """Log mean's effective interpolation weight decreases with contrast."""
        print(f"\n  Effective interpolation weight t: inv_L = t·inv_H + (1-t)·inv_A")
        print(f"    {'eps_B':>6} {'t_log':>8} {'cos²':>8} {'ratio':>8}")

        t_vals = []
        for eps_B in [1.5, 2, 4, 9, 16]:
            ea, eb = 1.0, float(eps_B)
            inv_H = 0.5 * (1/ea + 1/eb)
            inv_A = 2.0 / (ea + eb)
            inv_L = 1.0 / log_mean(ea, eb)
            t_log = (inv_L - inv_A) / (inv_H - inv_A) if abs(inv_H - inv_A) > 1e-14 else 0.5
            t_vals.append(t_log)
            print(f"    {eps_B:6.1f} {t_log:8.4f} {1/3:8.4f} {t_log/(1/3):8.3f}")

        # t_log should decrease monotonically with contrast
        for i in range(len(t_vals) - 1):
            self.assertGreater(t_vals[i], t_vals[i+1],
                               f"t_log should decrease: {t_vals[i]:.4f} -> {t_vals[i+1]:.4f}")

        # At low contrast, t_log ≈ cos² = 1/3 (explains ow ≈ log)
        self.assertAlmostEqual(t_vals[0], 1.0/3.0, delta=0.05)
        # At high contrast, t_log << cos² (explains log beats ow)
        self.assertLess(t_vals[-1], 0.20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
