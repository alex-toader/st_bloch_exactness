"""
R2-3/R2-6: Theorem — exactness implies ker(K) = im(d0)

Statement:
  If d1(k)d0(k) = 0 and d0(k) has rank V (generic k != 0), then:
    (i)   im(d0) = ker(d1)           [exactness + dimension count]
    (ii)  ker(K) = im(d0)            [K = d1† star2 d1]
    (iii) physical eigvecs M-perp to im(d0)  [gauge/physical separation]

  Conversely, if d1 d0 != 0, then im(d0) is NOT contained in ker(d1),
  gradient modes leak into the physical spectrum (spurious eigenvalues).

Proof sketch:
  1. d1 d0 = 0 => im(d0) subset ker(d1), so dim ker(d1) >= rank(d0).
  2. Proposition (separate): For k != 0, Bloch periodicity eliminates constant
     0-forms, hence ker(d0(k)) = 0 and rank(d0) = V (d0 injective).
     Therefore dim ker(d1) >= V.
  3. Upper bound on dim ker(d1): requires dim ker(d1) <= V. This is the
     non-trivial step — equivalent to H^1(T^3, L_k) = 0 for the local
     coefficient system defined by k. See paper Prop 5 for the cohomological
     argument (Kunneth on T^3 with acyclic L_k for generic k != 0).
     [NOTE: Cannot derive this from Euler characteristic alone without
     accounting for d2 and 3-cells. The sketch V-E+F=0 is WRONG on T^3;
     the correct relation is V-E+F-C=0.]
  4. Steps 2+3: dim ker(d1) = V = dim im(d0), combined with (1) => im(d0) = ker(d1).
  5. ker(K) = ker(d1) (since star2 > 0 entrywise), so ker(K) = im(d0).
     Physical eigvecs (lambda > 0) are M-orthogonal to ker(K) = im(d0). QED.

Paper actions:
  - Proposition: "For k != 0, ker(d0(k)) = 0" (separate, non-trivial for referees)
  - Theorem: the chain above. Step 3 requires Prop 5 (cohomological, not avoidable).
  - Optional lemma: C^1 = im(d0) +_M ker(K)^perp (Hodge splitting)

RAW OUTPUT:

  Kelvin N=2 [100]: 4 k-fracs, ALL OK. rank_d0=ker_d1=ker_K=96, max_grad<3e-12
  Kelvin N=2 [111]: 4 k-fracs, ALL OK. rank_d0=ker_d1=ker_K=96, max_grad<3e-12
  C15 N=1 [100]:    4 k-fracs, ALL OK. rank_d0=ker_d1=ker_K=136, max_grad<3e-12
  WP N=1 [100]:     4 k-fracs, ALL OK. rank_d0=ker_d1=ker_K=46, max_grad<1e-12
  SC N=3 [100]:     4 k-fracs, ALL OK. rank_d0=ker_d1=ker_K=27, max_grad<8e-13

  Standard (broken): Kelvin N=2, frac=0.05
    im(d0) NOT subset ker(d1): ||d1d0|| = 3.96
    ker_K = 90 != rank_d0 = 96 (6 modes expelled)
    8 physical eigvecs with grad% > 0.5 (gradient leakage)

ANSWER:
  Theorem verified on all 5 structures, 2 directions, 4 k-fractions each.
  Chain im(d0) = ker(d1) = ker(K) holds exactly (to machine precision).
  Physical eigenvectors have zero gradient component (max < 10^{-8}).
  Standard construction violates every link in the chain.
"""
import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import (build_kelvin_with_dual_info, build_hodge_stars_voronoi,
                            build_c15_with_dual_info, build_wp_with_dual_info)
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup
from core_math.builders.solids_periodic import build_sc_supercell_periodic


def verify_exact_chain(name, V, E, F, L, L_vec, star1, star2,
                       fracs, k_hat):
    """Verify im(d0) = ker(d1) = ker(K) and physical M-perp im(d0)."""
    shifts = compute_edge_shifts(V, E, L_vec)
    M = np.diag(star1)
    nV = len(V)
    all_ok = True

    print(f"{name}: V={nV} E={len(E)} F={len(F)}")
    print(f"  {'frac':>6s}  {'||d1d0||':>10s}  {'rk_d0':>6s}  {'ker_d1':>7s}"
          f"  {'ker_K':>6s}  {'max_g':>8s}  {'ok':>4s}")

    for frac in fracs:
        k = (2 * np.pi / L) * frac * k_hat
        d0 = build_d0_bloch(V, E, k, L_vec, shifts)
        d1 = build_d1_bloch_exact(V, E, F, k, L_vec, d0)

        norm_d1d0 = np.linalg.norm(d1 @ d0)
        rank_d0 = np.linalg.matrix_rank(d0)
        dim_ker_d1 = len(E) - np.linalg.matrix_rank(d1)

        K = d1.conj().T @ np.diag(star2) @ d1
        K = 0.5 * (K + K.conj().T)
        eigs, vecs = eigh(K, M)
        eigs = np.real(eigs)
        idx = np.argsort(eigs)
        eigs = eigs[idx]
        vecs = vecs[:, idx]
        thresh = max(np.max(np.abs(eigs)) * 1e-12, 1e-14)
        dim_ker_K = int(np.sum(np.abs(eigs) < thresh))

        # Max gradient fraction of physical eigenvectors (M-weighted)
        phys = vecs[:, np.abs(eigs) > thresh]
        Md0 = M @ d0
        Pg = d0 @ np.linalg.solve(d0.conj().T @ Md0, Md0.conj().T)
        max_g = 0.0
        for j in range(phys.shape[1]):
            v = phys[:, j]
            Pv = Pg @ v
            g = np.sqrt(abs(np.real(Pv.conj() @ M @ Pv) /
                            np.real(v.conj() @ M @ v)))
            max_g = max(max_g, g)

        ok = (rank_d0 == dim_ker_d1 == dim_ker_K) and max_g < 1e-8
        if not ok:
            all_ok = False
        print(f"  {frac:6.2f}  {norm_d1d0:10.2e}  {rank_d0:6d}  {dim_ker_d1:7d}"
              f"  {dim_ker_K:6d}  {max_g:8.2e}  {'YES' if ok else 'NO':>4s}")

    print(f"  ALL OK: {all_ok}")
    print()
    return all_ok


def verify_standard_failure(N=2, frac=0.05, k_hat=np.array([1., 0., 0.])):
    """Show that standard construction violates every link in the chain."""
    data = build_kelvin_with_dual_info(N=N, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)
    cr = compute_edge_crossings(V, E, L)
    el = build_edge_lookup(E, cr)
    M = np.diag(star1)

    k = (2 * np.pi / L) * frac * k_hat
    d0 = build_d0_bloch(V, E, k, L_vec, shifts)
    d1s = build_d1_bloch_standard(V, E, F, L, k, el, cr)

    norm_d1d0 = np.linalg.norm(d1s @ d0)
    rank_d0 = np.linalg.matrix_rank(d0)
    dim_ker_d1 = len(E) - np.linalg.matrix_rank(d1s)

    Ks = d1s.conj().T @ np.diag(star2) @ d1s
    Ks = 0.5 * (Ks + Ks.conj().T)
    eigs, vecs = eigh(Ks, M)
    eigs = np.real(eigs)
    idx = np.argsort(eigs)
    eigs = eigs[idx]
    vecs = vecs[:, idx]
    thresh = max(np.max(np.abs(eigs)) * 1e-12, 1e-14)
    dim_ker_K = int(np.sum(np.abs(eigs) < thresh))

    phys = vecs[:, np.abs(eigs) > thresh]
    Md0 = M @ d0
    Pg = d0 @ np.linalg.solve(d0.conj().T @ Md0, Md0.conj().T)
    n_above_half = 0
    max_g = 0.0
    for j in range(phys.shape[1]):
        v = phys[:, j]
        Pv = Pg @ v
        g = np.sqrt(abs(np.real(Pv.conj() @ M @ Pv) /
                        np.real(v.conj() @ M @ v)))
        max_g = max(max_g, g)
        if g > 0.5:
            n_above_half += 1

    print(f"Standard (Kelvin N={N}, frac={frac}):")
    print(f"  ||d1d0|| = {norm_d1d0:.4f}  (should be 0)")
    print(f"  rank(d0) = {rank_d0}  dim ker(d1) = {dim_ker_d1}"
          f"  => im(d0) ⊆ ker(d1): {norm_d1d0 < 1e-10}")
    print(f"  dim ker(K) = {dim_ker_K}  (expect {rank_d0},"
          f" deficit {rank_d0 - dim_ker_K})")
    print(f"  Physical modes with grad% > 0.5: {n_above_half}"
          f"  max grad% = {max_g:.4f}")
    print(f"  => Every link in the chain BROKEN")
    print()


if __name__ == '__main__':
    print("R2-3/R2-6: Theorem verification — ker(K) = im(d0)")
    print("=" * 70)

    fracs = [0.02, 0.05, 0.10, 0.20]

    # Exact construction: theorem holds on all structures
    print("\n--- Exact construction ---\n")

    d = build_kelvin_with_dual_info(N=2, L_cell=4.0)
    s1, s2 = build_hodge_stars_voronoi(d)
    verify_exact_chain('Kelvin N=2 [100]', d['V'], d['E'], d['F'],
                       d['L'], d['L_vec'], s1, s2, fracs,
                       np.array([1., 0., 0.]))
    verify_exact_chain('Kelvin N=2 [111]', d['V'], d['E'], d['F'],
                       d['L'], d['L_vec'], s1, s2, fracs,
                       np.array([1., 1., 1.]) / np.sqrt(3))

    d = build_c15_with_dual_info(N=1, L_cell=10.0)
    s1, s2 = build_hodge_stars_voronoi(d)
    verify_exact_chain('C15 N=1 [100]', d['V'], d['E'], d['F'],
                       d['L'], d['L_vec'], s1, s2, fracs,
                       np.array([1., 0., 0.]))

    d = build_wp_with_dual_info(N=1, L_cell=10.0)
    s1, s2 = build_hodge_stars_voronoi(d)
    verify_exact_chain('WP N=1 [100]', d['V'], d['E'], d['F'],
                       d['L'], d['L_vec'], s1, s2, fracs,
                       np.array([1., 0., 0.]))

    V, E, F, _ = build_sc_supercell_periodic(3)
    L = 6.0
    L_vec = np.array([L, L, L])
    s1 = np.full(len(E), 2.0)
    s2 = np.full(len(F), 2.0)
    verify_exact_chain('SC N=3 [100]', V, E, F, L, L_vec, s1, s2,
                       fracs, np.array([1., 0., 0.]))

    # Standard construction: theorem fails
    print("--- Standard construction (counterexample) ---\n")
    verify_standard_failure()

    print("Done.")
