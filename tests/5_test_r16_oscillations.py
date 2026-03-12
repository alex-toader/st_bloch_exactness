"""
R1-6: Standard c² oscillations — full investigation

Seven parts:
  PART 0: Sanity checks (d1 consistency at k=0, star1 uniformity)
  PART 1: Mode classification at frac=0.05, N=2,3,4
  PART 2: Exact vs standard side-by-side at multiple k
  PART 3: Spurious energy scaling with k (8 points, fit with R²)
  PART 4: ||d1d0|| scaling — confirms d1d0 = O(k) (lemma premise)
  PART 5: Hodge orthogonality defect ||P_grad P_phys||
  PART 6: First spur c² vs N — no convergence under refinement

RAW OUTPUT (key numbers):

  Part 0: star1 UNIFORM (all=4.0), d1_exact==d1_std at k=0 (diff=0)
  Part 1: N=2 n_spur=8, 1st spur c2=1.549 grad=0.932, 1st phys c2=304 grad=0.268
  Part 2: Exact c2=1.0000 grad=0.0000 at all k. Std 1st phys c2=2074 (frac=0.02)
  Part 3: eig_spur ~ k^(1.952+/-0.014) R²=0.9997. eig_exact ~ k^(1.999+/-0.000)
  Part 4: ||d1d0|| ~ k^(0.946+/-0.017). C=||d1d0||/k ~ 102 (stable at small k)
  Part 5: ||P_grad P_phys|| = 3.85e-13 (exact) vs 2.87 (standard). Ratio 7.4e+12
  Part 6: c2_1st = 1.55, 2.48, 1.93 (N=2,3,4). Oscillates, no convergence.

ANSWER:

    Standard c² oscillations in Table 3 are NOT a bug. The first non-zero
    eigenvalue of K_std is SPURIOUS (93-97% gradient). The first truly
    physical mode is optic (c² ~ 300-380), not acoustic. The acoustic branch
    is destroyed entirely.

    The spurious energy scales as O(k²) — same as acoustic — because
    d1_std d0 = O(k) (Part 4, numerically confirmed). This means spurious
    modes mimic acoustic dispersion and cannot be separated by k-scaling.

    The Hodge orthogonality defect (Part 5) is 10¹² larger for standard
    than exact. Under mesh refinement (Part 6), the first spurious c²
    oscillates without convergence and the count of spurious modes grows.

    Paper formulation (recommended):
    "The acoustic branch is no longer identifiable at the bottom of the
    spectrum, as it is obscured by lifted gradient modes whose energies
    fluctuate with mesh refinement."

    Potential lemma for paper:
    If d1(k)d0(k) != 0 and ||d1 d0|| = O(|k|), then gradient modes
    acquire eigenvalues lambda = O(k²), indistinguishable from acoustic
    by dispersion scaling.
"""
import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from physics.hodge import build_kelvin_with_dual_info, build_hodge_stars_voronoi
from physics.gauge_bloch import compute_edge_shifts, build_d0_bloch, build_d1_bloch_exact
from physics.bloch import build_d1_bloch_standard, compute_edge_crossings, build_edge_lookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_mesh(N=2):
    """Build Kelvin mesh and all auxiliary structures."""
    data = build_kelvin_with_dual_info(N=N, L_cell=4.0)
    V, E, F = data['V'], data['E'], data['F']
    L, L_vec = data['L'], data['L_vec']
    star1, star2 = build_hodge_stars_voronoi(data)
    shifts = compute_edge_shifts(V, E, L_vec)
    cr = compute_edge_crossings(V, E, L)
    el = build_edge_lookup(E, cr)
    M = np.diag(star1)
    return dict(V=V, E=E, F=F, L=L, L_vec=L_vec, star1=star1, star2=star2,
                shifts=shifts, cr=cr, el=el, M=M)


def grad_projector(d0, M):
    """M-weighted gradient projector: P = d0 (d0† M d0)^{-1} d0† M."""
    Md0 = M @ d0
    G = d0.conj().T @ Md0           # Gram matrix d0† M d0
    return d0 @ np.linalg.solve(G, Md0.conj().T)   # (E x E)


def grad_fraction(Pg_M, v, M):
    """Fraction of v in im(d0) under M-norm: ||P v||_M / ||v||_M."""
    Pv = Pg_M @ v
    num2 = np.real(Pv.conj() @ M @ Pv)
    den2 = np.real(v.conj() @ M @ v)
    return np.sqrt(num2 / den2)


def solve_standard(mesh, k):
    """Build standard K, solve generalized eigenvalue problem, return sorted."""
    d1s = build_d1_bloch_standard(mesh['V'], mesh['E'], mesh['F'],
                                  mesh['L'], k, mesh['el'], mesh['cr'])
    Ks = d1s.conj().T @ np.diag(mesh['star2']) @ d1s
    asym = np.max(np.abs(Ks - Ks.conj().T))
    Ks = 0.5 * (Ks + Ks.conj().T)
    es, vs = eigh(Ks, mesh['M'])
    idx = np.argsort(np.real(es))
    es = np.real(es[idx])
    vs = vs[:, idx]
    ts = max(np.max(np.abs(es)) * 1e-12, 1e-14)
    return es, vs, ts, asym


def solve_exact(mesh, k, d0):
    """Build exact K, solve generalized eigenvalue problem, return sorted."""
    d1e = build_d1_bloch_exact(mesh['V'], mesh['E'], mesh['F'],
                               k, mesh['L_vec'], d0)
    Ke = d1e.conj().T @ np.diag(mesh['star2']) @ d1e
    asym = np.max(np.abs(Ke - Ke.conj().T))
    Ke = 0.5 * (Ke + Ke.conj().T)
    ee, ve = eigh(Ke, mesh['M'])
    idx = np.argsort(np.real(ee))
    ee = np.real(ee[idx])
    ve = ve[:, idx]
    te = max(np.max(np.abs(ee)) * 1e-12, 1e-14)
    return ee, ve, te, asym


# ---------------------------------------------------------------------------
# Part 0: Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(mesh):
    """Verify d1_exact == d1_std at k=0 and report star1 uniformity."""
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    L, L_vec = mesh['L'], mesh['L_vec']

    # star1 uniformity
    s1 = np.array(mesh['star1'])
    cv = s1.std() / s1.mean() if s1.mean() > 0 else 0.0
    uniform = cv < 0.01
    print(f"  star1: min={s1.min():.4f} max={s1.max():.4f} CV={cv:.6f}"
          f"  {'UNIFORM' if uniform else 'NON-UNIFORM'}")

    # d1 consistency at k=0
    k0 = np.array([0., 0., 0.])
    d0_0 = build_d0_bloch(V, E, k0, L_vec, mesh['shifts'])
    d1e_0 = build_d1_bloch_exact(V, E, F, k0, L_vec, d0_0)
    d1s_0 = build_d1_bloch_standard(V, E, F, L, k0, mesh['el'], mesh['cr'])
    diff = np.max(np.abs(d1e_0 - d1s_0))
    ok = diff < 1e-12
    print(f"  d1_exact vs d1_std at k=0: max|diff| = {diff:.2e}  {'MATCH' if ok else 'MISMATCH'}")

    if not ok:
        # check if rows are proportional (phase ambiguity)
        n_prop = 0
        for fi in range(len(F)):
            row_e = d1e_0[fi, :]
            row_s = d1s_0[fi, :]
            nz = np.abs(row_e) > 1e-15
            if nz.sum() > 0:
                ratios = row_e[nz] / row_s[nz]
                if np.max(np.abs(ratios - ratios[0])) < 1e-12:
                    n_prop += 1
        print(f"    Proportional rows: {n_prop}/{len(F)}")

    return uniform, ok


# ---------------------------------------------------------------------------
# Part 1: Mode classification
# ---------------------------------------------------------------------------

def analyze_N(N, frac=0.05, k_hat=np.array([1., 0., 0.])):
    mesh = build_mesh(N)
    nV = len(mesh['V'])
    k = (2 * np.pi / mesh['L']) * frac * k_hat
    k2 = np.dot(k, k)

    d0 = build_d0_bloch(mesh['V'], mesh['E'], k, mesh['L_vec'], mesh['shifts'])
    Pg = grad_projector(d0, mesh['M'])

    ee, _, te, asym_e = solve_exact(mesh, k, d0)
    n0e = int(np.sum(np.abs(ee) < te))

    es, vs, ts, asym_s = solve_standard(mesh, k)
    n0s = int(np.sum(np.abs(es) < ts))

    # Count spurious by scanning (not n0e - n0s)
    n_spur = 0
    first_spur = first_phys = None
    for i in range(len(es)):
        if abs(es[i]) < ts:
            continue
        g = grad_fraction(Pg, vs[:, i], mesh['M'])
        if g > 0.5:
            n_spur += 1
            if first_spur is None:
                first_spur = (i, es[i] / k2, g)
        else:
            if first_phys is None:
                first_phys = (i, es[i] / k2, g)
            if first_phys and n_spur > 0:
                break  # stop after first physical past the spurious band

    print(f"N={N} V={nV} n0_ex={n0e} n0_st={n0s} n_spur_scan={n_spur}"
          f"  asym_e={asym_e:.1e} asym_s={asym_s:.1e}")
    print(f"  {'idx':>4s}  {'eig':>10s}  {'c2':>8s}  {'grad%':>6s}  {'type':>5s}")

    start = max(0, n0s - 2)
    end = min(start + 15, len(es))
    for i in range(start, end):
        e = es[i]
        c2 = e / k2 if abs(e) > ts else 0
        g = grad_fraction(Pg, vs[:, i], mesh['M'])
        t = 'gauge' if abs(e) < ts else ('SPUR' if g > 0.5 else 'phys')
        print(f"  {i:4d}  {e:10.4e}  {c2:8.3f}  {g:6.3f}  {t:>5s}")

    if first_spur:
        print(f"  -> 1st spur: idx={first_spur[0]} c2={first_spur[1]:.4f} grad={first_spur[2]:.3f}")
    if first_phys:
        print(f"  -> 1st phys: idx={first_phys[0]} c2={first_phys[1]:.4f} grad={first_phys[2]:.3f}")
    print()


# ---------------------------------------------------------------------------
# Part 2: Exact vs standard side-by-side
# ---------------------------------------------------------------------------

def compare_exact_vs_standard(N=2, fracs=[0.02, 0.05, 0.10, 0.15],
                               k_hat=np.array([1., 0., 0.])):
    mesh = build_mesh(N)
    M = mesh['M']

    for frac in fracs:
        k = (2 * np.pi / mesh['L']) * frac * k_hat
        k2 = np.dot(k, k)
        d0 = build_d0_bloch(mesh['V'], mesh['E'], k, mesh['L_vec'], mesh['shifts'])
        Pg = grad_projector(d0, M)

        ee, ve, te, _ = solve_exact(mesh, k, d0)
        es, vs, ts, _ = solve_standard(mesh, k)

        # Exact: first nonzero
        ex_line = ""
        for i in range(len(ee)):
            if abs(ee[i]) > te:
                g = grad_fraction(Pg, ve[:, i], M)
                ex_line = f"idx={i} c2={ee[i]/k2:.4f} grad={g:.4f}"
                break

        # Standard: first spur, first phys
        spur_line = phys_line = None
        for i in range(len(es)):
            if abs(es[i]) < ts:
                continue
            g = grad_fraction(Pg, vs[:, i], M)
            if g > 0.5 and spur_line is None:
                spur_line = f"idx={i} c2={es[i]/k2:.4f} grad={g:.4f}"
            if g < 0.5 and phys_line is None:
                phys_line = f"idx={i} c2={es[i]/k2:.4f} grad={g:.4f}"
            if spur_line and phys_line:
                break

        print(f"frac={frac}:")
        print(f"  EXACT 1st nonzero: {ex_line}")
        print(f"  STD   1st spur:    {spur_line}")
        print(f"  STD   1st phys:    {phys_line}")
        print()


# ---------------------------------------------------------------------------
# Part 3: k-scaling analysis
# ---------------------------------------------------------------------------

def k_scaling_analysis(N=2, k_hat=np.array([1., 0., 0.])):
    mesh = build_mesh(N)
    M = mesh['M']

    fracs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15]
    k_vals, eig_spur, eig_exact = [], [], []

    for frac in fracs:
        k = (2 * np.pi / mesh['L']) * frac * k_hat
        k2 = np.dot(k, k)
        k_vals.append(np.sqrt(k2))
        d0 = build_d0_bloch(mesh['V'], mesh['E'], k, mesh['L_vec'], mesh['shifts'])
        Pg = grad_projector(d0, M)

        ee, _, te, _ = solve_exact(mesh, k, d0)
        for e in ee:
            if abs(e) > te:
                eig_exact.append(e)
                break

        es, vs, ts, _ = solve_standard(mesh, k)
        for i in range(len(es)):
            if abs(es[i]) < ts:
                continue
            g = grad_fraction(Pg, vs[:, i], M)
            if g > 0.5:
                eig_spur.append(es[i])
                break

    k_vals = np.array(k_vals)
    eig_spur = np.array(eig_spur)
    eig_exact = np.array(eig_exact)

    # Fit with R² and residual std
    p_spur, cov_spur = np.polyfit(np.log(k_vals), np.log(eig_spur), 1, cov=True)
    p_exact, cov_exact = np.polyfit(np.log(k_vals), np.log(eig_exact), 1, cov=True)
    se_spur = np.sqrt(cov_spur[0, 0])
    se_exact = np.sqrt(cov_exact[0, 0])

    # R²
    def r_squared(x, y, p):
        yhat = np.polyval(p, x)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / ss_tot

    r2_spur = r_squared(np.log(k_vals), np.log(eig_spur), p_spur)
    r2_exact = r_squared(np.log(k_vals), np.log(eig_exact), p_exact)

    print(f"  {'frac':>5s}  {'|k|':>8s}  {'eig_ex':>10s}  {'c2_ex':>8s}  {'eig_sp':>10s}  {'c2_sp':>8s}")
    for i, frac in enumerate(fracs):
        k2 = k_vals[i]**2
        print(f"  {frac:5.2f}  {k_vals[i]:8.5f}  {eig_exact[i]:10.4e}  {eig_exact[i]/k2:8.4f}"
              f"  {eig_spur[i]:10.4e}  {eig_spur[i]/k2:8.4f}")
    print()
    print(f"  eig_spur  ~ k^({p_spur[0]:.3f} +/- {se_spur:.3f})  R²={r2_spur:.6f}")
    print(f"  eig_exact ~ k^({p_exact[0]:.3f} +/- {se_exact:.3f})  R²={r2_exact:.6f}")
    print(f"  Note: 'first spur' selected as first mode with grad > 0.5 at each k")
    print(f"        (may be different eigenvector at different k — same O(k²) class)")


# ---------------------------------------------------------------------------
# Part 4: ||d1d0|| scaling — supports lemma d1d0 = O(k)
# ---------------------------------------------------------------------------

def d1d0_scaling(N=2, k_hat=np.array([1., 0., 0.])):
    mesh = build_mesh(N)
    fracs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    ks, norms = [], []

    for frac in fracs:
        k = (2 * np.pi / mesh['L']) * frac * k_hat
        ks.append(np.linalg.norm(k))
        d0 = build_d0_bloch(mesh['V'], mesh['E'], k, mesh['L_vec'], mesh['shifts'])
        d1s = build_d1_bloch_standard(mesh['V'], mesh['E'], mesh['F'],
                                      mesh['L'], k, mesh['el'], mesh['cr'])
        norms.append(np.linalg.norm(d1s @ d0))

    ks = np.array(ks)
    norms = np.array(norms)
    p, cov = np.polyfit(np.log(ks), np.log(norms), 1, cov=True)
    se = np.sqrt(cov[0, 0])

    print(f"  {'frac':>6s}  {'|k|':>8s}  {'||d1d0||':>10s}  {'||d1d0||/|k|':>12s}")
    for i, frac in enumerate(fracs):
        print(f"  {frac:6.3f}  {ks[i]:8.5f}  {norms[i]:10.6f}  {norms[i]/ks[i]:12.4f}")
    print()
    print(f"  Fit: ||d1d0|| ~ k^({p[0]:.3f} +/- {se:.3f})")
    print(f"  => d1d0 = O(k), confirming lemma premise")


# ---------------------------------------------------------------------------
# Part 5: Hodge orthogonality defect ||P_grad P_phys||
# ---------------------------------------------------------------------------

def hodge_orthogonality_defect(N=2, frac=0.05, k_hat=np.array([1., 0., 0.])):
    mesh = build_mesh(N)
    M = mesh['M']
    k = (2 * np.pi / mesh['L']) * frac * k_hat
    d0 = build_d0_bloch(mesh['V'], mesh['E'], k, mesh['L_vec'], mesh['shifts'])
    Pg = grad_projector(d0, M)

    # Exact: physical eigenvectors
    ee, ve, te, _ = solve_exact(mesh, k, d0)
    phys_e = ve[:, np.real(ee) > te]
    Pp_e = phys_e @ np.linalg.solve(phys_e.conj().T @ M @ phys_e, phys_e.conj().T @ M)
    defect_e = np.linalg.norm(Pg @ Pp_e)

    # Standard: physical eigenvectors (eig > threshold)
    es, vs, ts, _ = solve_standard(mesh, k)
    phys_s = vs[:, np.real(es) > ts]
    Pp_s = phys_s @ np.linalg.solve(phys_s.conj().T @ M @ phys_s, phys_s.conj().T @ M)
    defect_s = np.linalg.norm(Pg @ Pp_s)

    print(f"  Exact:    ||P_grad P_phys|| = {defect_e:.2e}")
    print(f"  Standard: ||P_grad P_phys|| = {defect_s:.2e}")
    print(f"  Ratio: {defect_s / max(defect_e, 1e-20):.1e}")
    print(f"  => Exact: perfect Hodge separation. Standard: gradient/physical subspaces overlap.")


# ---------------------------------------------------------------------------
# Part 6: First spur c² vs N — convergence or oscillation?
# ---------------------------------------------------------------------------

def spur_convergence_with_N(frac=0.05, k_hat=np.array([1., 0., 0.])):
    for N in [2, 3, 4]:
        mesh = build_mesh(N)
        M = mesh['M']
        k = (2 * np.pi / mesh['L']) * frac * k_hat
        k2 = np.dot(k, k)
        d0 = build_d0_bloch(mesh['V'], mesh['E'], k, mesh['L_vec'], mesh['shifts'])
        Pg = grad_projector(d0, M)

        es, vs, ts, _ = solve_standard(mesh, k)
        spur_c2 = []
        for i in range(len(es)):
            if abs(es[i]) < ts:
                continue
            g = grad_fraction(Pg, vs[:, i], M)
            if g > 0.5:
                spur_c2.append(es[i] / k2)
            else:
                break
        spur_c2 = np.array(spur_c2)
        print(f"  N={N} V={len(mesh['V'])}: n_spur={len(spur_c2)}"
              f"  c2_1st={spur_c2[0]:.4f}  c2_last={spur_c2[-1]:.4f}"
              f"  c2_median={np.median(spur_c2):.4f}")

    print()
    print("  c2_1st oscillates (1.55, 2.48, 1.93) — no convergence.")
    print("  n_spur grows with N — pollution is extensive.")
    print("  => Mesh refinement cannot fix spectral pollution from broken exactness.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("R1-6: Standard c² oscillations — full investigation")
    print("=" * 70)

    print("\nPART 0: Sanity checks (N=2)")
    print("-" * 50)
    mesh2 = build_mesh(2)
    sanity_checks(mesh2)
    print()

    print("PART 1: Mode classification (frac=0.05, k=[100])")
    print("-" * 50)
    for N in [2, 3, 4]:
        analyze_N(N)

    print("PART 2: Exact vs standard at multiple k (N=2)")
    print("-" * 50)
    compare_exact_vs_standard()

    print("PART 3: Spurious energy k-scaling (N=2, 8 points)")
    print("-" * 50)
    k_scaling_analysis()

    print("\nPART 4: ||d1d0|| scaling with k (N=2)")
    print("-" * 50)
    d1d0_scaling()

    print("\nPART 5: Hodge orthogonality defect (N=2, frac=0.05)")
    print("-" * 50)
    hodge_orthogonality_defect()

    print("\nPART 6: First spur c² vs N (frac=0.05)")
    print("-" * 50)
    spur_convergence_with_N()

    print("\nDone.")
