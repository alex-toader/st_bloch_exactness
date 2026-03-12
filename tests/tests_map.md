# Tests Map

Complete inventory of tests → paper claims.

---

## 1_test_core.py (3 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `analyze_structure(name, data)` | For each structure: ||d₁d₀|| exact vs std, n_zero, n_spur, c², gradient leakage, acoustic split. 4 k-fractions × 2 directions per structure. Runs on Kelvin, C15, WP. | Tables 2, 5; §5.1 |
| `test_sc_cubic()` | SC cubic N=3: same analysis. c²≈1.0. n_spur=5 [100], 13 [111]. | Tables 1, 5; §5.5 |
| `test_hodge_splitting()` | Full Hodge Laplacian Δ₁=K_grad+K_curl on Kelvin N=2. Counts mixed modes (0.01 < grad% < 0.99). Exact: 0 mixed. Std: 169–187 out of 192. | Table 4; §5.3 |

**Structures tested:** Kelvin N=2 (V=96), C15 N=1 (V=136), WP N=1 (V=46), SC N=3 (V=27).

**Quantitative results verified in header:**
- Exact: ||d₁d₀|| ~ 10⁻¹⁶, n_zero = V, c² → 1.0 on all structures
- Standard: ||d₁d₀|| ~ O(1), n_spur = 3–16 depending on structure/direction
- c²_std wrong on ALL: 1.57 (Kelvin), 0.52 (C15), 1.13 (WP)
- Gradient leakage: spurious modes ~87–98% gradient contamination

---

## 2_test_convergence.py (3 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `analyze_N(N, ...)` | Kelvin N=2..6 at k=5% BZ [100]. Physical eigenvalues exact vs std. | Table 3; Fig 1; §5.2 |
| `analyze_sc(N, ...)` | SC cubic N=3..7 at k=5% BZ [100] and [111]. Same analysis. | Table 3; R1-3/R2-2 |
| `convergence_fit(...)` | Log-log fit |error| ~ C/N^p → (p, R²). | §5.2 |

**Two mesh families:** Kelvin (BCC, Im3̄m) N=2..6 and SC cubic (Pm3̄m) N=3..7.

**Key finding:** Cell size h is constant (h=√2 Kelvin, h=2 SC). What varies with N is
k=2π·frac/(aN)∝1/N. Error = O((ka)²) = O(1/N²): dispersion convergence, not h-refinement.
Supercell eigenvalue = infinite lattice eigenvalue (verified on SC to machine precision).

**Quantitative results verified in header:**
- Kelvin: c²→1.0 (numerical), p=2.00, R²=1.0000.
- SC [100]: c²→1.0 = (⋆₂/⋆₁)·a² (analytical, ⋆₁=a, ⋆₂=1/a), p=2.00, R²=1.0000.
- SC [111]: c²→1.0, p=2.00, R²=1.0000. Same rate, both directions.
- Standard (Kelvin): 1.55→2.48→1.93→1.03→3.34 (SPURIOUS, not acoustic).
  n_spur=6→17→29→47→62.
- Standard (SC): 5.99→20.30→3.93→6.11→7.03 (SPURIOUS). n_spur=5→9→19→29→41.
- Two families, different symmetry groups: O(1/N²) is structural, not symmetry-dependent.
- R²=1.0000 genuine: both lattices perfectly regular, sub-leading correction <0.04%.

---

## 3_test_robustness.py (7 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_asymptotics()` | c² from k/BZ=10⁻⁴ to 0.20. Threshold sensitivity (4 relative thresholds). | Table 6 |
| `test_random_directions()` | 3 axis + 20 random k-directions at 10% BZ. n_spur and ||d₁d₀|| for each. | Table 6; §6 |
| `test_mesh_distortion()` | Kelvin with geometric noise ε=0..0.10, 3 seeds each (12 configs). | Table 6; §6 |
| `test_bz_boundary()` | k/BZ = 0.1, 0.5, 0.9, 0.999, 1.0. rank(d₀), ||d₁d₀||, n_zero. | Table 6 |
| `test_face_ordering()` | K under cycle, reverse, random vertex permutations per face. | Prop 2 |
| `test_gauge_transform()` | Eigenvalues under random vertex gauge transform e^{iθ(v)}, 5 seeds. | Table 6 |
| `test_operator_norm()` | ||K_std − K_exact||_F at k/BZ = 1%..30%. | §6; Reproducibility |

**Quantitative results verified in header:**
- Asymptotics: exact c² = 1.00000 stable; std c² = 1.57 (wrong, stable at wrong value)
- Random dirs: exact n_zero = 96 = V for ALL 23 directions. Std: 6 on axes, 14 on generic.
- Distortion: all 12 configs PASS (||d₁d₀|| < 10⁻¹²)
- BZ boundary: PASS up to 99.9%. DEGRADED at exact boundary (rank drop 96→95, n_zero=98)
- Face ordering: ||K−K_ref|| < 10⁻¹⁴ for all 6 permutations. K canonical.
- Gauge: max|Δeig| < 10⁻¹⁴ for all 5 seeds.
- Operator norm: ||ΔK||/||K|| grows from 3% (1% BZ) to 46% (30% BZ).

---

## 4_test_structure.py (4 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_random_voronoi()` | 10 random Voronoi seeds (n=50 cells). Exactness + pollution on each. | Table 5; §5.5 |
| `test_voronoi_scaling()` | n_cells=50,100,200,400 (3 seeds, 1 for 400). c² vs size. | R1-7 |
| `test_n_scaling()` | Kelvin N=2,3,4. n_spur at [100] and [111]. | §5.5; §6 |
| `test_scalar_laplacian()` | K₀=d₀†⋆₁d₀ at 4 k-values. No zero modes → no pollution at level 0. | §5.5 |
| `test_minimal_counterexample()` | SC 1×1×1: degenerate self-loop topology. Uses per-half-edge formula (not library standard). NEEDS REVISION — claim "standard FAILS" is based on formula differing from library. | §5.5 |

**Quantitative results verified in header:**
- Random Voronoi: 10/10 valid, 10/10 PASS. ||d₁d₀|| < 10⁻¹⁵ on all.
- N-scaling: n_spur grows (6→17→29 on [100]; 14→38→72 on [111]). Extensive.
- Scalar Laplacian: n_zero_K₀ = 0 for all k≠0. Problem specific to level 1→2.
- Minimal: SC 1×1×1 DEGENERATE. Test uses per-half-edge formula, not library standard. Topological d1=0 on self-loops → standard d1d0=0 trivially. Claim needs revision.

---

## 5_test_r16_oscillations.py (7 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `sanity_checks(mesh)` | Part 0: star1 uniformity, d1_exact==d1_std at k=0. | Sanity |
| `analyze_N(N)` | Part 1: Mode classification at frac=0.05 for N=2,3,4. M-weighted gradient projection. Spur count from scan. | R1-6; Table 3 |
| `compare_exact_vs_standard()` | Part 2: Exact (c²≈1, 0% grad) vs standard (1st spur c²~1.5, 94% grad; 1st phys c²~300) at frac=0.02..0.15. | R1-6 |
| `k_scaling_analysis()` | Part 3: Log-log fit eig vs k (8 points). Spur ~ k^1.95, exact ~ k^2.00. Error bars + R². | R1-6 |
| `d1d0_scaling()` | Part 4: ||d₁d₀|| ~ k^0.95, C≈102. Confirms d₁d₀ = O(k) (lemma premise). | R1-6; potential lemma |
| `hodge_orthogonality_defect()` | Part 5: ||P_grad P_phys|| = 4e-13 (exact) vs 2.87 (standard). | R1-6; Table 4 |
| `spur_convergence_with_N()` | Part 6: c²_1st oscillates (1.55, 2.48, 1.93) with N. No convergence. | R1-6; Table 3 |

**Quantitative results verified in header:**
- Part 0: star1 UNIFORM (all=4.0), d1 match at k=0 (diff=0), K asymmetry ~10⁻¹⁷
- Part 1: n_spur_scan=8 (N=2), 19 (N=3), 31 (N=4). Grad 93-97%.
- Part 2: Exact c² = 1.0000 at all k. Standard 1st phys = optic (c² >> 1).
- Part 3: eig_spur ~ k^(1.952±0.014) R²=0.9997. Both O(k²).
- Part 4: ||d₁d₀|| ~ k^(0.946±0.017). C≈102 stable at small k.
- Part 5: Hodge defect ratio 7.4×10¹².
- Part 6: c²_1st oscillates, n_spur grows with N.

**Note:** Gradient projector is M-weighted: P = d₀(d₀†Md₀)⁻¹d₀†M.
Correct on all meshes (not just uniform ⋆₁).

---

## 7_test_ker_theorem.py (2 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `verify_exact_chain(...)` | rank(d0)=ker(d1)=ker(K)=V + max grad% of physical < 10⁻⁸. Tested on 5 structures × 4 k-fractions. | R2-3/R2-6; potential Proposition |
| `verify_standard_failure()` | Standard: ||d1d0||=3.96, ker deficit=6, 8 modes with grad>0.5. Every chain link broken. | R2-3/R2-6 |

**Quantitative results verified in header:**
- Exact: rank(d0)=ker(d1)=ker(K)=V on all 5 structures, all k. max grad% < 3×10⁻¹².
- Standard: im(d0) ⊄ ker(d1), ker(K)=90≠96, gradient leakage up to 97%.

---

## 8_test_cor1_proof.py (4 parts)

| Part | What it computes | Paper ref |
|------|-----------------|-----------|
| Part 0: Uniqueness | Per-face constraint matrix from d₁d₀=0 has 1D null space. Kelvin: 112/112, SC: 81/81. | Cor 1 premise |
| Part 1: Intra-face contradictions | Count faces with same-shift edges having different recurrence phases. Kelvin: 37/112, SC: 45/81. | Cor 1; R1-1 |
| Part 2: Explicit example | Auto-found face: 3 shift-0 edges at positions 0,2,4. Net shift S=(-1,0,0) computed from data. ψ₂/ψ₀ = exp(ik·S·L) ≠ 1. | Cor 1 proof |
| Part 3: Standard verification | ||d₁ˢᵗᵈd₀|| = 8.37 (Kelvin), 7.74 (SC). Per-edge phases fail. | Cor 1 |

**Structures tested:** Kelvin N=2, SC cubic N=3.

**Logic chain:** uniqueness (Part 0) + intra-face inconsistency (Parts 1-2) → Cor 1.

**Quantitative results verified in header:**
- Uniqueness: all faces on both structures have exactly 1D null space
- 37/112 Kelvin faces, 45/81 SC faces have intra-face contradictions
- Explicit: |ψ₂/ψ₀ - 1| = 0.618, S=(-1,0,0) computed from data, match: 0
- Standard ||d₁d₀|| ~ O(1) on both structures, exact ~ 10⁻¹⁶

---

## 6_make_figures.py (2 functions)

| Function | What it generates | Paper ref |
|----------|------------------|-----------|
| `make_fig_bandstructure()` | Fig 2: Γ-X-M-R-Γ band structure, 121 k-points, 8 bands. 3 panels: exact bands, std bands, ||d₁d₀|| along path. | Fig 2; §5.5 |
| `make_fig_meshes()` | Fig 3: 3D wireframes of Kelvin foam and Random Voronoi. | Fig 3; §2.4 |

---

## 9_test_d2_exactness.py (3 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_sc()` | SC N=3: d₂_top·d₁_ex ~ O(k), d₂_exact·d₁_ex ~ 10⁻¹⁶, n_incon=9/81 [100], 27/81 [111]. | §7 Discussion |
| `test_kelvin()` | Kelvin N=2: d₂_top·d₁_ex ~ O(k), d₂_exact·d₁_ex ~ 10⁻¹⁵, n_incon=28/112 [100], 66/112 [111]. | §7 Discussion |
| `test_universality()` | All 4 structures: same recurrence pattern at level 2→3 as level 1→2. | §7 Discussion |

**Structures tested:** SC N=3, Kelvin N=2, C15 N=1, WP N=1.

**Key finding:** The exactness failure generalizes to d₂d₁. Same recurrence fixes it.
Per-face standard phase insufficient. Universal pattern:
d_{p+1}[σ,τ'] = -d_{p+1}[σ,τ]·d_p[τ,ρ]/d_p[τ',ρ].

---

## 10_test_dielectric.py (4 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------
| `test_uniform_eps()` | SC N=5: c²=1/ε for ε=1..12. Verifies H-field formulation K=d₁†(⋆₂/ε)d₁. | DIR-1; §7 |
| `test_dielectric_contrast()` | Kelvin N=2 with BCC sublattice contrast ε_A=1, ε_B=2..9. Exact vs standard at [100] and [111]. | DIR-1; §7 |
| `test_exactness_eps_independent()` | ||d₁d₀|| identical at all ε on all structures. n_zero=V for exact at all ε. | DIR-1; §7 |
| `test_tensor_eps_and_mu()` | Diagonal tensor ε and μ≠1. Per-face ε⁻¹_eff = n̂ᵀ·ε⁻¹·n̂, per-edge μ_eff = êᵀ·μ·ê. 3 configs × 3 structures. | DIR-4; §7 |

**Structures tested:** SC N=5 (uniform ε), Kelvin N=2 (sublattice contrast + tensor), C15 N=1, WP N=1.

**Key finding:** Exactness is independent of ALL material parameters — scalar ε, tensor ε, and μ≠1.
d₁(k) depends on neither ⋆₂ nor ⋆₁. Exactness is purely topological.
Standard pollution persists at ALL material configs.

**MPB reference:** BCC Voronoi, resolution=32:
eps_B=2: c²=0.691, eps_B=4: c²=0.453, eps_B=9: c²=0.255.
DEC ~20% higher (staircase error, lowest-order discretization).

**Quantitative results verified in header:**
- Uniform: c²=1/ε with 3.29e-04 relative error (O((ka)²) dispersion)
- Contrast exact: n_zero=96 at ALL ε, c² degenerate
- Contrast std: n_zero=90 ([100]) or 82 ([111]), c² split
- d₁d₀: 10⁻¹⁶ exact, ~4-8 standard, unchanged by ε
- C15 N=1: n_zero_ex=136=V at all ε. Standard: 127 (9 spurious).
- WP N=1: n_zero_ex=46=V at all ε. Standard: 43 (3 spurious).

---

## 11_test_h_refinement.py (3 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_kelvin_h_refinement()` | Kelvin N=2: vary L_cell=8→1 at fixed k=[0.05,0,0]. Error vs h, exact vs standard. | §5.2; DIR-S1 |
| `test_voronoi_h_refinement()` | Random Voronoi n=50..350 at fixed L=10, k=[0.05,0,0]. 5 seeds each. | §5.2; DIR-S1 |
| `test_standard_no_convergence()` | Standard c²≈1.57 at all h (Kelvin). No convergence. | §5.2; DIR-S1 |

**Structures tested:** Kelvin N=2 (6 L_cell values), Random Voronoi (5 sizes × 5 seeds).

**Key findings:**
- Kelvin: err ~ h^2.00, R² = 1.000000. On regular lattice, equivalent to k-convergence.
- Voronoi: err ~ h^2.05, R² = 0.981. Genuine h-refinement with topology changes.
- Standard: c²_std ≈ 1.57 constant (Kelvin), chaotic (Voronoi). No convergence.
  Spurious eigenvalues scale as k² (same as physical) → O(1) signal-to-pollution ratio.

---

## 12_test_mpb_comparison.py (5 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_vacuum_sanity()` | Both MPB and DEC give c²=1.0 at ε=1. DEC n_zero=V=96. | DIR-3; §7 |
| `test_dielectric_comparison()` | DEC vs MPB at eps_B=2,4,9. Ratio > 1, monotonic in eps_B. Exactness at all contrasts. | DIR-3; §7 |
| `test_k_scaling()` | Vacuum: err ratio 4.00 (O(k²) dispersion). eps_B=4: err ratio 1.00 (k-independent interface). | DIR-3; §7 |
| `test_hodge_star_sensitivity()` | Face-averaging sensitivity at eps_B=4, [100] and [111]. Harmonic: +21%, geometric: +4%, arithmetic: -12% vs MPB. Direction-independent. All preserve exactness. 57% interface. | DIR-3; §7 |
| `test_mpb_resolution_convergence()` | MPB c² converges res=16→32→64 at eps_B=4. res=32 vs 64: 0.5%. | DIR-3 |

**Structures tested:** BCC Voronoi (Kelvin N=2, L_cell=4.0) vs MPB (resolution=32).

**REQUIRES:** meep/mpb (conda-forge), h5py. Run with mpb_env:
  `/Users/alextoader/miniconda3/envs/mpb_env/bin/python tests/12_test_mpb_comparison.py`

**Key findings:**
- DEC overestimates c²: 5% (eps_B=2), 21% (eps_B=4), 50% (eps_B=9).
  Ratio monotonically increasing with contrast.
- Error decomposition via k-scaling:
  - Vacuum: pure O(k²) dispersion (ratio 4.00 at k doubling)
  - Dielectric: k-independent interface error (~20% at eps_B=4), from
    piecewise-constant ε in the Hodge star
- Both agree at vacuum (eps_B=1).
- MPB BCC Voronoi: eps defined on voxel grid via nearest-neighbor rule.
- Exactness preserved at all contrasts (n_zero = V = 96).
- MPB res=32 vs 64: 0.5% → MPB error negligible vs DEC error.

**Quantitative results verified in header:**
- Vacuum: MPB c²=1.0000, DEC c²=0.9987
- eps_B=2: MPB c²=0.6907, DEC c²=0.7263, ratio=1.0515
- eps_B=4: MPB c²=0.4528, DEC c²=0.5469, ratio=1.2078
- eps_B=9: MPB c²=0.2553, DEC c²=0.3832, ratio=1.5006
- K-scaling vacuum: err(0.05)/err(0.025)=4.00 (O(k²))
- K-scaling eps_B=4: err(0.05)/err(0.025)=1.00 (k-independent)
- MPB resolution: c²=0.449 (16), 0.453 (32), 0.455 (64)

---

## 13_test_bloch_cohomology.py (4 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_betti_all_structures()` | β₀,β₁,β₂,β₃ at Γ and 4 BZ points on all 4 structures. β(Γ)=(1,3,3,1), β(k≠0)=(0,0,0,0). | §5.X (new section); DIR-M2 |
| `test_bz_scan()` | Fine BZ scan Γ-X-M-R-Γ (36 k-points). β=(0,0,0,0) everywhere except Γ. | §5.X |
| `test_standard_no_cohomology()` | Standard complex: d₁d₀≠0, rank unstable (94-102), cohomology undefined. | §5.X |
| `test_rank_structure()` | Universal rank identities: Γ: (V-1, E-V-2, C-1), k≠0: (V, E-V, C). All structures. | §5.X |

**Structures tested:** SC N=3, Kelvin N=2, C15 N=1, WP N=1.

**Key findings:**
- Exact complex reproduces H*(T³) = (1,3,3,1) at Γ and trivial twisted cohomology at all k≠0.
- Standard complex is NOT a cochain complex at k≠0 → cohomology undefined.
- Rank of d₁_std jumps erratically (94-102); d₁_exact is stable (96 = V on Kelvin).
- Full BZ scan: no surprises at high-symmetry points (X, M, R). Sharp transition at Γ only.

---

## 14_test_acoustic_deformation.py (4 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_harmonic_extraction()` | 3 M₁-orthonormal harmonic 1-forms at Γ via Hodge decomposition within ker(d₁). Gap 10¹³. | §7 Discussion; DIR-S5 |
| `test_helmholtz_splitting()` | Subspace traces: Tr(H,grad)=1, Tr(H,acou)=2, Tr(H,opt)=0 at k→0⁺. 3 structures × 4 directions. | §7 Discussion; DIR-S5 |
| `test_acoustic_speed()` | ω²/k² = 1 (natural units) on all 3 foam structures. Lattice dispersion < 1% at small k. | §7 Discussion |
| `test_dispersion_curve()` | Full Γ→X dispersion on Kelvin N=2. v_s monotone 1.000→0.984. Tr(H,acou) tracks deformation. | §7 Discussion |

**Structures tested:** Kelvin N=2, C15 N=1, WP N=1.

**Key findings:**
- Exact complex implements discrete Helmholtz decomposition: at k≠0, H¹(T³)=ℝ³ splits into
  1 longitudinal (→gradient) + 2 transverse (→acoustic ω=|k|).
- Subspace traces 1+2+0=3 (Parseval) universal across structures and k-directions.
- Speed of sound v_s=1 in natural units, 1.6% lattice dispersion at zone boundary.
- Standard complex cannot do this: no cohomology at k≠0.

---

## 15_test_hodge_star_interface.py (8 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| test_formula_comparison | c² error vs MPB for 4 formulas × 7 contrasts | DIR-M7 |
| test_exactness_preserved | n_zero = V for all 4 formulas | DIR-M7 |
| test_k_independence | c² vs k_frac for logarithmic mean | DIR-M7 |
| test_direction_independence | c² along [100], [110], [111] | DIR-M7 |
| test_mesh_refinement | c² at N=2,3,4 fixed k_abs | DIR-M7 |
| test_high_contrast_stability | ε_B=16 error vs k_frac (cancellation check) | DIR-M7 |
| test_mpb_convergence | MPB res=32 vs 64 validation | DIR-M7 |
| test_scale_invariance | c²(α·ε_A, α·ε_B) = c²(ε_A, ε_B)/α for α=2,3 | DIR-M7 |

**Structures tested:** Kelvin N=2 (primary), N=3,4 (refinement), MPB reference.

**Key findings:**
- Logarithmic mean ε_L = (ε_B−ε_A)/ln(ε_B/ε_A) gives ≤2.8% error across ε_B=1.5–16×.
  26× better than harmonic (current), 7× better than geometric.
- Error is O(1) in BCC mesh size, k-independent, direction-independent.
- All formulas preserve exactness (d₁d₀=0 independent of Hodge star).
- Scale-invariant: only ratio ε_B/ε_A matters, not absolute values.
- Physical justification empirical (Whitney predicts harmonic; harmonic fails). DIR-S10 open.
- BCC has 57% interface faces (hexagonal) + 43% bulk (square). Formula affects 57% only.
- Optimal is power mean with p=0.14→0.31 (drifts with contrast). Log mean within 4%.

---

## 16_test_voronoi_dielectric.py (7 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| test_1_exactness | n_zero = V at n=50,120,200 × 3 seeds with eps_B=4 | DIR-S7 |
| test_2_stability_and_variance | c² stable across sizes, variance decreases | DIR-S7 |
| test_3_h_convergence | c² vs h, convergence fit | DIR-S7 |
| test_4_effective_medium | c² vs EM prediction at eps_B=2,9 | DIR-S7 |
| test_5_two_polarizations | Two lowest modes split (c²_1 < c²_2) | DIR-S7 |
| test_6_formula_comparison | log vs harmonic vs arithmetic on same meshes | DIR-S7 |
| test_7_high_contrast | Stable at eps_B=16 | DIR-S7 |

**Structures tested:** Random Voronoi (n=50..200, 3 seeds per size), z-plane dielectric split.

**Key findings:**
- Exactness preserved on ALL irregular meshes with dielectric contrast.
- c² already converged at n=50. Spread < 1% across sizes. Interface error O(1) per face.
- Effective medium: -0.3% (eps_B=2), +3.6% (eps_B=9). Sanity check, not precision ref.
- Formula comparison on Voronoi: arithmetic -1.5%, log +1.0%, harmonic +5.6% vs EM.
  Different from Kelvin where log wins. Log mean robust across both geometries.
- Two polarizations resolved. High contrast (eps_B=16) stable.
- Runtime: 117s with mesh cache (12 meshes built, 1 failed).

---

## 17_test_face_normal_statistics.py (5 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| test_1_kelvin_cos2_uniform | cos²(n̂,k̂) = 1/3 for all Kelvin interface faces | DIR-S7a |
| test_2_voronoi_cos2_distributed | cos²(n̂,ẑ) ≈ 0.52 distributed on Voronoi | DIR-S7a |
| test_3_kelvin_ow_vs_log | ow beats log at low ε_B, log wins at high ε_B | DIR-S7a |
| test_4_voronoi_ow_vs_log | ow ≈ log on Voronoi at eps_B=4 | DIR-S7a |
| test_5_effective_weight | t_log decreases with contrast (0.33 → 0.16) | DIR-S7a |

**Structures tested:** Kelvin N=2 (BCC), random Voronoi n=120 (3 seeds), z-plane dielectric split.

**Key findings:**
- Orientation-weighted formula NOT universal: works at low contrast, fails at high contrast.
- Log mean's effective interpolation weight ADAPTS to contrast (decreases from 0.33 to 0.16).
- Formula flip explained: Kelvin cos²=1/3, Voronoi cos²≈0.52 → different geometry, different ranking.
- Log mean is robust because contrast-dependence compensates for geometry-dependence.
- Runtime: 11s.

---

## 18_test_spectral_pairing.py (5 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_1_exact_pairing_kelvin()` | All 4 Hodge Laplacians Δ₀,Δ₁,Δ₂,Δ₃ at generic k on Kelvin. Nonzero eigenvalues paired to 2.6e-13. | §4.6 |
| `test_2_harmonics_at_gamma()` | Zero eigenvalue counts = (1,3,3,1) at Γ. Nonzero pairing still holds. | §4.6 |
| `test_3_standard_pairing_fails()` | Standard complex: max pairing residual 5.1e+01 (ratio 2×10¹⁴ vs exact). | §4.6 |
| `test_4_cross_talk()` | Normalized ||D1_lower·D1_upper||: exact 5.6e-18, standard 3.9e-02. | §4.6 |
| `test_5_universality()` | WP N=1 at Γ, generic k, R. All < 4e-13. | §4.6 |

**Structures tested:** Kelvin N=2, WP N=1, C15 N=1.
- Adv: #95-99. DIR-S11.
- Runtime: ~20s.

---

## 19_test_hodge_decomposition.py (5 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_1_decomposition_generic_k()` | SVD-based Hodge decomposition at generic k on Kelvin. dim=(96,0,96), ||P_E·P_C||=1.16e-14. | §4.6 |
| `test_2_decomposition_gamma()` | Hodge decomposition at Γ: dim=(95,3,94). P_H matches ker(Δ₁) to 3.4e-15. | §4.6 |
| `test_3_standard_fails()` | Standard complex: ||P_E·P_C||=5.38, P_H min eig=-1.00. Not a decomposition. | §4.6 |
| `test_4_bz_path()` | Full BZ path Γ-X-M-R + generic. Clean decomposition at all k-points. | §4.6 |
| `test_5_universality()` | WP N=1 at Γ, generic k, R. Same clean decomposition. | §4.6 |

**Structures tested:** Kelvin N=2, WP N=1, C15 N=1.
- Adv: #100-104. DIR-M12.
- Runtime: ~25s.

---

## 20_test_perturbation_stability.py (5 functions)

| Function | What it computes | Paper ref |
|----------|-----------------|-----------|
| `test_1_exactness_stable()` | d₁d₀=0 and d₂d₁=0 at eps=0..20% of mean edge length. | §6; DIR-S17 |
| `test_2_betti_stable()` | β(k)=(0,0), β(Γ)=(1,3) invariant under perturbation. | §6; DIR-S17 |
| `test_3_hodge_stable()` | Hodge decomposition dim=(96,0,96) and ||P_E·P_C|| < 1.2e-14 at all eps. | §6; DIR-S17 |
| `test_4_pairing_stable()` | Spectral pairing residual < 3.3e-13 at all eps. | §6; DIR-S17 |
| `test_5_multiple_seeds()` | 5 random seeds at 10% perturbation, all pass. | §6; DIR-S17 |

**Structures tested:** Kelvin N=2.
- Adv: #105-109. DIR-S17.
- Runtime: ~120s (5 tests × 5 perturbation levels each, rebuilds complex each time).

---

## Summary counts

| File | Functions | Structures tested | k-points computed | Paper tables/figs |
|------|-----------|-------------------|-------------------|-------------------|
| 1_test_core.py | 3 | 4 (Kelvin, C15, WP, SC) | ~40 | T1, T2, T4, T5 |
| 2_test_convergence.py | 3 | 2 (Kelvin N=2..6, SC N=3..7) | 15 | T3, F1, R1-3/R2-2 |
| 3_test_robustness.py | 7 | 1 (Kelvin) + distorted | ~80 | T6 |
| 4_test_structure.py | 5 | 12+10 (Kelvin + 10 Voronoi + SC + scaling) | ~40 | T5, R1-7 |
| 5_test_r16_oscillations.py | 7 | 1 (Kelvin N=2,3,4) | ~30 | R1-6, potential lemma |
| 7_test_ker_theorem.py | 2 | 5 (all structures) | 20 | R2-3/R2-6 |
| 8_test_cor1_proof.py | 3 | 2 (Kelvin, SC) | 2 | Cor 1, R1-1 |
| 9_test_d2_exactness.py | 3 | 4 (SC N=3, Kelvin N=2, C15 N=1, WP N=1) | 12 | §7 Discussion |
| 10_test_dielectric.py | 4 | 4 (SC N=5, Kelvin N=2, C15 N=1, WP N=1) | ~40 | DIR-1, DIR-4, §7 |
| 11_test_h_refinement.py | 3 | 1 (Kelvin, 6 L_cell) + Voronoi (5 sizes × 5 seeds) | ~130 | §5.2, DIR-S1 |
| 12_test_mpb_comparison.py | 5 | 1 (Kelvin N=2) vs MPB | ~10 | DIR-3, §7 |
| 13_test_bloch_cohomology.py | 4 | 4 (SC, Kelvin, C15, WP) | ~56 | §5.X, DIR-M2 |
| 14_test_acoustic_deformation.py | 4 | 3 (Kelvin, C15, WP) | ~100 | §7, DIR-S5 |
| 15_test_hodge_star_interface.py | 8 | 1 (Kelvin N=2,3,4) vs MPB | ~60 | DIR-M7 |
| 16_test_voronoi_dielectric.py | 7 | Random Voronoi (n=50..200) | ~36 | DIR-S7 |
| 17_test_face_normal_statistics.py | 5 | Kelvin + Voronoi | ~20 | DIR-S7a |
| 18_test_spectral_pairing.py | 5 | 2 (Kelvin, WP) | ~8 | DIR-S11 |
| 19_test_hodge_decomposition.py | 5 | 2 (Kelvin, WP) | ~10 | DIR-M12 |
| 20_test_perturbation_stability.py | 5 | 1 (Kelvin) | ~50 | DIR-S17 |
| 6_make_figures.py | 2 | 1 (Kelvin) + 1 (Voronoi) | 121 | F2, F3 |
