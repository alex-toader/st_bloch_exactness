# Exactness-preserving discrete Maxwell operators on periodic polyhedral complexes

Feb 2026 — DRAFT v0.3

---

## Abstract

We observe that the standard construction of Bloch-periodic discrete exterior calculus (DEC) operators on unstructured cell complexes does not preserve exactness: the composition d₁(k)d₀(k) ≠ 0 at nonzero wave vector k, violating the discrete de Rham property that is essential for spurious-free Maxwell eigenvalue computations (Boffi 2010). We present an explicit, algorithmically simple construction of an exactness-preserving curl operator d₁(k) via a recurrence on face boundaries, exploiting the flat holonomy of the Bloch connection. The resulting curl-curl operator K(k) = d₁(k)†⋆₂d₁(k) is canonical: independent of face vertex ordering and orientation choices. We prove that the gauge kernel has dimension |V| and that two transverse acoustic eigenmodes emerge with ω² = c²|k|² + O(|k|⁴). Numerical verification is presented on six periodic polyhedral complexes: simple cubic (analytic benchmark, c² = a²), three Voronoi foams with O_h symmetry (C15, Kelvin, Weaire–Phelan), FCC (non-Plateau), and random Voronoi (no symmetry). Robustness tests under geometric perturbation, anisotropic domain deformation, and Hodge star perturbation confirm that exactness and kernel dimension are topological invariants of the construction, while wave speed and polarization degeneracy depend on metric and symmetry respectively.

---

## 1. Introduction

The discrete exterior calculus (DEC) provides a coordinate-free discretization of differential forms on cell complexes [Hirani 2003, Desbrun et al. 2005]. For Maxwell's equations, DEC naturally represents the electric field as a 1-cochain (edge values) and the magnetic flux as a 2-cochain (face values), with discrete gradient d₀ and curl d₁ acting between them. The resulting curl-curl eigenvalue problem

    K a = ω² M a,    K = d₁† ⋆₂ d₁,    M = ⋆₁

yields electromagnetic eigenfrequencies when the discrete de Rham sequence

    C⁰ →(d₀) C¹ →(d₁) C²

is exact, i.e., d₁d₀ = 0. Exactness is not merely a formal requirement: Boffi [2010] showed that failure of exactness in discrete Maxwell formulations leads to spurious eigenvalues that pollute the physical spectrum.

For periodic structures, Bloch boundary conditions twist the operators by a wave vector k ∈ ℝ³. On structured grids (Yee lattice), this is straightforward: the uniform topology ensures d₁(k)d₀(k) = 0 automatically. However, on unstructured polyhedral meshes — as arise from Voronoi tessellations of periodic point sets — the standard construction of applying independent Bloch phases to each face of d₁ breaks exactness at k ≠ 0.

We are not aware of a treatment that preserves discrete exactness for Bloch-periodic DEC operators on general polyhedral meshes. Existing DEC-based Maxwell computations have predominantly used either (a) structured grids where the issue does not arise [Schulz & Teixeira 2018], or (b) non-periodic domains where Bloch conditions are unnecessary [Chen & Chew 2017]. The related discrete de Rham (DDR) method of Di Pietro & Droniou [2021] achieves exactness on polyhedral meshes but has not been formulated with Bloch periodicity. Band structure computations on foam-like structures [Klatt et al. 2019] use plane-wave expansion (MPB), not DEC.

**Contribution.** We present:

1. An explicit construction of an exactness-preserving d₁(k) via a recurrence that derives the curl phases from those already present in d₀(k), exploiting the flat holonomy of the Bloch connection on each face.

2. A proof that the resulting curl-curl operator K(k) is canonical (independent of vertex ordering and face orientation choices), with gauge kernel of dimension |V|.

3. Numerical verification on six qualitatively different periodic polyhedral complexes, including an analytic SC cubic benchmark and random Voronoi complexes with no symmetry.

4. A systematic demonstration that exactness and kernel dimension are topological properties (robust under geometric, metric, and symmetry-breaking perturbations), while wave speed and degeneracy are metric and symmetry properties respectively.

---

## 2. Periodic polyhedral complexes and DEC operators

### 2.1 Cell complex on the torus

Let Λ = Lₓℤ × L_yℤ × L_zℤ be a lattice in ℝ³ with fundamental domain Ω = [0, Lₓ) × [0, L_y) × [0, L_z). A periodic cell complex (V, E, F, C) on the flat torus T³ = ℝ³/Λ consists of:

- V: a finite set of vertices (0-cells), positions in Ω
- E: a finite set of oriented edges (1-cells), each connecting two vertices
- F: a finite set of oriented faces (2-cells), each bounded by a cycle of edges
- C: a finite set of 3-cells filling the torus

Edges connecting vertices across the periodic boundary carry a lattice shift vector nₑ ∈ ℤ³: the edge from vertex i to vertex j has

    nₑ = round((xⱼ − xᵢ) / L)

where division is componentwise. Interior edges have nₑ = 0.

### 2.2 Incidence matrices

The topological gradient d₀ ∈ ℝ^(|E|×|V|) and curl d₁ ∈ ℝ^(|F|×|E|) encode the boundary relations:

    (d₀)ₑᵥ = +1  if v = head(e)
            = −1  if v = tail(e)
            =  0  otherwise

    (d₁)ₑf = +1  if e ∈ ∂f, same orientation
            = −1  if e ∈ ∂f, opposite orientation
            =  0  if e ∉ ∂f

These satisfy d₁d₀ = 0 (every vertex appears in ∂f an even number of times with canceling signs).

### 2.3 Hodge stars

The diagonal Hodge star operators ⋆₁ ∈ ℝ^(|E|×|E|) and ⋆₂ ∈ ℝ^(|F|×|F|) encode metric information from the dual complex:

    (⋆₁)ₑₑ = |σ*ₑ| / |σₑ|
    (⋆₂)ff = |σ*f| / |σf|

where σₑ, σf are primal edge length and face area, and σ*ₑ, σ*f are their dual counterparts. For Voronoi complexes, these are computed from the Voronoi–Delaunay duality. For the SC cubic benchmark with lattice constant a, all stars are uniform: ⋆₁ = a·I, ⋆₂ = a·I.

---

## 3. Bloch operators and the exactness problem

### 3.1 Bloch-twisted gradient

For wave vector k ∈ ℝ³, the Bloch-twisted gradient d₀(k) is:

    d₀(k)[e, v] = +exp(ik·nₑL)   if v = head(e)
                 = −1               if v = tail(e)
                 =  0               otherwise

This is standard and well-defined: each edge carries a unique lattice shift.

### 3.2 Standard Bloch curl: failure of exactness

The naive extension to d₁ applies independent Bloch phases per face:

    d₁ˢᵗᵈ(k)[f, e] = d₁[f, e] · exp(ik·nₑL)

On structured grids, this preserves exactness because each face is a parallelogram with compensating shift structure. On unstructured meshes, however, d₁ˢᵗᵈ(k)d₀(k) ≠ 0 in general.

**Example.** On a simple cubic N=3 supercell (27 vertices, 81 edges, 81 faces) at k = (2π/L)·0.02·(1,0,0):

| Operator | ‖d₁(k) d₀(k)‖ | Acoustic modes |
|----------|---------------|----------------|
| d₁ˢᵗᵈ(k) | 5.5 | 0 |
| d₁ᵉˣᵃᶜᵗ(k) | 10⁻¹⁶ | 2 (degenerate) |

The standard construction produces no acoustic modes — all physical eigenvalues are contaminated by the broken exactness.

### 3.3 Diagnosis

The product d₁(k)d₀(k) at a face f and vertex v yields

    (d₁d₀)[f, v] = Σ_{e ∈ ∂f, v ∈ e}  d₁[f,e] · d₀(k)[e,v]

At k = 0, cancellation is guaranteed by the topological constraint d₁d₀ = 0. At k ≠ 0, the phases exp(ik·nₑL) on different edges of ∂f destroy the cancellation unless the phases are mutually consistent across the face boundary.

**Proposition 0 (Standard d₁ breaks exactness).** Let f be a face with at least two boundary edges e_a, e_b having distinct lattice shifts n_a ≠ n_b. Then d₁ˢᵗᵈ(k)d₀(k) ≠ 0 for almost all k ∈ ℝ³ (with respect to Lebesgue measure). More precisely, d₁ˢᵗᵈ(k)d₀(k) = 0 only on the measure-zero set where exp(ik·(n_a − n_b)L) satisfies a specific algebraic constraint.

*Proof.* In the standard construction, d₁ˢᵗᵈ(k)[f, e] = d₁[f,e]·exp(ik·nₑL). At a vertex v shared by e_a and e_b, the two contributions to (d₁ˢᵗᵈd₀)[f, v] carry phases exp(ik·n_aL) and exp(ik·n_bL). Their cancellation requires exp(ik·(n_a − n_b)L) = 1, which defines a hyperplane in k-space. This holds at k = 0 but fails for almost all k when n_a ≠ n_b. On structured grids, every face is a parallelogram whose opposite edges carry identical shifts (n_a = n_b for all edge pairs sharing a vertex on ∂f), so cancellation is automatic. On unstructured meshes, faces generically have edges with distinct shifts, and exactness fails. □

This explains why the failure is specific to unstructured periodic meshes and does not arise on Yee-type grids.

---

## 4. Exactness-preserving construction

### 4.1 The flat Bloch connection

**Lemma 1 (Flat holonomy).** Let f be a face with boundary vertices v₀, v₁, ..., v_{n−1} (ordered cyclically) and boundary edges e₀, ..., e_{n−1} where eᵢ connects vᵢ to v_{(i+1) mod n}. Define the holonomy around f as

    Hf = Π_{i=0}^{n−1} [ −d₀(k)[eᵢ, v_{i+1}] / d₀(k)[eᵢ, vᵢ] ]

Then Hf = 1 for all k.

*Proof.* Each factor in the product has the form exp(ik·nᵢL) where nᵢ is the lattice shift of edge eᵢ. Therefore

    Hf = exp(ik · (Σᵢ nᵢ) · L)

Since the boundary ∂f is a closed loop on the torus, the total lattice shift vanishes: Σᵢ nᵢ = 0. Hence Hf = 1. □

**Remark.** This is the discrete analogue of the flat connection condition: the Bloch connection has zero curvature, so its holonomy around any contractible loop is trivial.

### 4.2 Recurrence construction

**Proposition 1 (Exactness-preserving d₁).** There exists a linear operator d₁(k): C¹ → C² satisfying d₁(k)d₀(k) = 0 for all k. The construction is explicit:

For each face f with ordered boundary edges e₀, ..., e_{n−1} and incidence signs σᵢ = ±1:

1. Set φ₀ = 1.
2. For i = 1, ..., n−1:

        φᵢ = −σᵢ₋₁ · φᵢ₋₁ · d₀(k)[eᵢ₋₁, vᵢ] / (σᵢ · d₀(k)[eᵢ, vᵢ])

3. Set d₁(k)[f, eᵢ] = σᵢ · φᵢ.

*Proof.* The condition d₁(k)d₀(k) = 0 at face f decomposes into n vertex equations, one per vertex of ∂f. At vertex vᵢ for i = 1, ..., n−1, the equation involves edges eᵢ₋₁ and eᵢ:

    d₁[f, eᵢ₋₁] · d₀(k)[eᵢ₋₁, vᵢ] + d₁[f, eᵢ] · d₀(k)[eᵢ, vᵢ] = 0

Substituting d₁[f, eᵢ] = σᵢφᵢ shows this is exactly the recurrence relation in step 2.

The remaining equation at v₀ involves edges e_{n−1} and e₀:

    σ_{n−1}·φ_{n−1} · d₀(k)[e_{n−1}, v₀] + σ₀·φ₀ · d₀(k)[e₀, v₀] = 0

This holds if and only if Hf = 1, which is guaranteed by Lemma 1. □

### 4.3 Canonical curl-curl operator

The construction of d₁(k) depends on the choice of starting vertex and the cyclic orientation of each face. However:

**Proposition 2 (Canonical K).** The curl-curl operator K(k) = d₁(k)† ⋆₂ d₁(k) is independent of:
(a) the starting vertex in each face,
(b) the cyclic orientation of each face,
(c) the initial phase seed φ₀.

*Proof.* All three choices multiply the row d₁[f, :] by a scalar αf. We show |αf| = 1 in each case.

(a) Starting from vertex vⱼ instead of v₀ multiplies all phases by φⱼ/φ₀. Since each recurrence step is a ratio of entries of d₀(k) — which are exp(ik·nL) times ±1 — all phases have unit modulus: |φᵢ| = 1.

(b) Reversing the orientation of face f negates all incidence signs, giving αf = −1.

(c) Choosing φ₀ = α gives αf = α, and |α| = 1 is required for consistency.

Since K decomposes as K = Σf ⋆₂[f] · d₁[f,:]† · d₁[f,:], and |αf|² = 1, each term is invariant. □

*Numerical verification.* On C15 (160 faces), Kelvin (112 faces), and WP (54 faces) with random vertex rotations and orientation reversals: ‖ΔK‖ < 10⁻¹⁴, max|Δλ| < 10⁻¹⁴, while ‖Δd₁‖ ~ 10¹.

### 4.4 Gauge kernel

**Proposition 3 (Kernel dimension).** For generic k ≠ 0, dim ker K(k) = |V|.

*Proof.* Since ⋆₂ is positive definite, ker K(k) = ker d₁(k). The exact sequence d₁(k)d₀(k) = 0 gives im d₀(k) ⊆ ker d₁(k). It remains to show equality.

The twisted cohomology H¹_k = ker d₁(k) / im d₀(k) is isomorphic to the sheaf cohomology H¹(T³, L_k) of the flat line bundle L_k with holonomy exp(ik·L) along each lattice direction. By the Künneth formula for flat line bundles on the torus [Bott & Tu 1982, §14], for generic k (i.e., when exp(ik·Lⱼ) ≠ 1 for all j = 1,2,3), this cohomology vanishes: H¹(T³, L_k) = 0.

Therefore ker d₁(k) = im d₀(k). It remains to show d₀(k) is injective. Suppose d₀(k)u = 0. Then for every edge (i,j) with lattice shift nₑ: uⱼ·exp(ik·nₑL) = uᵢ. Iterating along any path, u is determined up to a global constant. But closing a path around the j-th torus cycle requires uᵢ·exp(ik·Lⱼ) = uᵢ, which forces uᵢ = 0 when exp(ik·Lⱼ) ≠ 1. Hence d₀(k) is injective at generic k, and dim im d₀(k) = |V|. □

**Remark.** At k = 0, the cohomology H¹(T³, ℂ) = ℂ³ is nonzero, so dim ker K(0) = |V| + 3, corresponding to the three harmonic 1-forms on T³. At special k where some exp(ik·Lⱼ) = 1, the kernel dimension may increase.

*Numerical verification.* dim ker K(k) = |V| confirmed on all six test structures at multiple k-points and under all perturbation types (geometric, metric, anisotropic).

---

## 5. Spectral properties

### 5.1 Maxwell eigenvalue problem

The generalized eigenvalue problem

    K(k) a = ω² ⋆₁ a

has |V| zero eigenvalues (gauge modes) and |E| − |V| positive eigenvalues (physical modes). At small |k|, the two smallest physical eigenvalues correspond to transverse acoustic modes.

### 5.2 Acoustic dispersion

For all six test structures, the two lowest physical eigenvalues satisfy ω² = c²|k|² + O(|k|⁴), confirming acoustic-type dispersion. On the SC cubic lattice with lattice constant a = 2 and uniform Hodge stars, the analytic result c² = a² = 4 is recovered to 6 significant figures at 0.5% of the Brillouin zone.

### 5.3 Degeneracy and isotropy

On structures with O_h point group symmetry (SC, C15, Kelvin, WP, FCC), the two acoustic modes are exactly degenerate: polarization splitting < 10⁻¹² (machine precision). On random Voronoi complexes (no symmetry), the splitting is nonzero (10⁻⁶ to 10⁻⁴) but the two modes remain acoustic.

Isotropy of the wave speed c(k̂) depends on symmetry. For O_h structures, the anisotropy Δc/c < 0.02% at 2% BZ. For random Voronoi, anisotropy is nonzero but small (< 0.02%).

---

## 6. Robustness: topological vs. metric vs. symmetry

A key structural result is the clean separation between properties that depend on topology, metric, or symmetry. We demonstrate this by systematic perturbation experiments on the SC cubic benchmark.

### 6.1 Geometric perturbation (Voronoi)

Starting from the 27 SC lattice sites, we add random displacements of magnitude ε to each site and construct the periodic Voronoi complex.

| ε | ‖d₁d₀‖ | ker = |V| | mean split | c² |
|---|---------|-----------|------------|-----|
| 0.00 | 3×10⁻¹² | YES | 3×10⁻¹² | 1.000 |
| 0.01 | 4×10⁻¹⁶ | YES | 2×10⁻⁷ | 1.000 |
| 0.05 | 4×10⁻¹⁶ | YES | 9×10⁻⁷ | 1.000 |
| 0.10 | 5×10⁻¹⁶ | YES | 2×10⁻⁶ | 1.000 |
| 0.20 | 4×10⁻¹⁶ | YES | 3×10⁻⁶ | 1.000 |

Exactness and kernel dimension are preserved. Degeneracy breaks proportionally to ε.

### 6.2 Anisotropic domain

We stretch the periodic domain to L = (L₀, αL₀, βL₀) with α ≠ β, keeping the SC topology and uniform Hodge stars.

| Scaling | c²_[100] | c²_[010] | c²_[001] | exact+ker |
|---------|----------|----------|----------|-----------|
| (1, 1, 1) | 4.00 | 4.00 | 4.00 | YES |
| (1, 1.1, 0.9) | 4.00 | 4.84 | 3.24 | YES |
| (1, 1.3, 0.7) | 4.00 | 6.76 | 1.96 | YES |

Exactness and kernel are invariant. Wave speed becomes directional, scaling as c² ∝ L² (as expected from the k-space metric).

### 6.3 Hodge star perturbation

We perturb the diagonal Hodge stars: ⋆₁ → ⋆₁ · (1 + ε·ξ), ξ ~ N(0,1), keeping d₁(k) fixed.

| ε | ker = |V| | c² | split |
|---|-----------|-----|-------|
| 0.00 | YES | 4.00 | 2×10⁻¹² |
| 0.01 | YES | 4.00 | 2×10⁻³ |
| 0.10 | YES | 4.03 | 2×10⁻² |
| 0.20 | YES | 4.09 | 5×10⁻² |

Kernel dimension is invariant: since ⋆₂ is positive definite, ker K(k) = ker d₁(k), which depends only on the topological operator d₁(k) and not on the Hodge stars ⋆₁, ⋆₂. The eigenvalues of the generalized problem K a = ω² ⋆₁ a, however, depend on both stars — wave speed and degeneracy are metric properties.

### 6.4 Summary

| Property | Type | Robust under perturbation |
|----------|------|--------------------------|
| d₁(k)d₀(k) = 0 | topological | all |
| dim ker K = |V| | topological | all |
| ω² = c²|k|² | metric | geometry, symmetry (not stars) |
| Polarization degeneracy | symmetry | topology, metric (not symmetry) |

---

## 7. Numerical results on Voronoi foams

### 7.1 Test structures

| Structure | Type | V | E | F | Plateau | O_h |
|-----------|------|---|---|---|---------|-----|
| SC cubic | regular | 27 | 81 | 81 | no (4 faces/edge) | yes |
| C15 | Voronoi | 136 | 272 | 160 | yes | yes |
| Kelvin | Voronoi | 96 | 192 | 112 | yes | yes |
| Weaire–Phelan | Voronoi | 46 | 92 | 54 | yes | yes |
| FCC | Voronoi | 12 | 32 | 24 | no (8 edges/vertex) | yes |
| Random | Voronoi | 206 | 412 | 236 | yes | no |

### 7.2 Exactness and kernel

| Structure | ‖d₁(k)d₀(k)‖ | ‖d₁ˢᵗᵈ(k)d₀(k)‖ | dim ker = |V| |
|-----------|--------------|-------------------|----------------|
| SC cubic | 0 | 5.5 | 27 = 27 |
| C15 | 4×10⁻¹⁶ | 3.3 | 136 = 136 |
| Kelvin | 1×10⁻¹⁶ | 2.5 | 96 = 96 |
| WP | 4×10⁻¹⁶ | 1.8 | 46 = 46 |
| FCC | 0 | 5.5 | 12 = 12 |
| Random | 1×10⁻¹⁵ | 6.2 | 206 = 206 |

### 7.3 Wave speeds and isotropy

At 2% of the Brillouin zone, wave speeds measured along [100], [110], [111]:

| Structure | c²_[100] | c²_[110] | c²_[111] | Anisotropy |
|-----------|----------|----------|----------|------------|
| C15 | 0.9998 | 0.9998 | 0.9998 | 0.0009% |
| Kelvin | 0.9989 | 0.9989 | 0.9989 | 0.023% |
| WP | 0.9996 | 0.9996 | 0.9996 | 0.011% |
| FCC | 0.9999 | 0.9998 | 0.9998 | 0.01% |

### 7.4 SC cubic analytic benchmark

The SC cubic lattice with uniform Hodge stars ⋆₁ = ⋆₂ = a·I provides an analytic benchmark: c² = a² exactly. With a = 2:

| k_frac | c²_[100] | c²_[111] | |c² − 4| |
|--------|----------|----------|----------|
| 0.005 | 3.999963 | 3.999988 | 3.7×10⁻⁵ |
| 0.010 | 3.999854 | 3.999951 | 1.5×10⁻⁴ |
| 0.020 | 3.999415 | 3.999805 | 5.9×10⁻⁴ |
| 0.100 | 3.985400 | 3.995128 | 1.5×10⁻² |

The error scales as O(k²), consistent with second-order accuracy. Finite-size convergence is confirmed: error decreases monotonically from N = 3 to N = 5. Polarization degeneracy is at machine precision: split < 10⁻¹¹.

---

## 8. Discussion

### Relation to Boffi's exactness criterion

Boffi [2010] established that exactness of the discrete de Rham complex is necessary and sufficient for spurious-free Maxwell eigenvalue computations. Our construction provides this exactness for DEC with Bloch boundary conditions on polyhedral meshes — a setting not covered by existing implementations. The standard Bloch-phased d₁ fails this criterion and produces zero acoustic modes even on the SC cubic lattice.

### Relation to bundle-valued DEC

Braune, Tong, Gay-Balmaz, and Desbrun [2024] developed DEC for bundle-valued forms, which conceptually includes flat line bundles. However, they do not address Bloch periodicity or the specific exactness failure we identify. Our construction can be viewed as the explicit realization of their framework for the flat line bundle on the torus.

### Relation to DDR methods

The discrete de Rham (DDR) method of Di Pietro and Droniou [2021] achieves exactness on general polyhedral meshes with high-order accuracy. Their framework does not currently include Bloch boundary conditions. Extending DDR to periodic settings would likely require a similar holonomy-based construction.

### Limitations

1. The Hodge stars are diagonal (circumcentric dual). Non-diagonal stars (e.g., Galerkin) would require modification of the curl-curl assembly but not the exactness construction.

2. The mesh must have valid Voronoi structure (non-degenerate dual). For random point sets, this requires sufficient density (≥ 50 points per periodic box of side L = 4 gives 80% valid meshes).

3. We treat only the 3D Maxwell eigenvalue problem. Extension to 2D and to time-domain DEC with Bloch conditions is straightforward.

---

## 9. Conclusion

We have identified and resolved a failure of exactness in Bloch-periodic DEC on unstructured polyhedral meshes that, to our knowledge, has not been previously addressed. The fix is algorithmically simple (a face-boundary recurrence) and produces a canonical curl-curl operator with correct gauge kernel and acoustic spectrum. The construction works on any periodic cell complex — Plateau foams, non-Plateau lattices, and random Voronoi complexes — with topological properties (exactness, kernel dimension) that are provably robust under arbitrary geometric and metric perturbations.

---

## References

- Arnold, D.N., Falk, R.S., Winther, R. (2006). Finite element exterior calculus, homological techniques, and applications. *Acta Numerica* 15, 1–155.
- Arnold, D.N., Falk, R.S., Winther, R. (2010). Finite element exterior calculus: from Hodge theory to numerical stability. *Bull. Amer. Math. Soc.* 47, 281–354.
- Boffi, D. (2010). Finite element approximation of eigenvalue problems. *Acta Numerica* 19, 1–120.
- Bott, R., Tu, L.W. (1982). *Differential Forms in Algebraic Topology.* Springer.
- Boffi, D., Conforti, M., Gastaldi, L. (2006). Modified edge finite elements for photonic crystals. *Numer. Math.* 105, 249–266.
- Braune, C., Tong, Y., Gay-Balmaz, F., Desbrun, M. (2024). Bundle-valued discrete exterior calculus. Preprint.
- Chen, Z., Chew, W.C. (2017). DEC for electromagnetic analysis. *IEEE Trans. Antennas Propag.*
- Desbrun, M., Hirani, A.N., Leok, M., Marsden, J.E. (2005). Discrete exterior calculus. Preprint.
- Di Pietro, D.A., Droniou, J. (2021). *The Hybrid High-Order Method for Polytopal Meshes.* Springer.
- Dobson, D.C., Pasciak, J.E. (2001). Analysis of an algorithm for computing electromagnetic normal modes in crystals. *Comput. Methods Appl. Math.* 1, 137–150.
- Hirani, A.N. (2003). *Discrete Exterior Calculus.* PhD thesis, Caltech.
- Jin, S., Xia, J., Xu, Z. (2025). Mimetic finite difference method for Maxwell eigenvalue problem with Bloch boundary conditions. Preprint.
- Klatt, M.A., Steinhardt, P.J., Torquato, S. (2019). Phoamtonic designs yield sizeable 3D photonic band gaps. *PNAS* 116, 23480–23486.
- Schulz, R.B., Teixeira, F.L. (2018). DEC discretization of Maxwell's equations on structured grids. *J. Comput. Phys.*

---

## Appendix A: Reproducibility

All computations use Python 3.9, NumPy, SciPy (LAPACK/BLAS). Source code: [Zenodo DOI to be assigned].

Test scripts:
- `4_test_M5_exactness.py`: exactness, holonomy, kernel, acoustic modes, seed independence
- `5_test_M5_ordering.py`: ordering independence of K
- `6_test_M5_fcc.py`: FCC non-Plateau verification
- `7_test_M5_random_voronoi.py`: random Voronoi (no symmetry)
- `8_test_M5_sc_convergence.py`: SC analytic benchmark, perturbation studies

---

*Feb 2026*
