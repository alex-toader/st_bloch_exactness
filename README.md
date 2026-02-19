# st_bloch_exactness

Exactness-preserving discrete Maxwell operators on periodic polyhedral complexes.

## Problem

Standard Bloch-periodic DEC operators on unstructured polyhedral meshes break the discrete de Rham exactness property (d₁d₀ ≠ 0 at k ≠ 0). This produces spectral pollution, broken Hodge splitting, and convergence failure in the curl-curl eigenvalue problem.

## Solution

A recurrence construction on face boundaries derives the curl operator d₁(k) from the gradient d₀(k), exploiting the flat holonomy of the Bloch connection. The resulting curl-curl operator K(k) = d₁(k)†⋆₂d₁(k) is canonical (independent of vertex ordering and face orientation) with gauge kernel dim = |V|.

## Results

Verified on five periodic polyhedral complexes:

| Structure | V | E | ‖d₁d₀‖ (exact) | ‖d₁d₀‖ (standard) | n_spur |
|-----------|---|---|-----------------|--------------------|----|
| SC cubic (N=3) | 27 | 81 | 0 | 6.2 | 5 |
| C15 (Laves) | 136 | 272 | 5×10⁻¹⁶ | 7.8 | 9-16 |
| Kelvin (BCC) | 96 | 192 | 6×10⁻¹⁶ | 7.5 | 6-14 |
| Weaire-Phelan | 46 | 92 | 5×10⁻¹⁶ | 5.1 | 3-7 |
| Random Voronoi (10 seeds) | 206-354 | 412-708 | 10⁻¹⁵ | 9-12 | 12-20 |

Mesh convergence: exact construction converges at O(h²) (second-order). Standard construction oscillates and does not converge.

## Requirements

- Python 3.9+
- NumPy, SciPy
- Matplotlib (for figure generation only)

## Running tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_core.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_convergence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_robustness.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_structure.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/make_figures.py
```

## Test suite → Paper claims

Each test file maps to specific claims in the paper. Individual tests within each file are listed below.

| Test file | Paper section | What it verifies |
|-----------|--------------|------------------|
| `test_core.py` | Tables 1, 2, 4, 5, §5.5 | Spectral pollution counts, Hodge splitting, universality across all structures including SC cubic |
| `test_convergence.py` | Table 3, Fig 1 | O(h²) mesh convergence (Kelvin N=2..6), standard oscillation |
| `test_robustness.py` | Table 6, §6 | k→0 asymptotics, random k-directions, mesh distortion, BZ boundary, face ordering (Prop 2), gauge transform, operator norms |
| `test_structure.py` | §5.5, §6 | Random Voronoi (10 seeds), N-scaling (N=2,3,4), scalar Laplacian clean, minimal counterexample |
| `make_figures.py` | Figs 2, 3 | Band structure Γ-X-M-R-Γ, mesh wireframes |

### test_core.py

| Test | Paper ref | What it checks |
|------|-----------|----------------|
| `analyze_structure` (Kelvin) | Table 2, 5 | ‖d₁d₀‖, n_spur, c², gradient leakage on Kelvin N=2 at 4 k-fractions × 2 directions |
| `analyze_structure` (C15) | Table 2, 5 | Same analysis on C15 Laves (V=136, E=272) |
| `analyze_structure` (Weaire-Phelan) | Table 2, 5 | Same analysis on Weaire-Phelan A15 (V=46, E=92) |
| `test_sc_cubic` | Table 1, 5, §5.5 | SC cubic N=3 (V=27, E=81): ‖d₁d₀‖_std=6.2, n_spur=5 [100] / 13 [111] |
| `test_hodge_splitting` | Table 4, §5.3 | Full Hodge Laplacian: mixed_exact=0 vs mixed_std=169–187 out of 192 modes |

### test_convergence.py

| Test | Paper ref | What it checks |
|------|-----------|----------------|
| `analyze_N` (N=2..6) | Table 3 | c² = ω²/\|k\|² for exact vs standard, first 3 modes at each mesh size |
| log-log fit | Fig 1, §5.2 | Convergence rate p=2.00, R²=1.0000 (second-order confirmed) |

### test_robustness.py

| Test | Paper ref | What it checks |
|------|-----------|----------------|
| `test_asymptotics` | Table 6 | c² stability from k/BZ=10⁻⁴ to 0.20; threshold sensitivity (4 relative thresholds) |
| `test_random_directions` | Table 6, §6 | n_spur for 3 axis + 20 random k-directions (n_zero_ex=96=V for all 23) |
| `test_mesh_distortion` | Table 6, §6 | Exactness under ε=0..0.10 geometric perturbation, 3 seeds each (all 12 PASS) |
| `test_bz_boundary` | Table 6 | Rank and exactness at k/BZ = 0.1, 0.5, 0.9, 0.999, 1.0 (DEGRADED only at exact BZ corner) |
| `test_face_ordering` | Prop 2 | K canonical under cycle, reverse, random vertex permutations (‖K−K_ref‖ < 10⁻¹⁴) |
| `test_gauge_transform` | Table 6 | Eigenvalues invariant under random vertex gauge transform (5 seeds, max\|Δeig\| < 10⁻¹⁴) |
| `test_operator_norm` | §6, Reproducibility | ‖K_std − K_exact‖_F grows from 2.1 to 33.8 as k/BZ goes from 1% to 30% |

### test_structure.py

| Test | Paper ref | What it checks |
|------|-----------|----------------|
| `test_random_voronoi` | Table 5, §5.5 | 10 random Voronoi seeds (n_cells=50): all 10/10 PASS, ‖d₁d₀‖ < 10⁻¹⁵ |
| `test_n_scaling` | §5.5, §6 | Spurious count grows with N (extensive pollution): N=2 → 6–14, N=3 → 17–38, N=4 → 29–72 |
| `test_scalar_laplacian` | §5.5 | Scalar Laplacian (level 0) has no pollution — problem specific to level 1→2 |
| `test_minimal_counterexample` | §5.5 | SC 1×1×1: standard = exact (d₁ unique, no room for pollution) |

### make_figures.py

| Figure | Paper ref | What it generates |
|--------|-----------|-------------------|
| `make_fig_bandstructure` | Fig 2 | Band structure Γ-X-M-R-Γ: exact (clean) vs standard (polluted), 121 k-points, 8 bands |
| `make_fig_meshes` | Fig 3 | 3D wireframes: Kelvin foam + random Voronoi |

## Structure

```
src/
├── physics/
│   ├── gauge_bloch.py      # Exactness-preserving d₁(k) construction
│   ├── hodge.py            # Voronoi complex builders + Hodge stars
│   ├── bloch.py            # Standard Bloch operators (for comparison)
│   └── constants.py        # Physical constants
└── core_math/
    ├── operators/
    │   └── incidence.py    # Topological d₀, d₁ (k=0)
    ├── builders/
    │   ├── solids.py       # Unit cell definitions
    │   └── solids_periodic.py  # Periodic supercell builders
    └── spec/
        ├── structures.py   # Mesh creation utilities
        └── constants.py    # Numerical constants

tests/
├── test_core.py           # Pollution + Hodge + universality + SC cubic (Tables 1, 2, 4, 5)
├── test_convergence.py    # Mesh refinement O(h²) (Table 3, Fig 1)
├── test_robustness.py     # k→0, random dirs, distortion, BZ, ordering, gauge, norms (Table 6, §6)
├── test_structure.py      # Random Voronoi, N-scaling, scalar, minimal (§5.5)
└── make_figures.py        # Band structure + mesh wireframes (Figs 2, 3)

paper/
├── draft_m5.tex               # LaTeX manuscript (elsarticle, JCP)
├── fig1_convergence_m5.pdf    # Figure 1: convergence plot
├── fig2_bandstructure_m5.pdf  # Figure 2: band structure Γ-X-M-R-Γ
└── fig3_meshes_m5.pdf         # Figure 3: mesh wireframes
```

## Key source file

The core contribution is in `src/physics/gauge_bloch.py`: the function `build_d1_bloch_exact()` implements the recurrence construction. Compare with `build_d1_bloch_standard()` in `src/physics/bloch.py` which breaks exactness at k ≠ 0.

## License

MIT
