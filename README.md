# st_bloch_exactness

Exactness-preserving discrete Maxwell operators on periodic polyhedral complexes.

## Problem

Standard Bloch-periodic DEC operators on unstructured polyhedral meshes break the discrete de Rham exactness property (d₁d₀ ≠ 0 at k ≠ 0). This produces spurious eigenvalues and zero acoustic modes in the curl-curl spectrum.

## Solution

A recurrence construction on face boundaries derives the curl operator d₁(k) from the gradient d₀(k), exploiting the flat holonomy of the Bloch connection. The resulting curl-curl operator K(k) = d₁(k)†⋆₂d₁(k) is canonical (independent of vertex ordering and face orientation) with gauge kernel dim = |V|.

## Results

Verified on 6 periodic polyhedral complexes:

| Structure | V | E | ‖d₁d₀‖ (exact) | ‖d₁d₀‖ (standard) | Acoustic modes |
|-----------|---|---|-----------------|--------------------|----|
| SC cubic | 27 | 81 | 0 | 5.5 | 2 (degenerate, c² = a²) |
| C15 (Laves) | 136 | 272 | 5×10⁻¹⁶ | 8.3 | 2 (degenerate) |
| Kelvin (BCC) | 96 | 192 | 6×10⁻¹⁶ | 7.4 | 2 (degenerate) |
| Weaire-Phelan | 46 | 92 | 5×10⁻¹⁶ | 5.1 | 2 (degenerate) |
| FCC | 12 | 32 | 0 | 5.5 | 2 (degenerate) |
| Random Voronoi | 206 | 412 | 10⁻¹⁵ | 6.2 | 2 (split ~10⁻⁴) |

Robustness: exactness and kernel dimension are topological invariants (robust under geometric, metric, and symmetry-breaking perturbations). Wave speed and polarization degeneracy are metric and symmetry properties respectively.

## Requirements

- Python 3.9+
- NumPy
- SciPy

## Running tests

```bash
cd tests/
OPENBLAS_NUM_THREADS=1 python3 test_exactness.py
OPENBLAS_NUM_THREADS=1 python3 test_ordering.py
OPENBLAS_NUM_THREADS=1 python3 test_fcc.py
OPENBLAS_NUM_THREADS=1 python3 test_random_voronoi.py
OPENBLAS_NUM_THREADS=1 python3 test_sc_convergence.py
```

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
├── test_exactness.py       # Core: exactness, holonomy, kernel, acoustic
├── test_ordering.py        # K canonical under vertex reordering
├── test_fcc.py             # Non-Plateau mesh verification
├── test_random_voronoi.py  # General complex (no symmetry)
└── test_sc_convergence.py  # Analytic benchmark + perturbation studies

paper/
└── draft_v04.md            # Manuscript draft
```

## Key source file

The core contribution is in `src/physics/gauge_bloch.py`: the function `build_d1_bloch_exact()` implements the recurrence construction. Compare with `build_d1_bloch_standard()` in `src/physics/bloch.py` which breaks exactness at k ≠ 0.

## License

MIT
