# st_bloch_exactness
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18704278.svg)](https://doi.org/10.5281/zenodo.18704278)

Exactness-preserving discrete Maxwell operators on periodic polyhedral complexes.

**Author:** Alexandru Toader (toader_alexandru@yahoo.com)

## Result

Standard Bloch-periodic DEC operators on unstructured polyhedral meshes break the discrete de Rham exactness property (d1*d0 != 0 at k != 0), producing spectral pollution, broken Hodge splitting, and convergence failure. A recurrence construction on face boundaries derives d1(k) from d0(k), exploiting the flat holonomy of the Bloch connection. The resulting curl-curl operator K(k) is canonical (independent of vertex ordering and face orientation) with exact gauge kernel dim = |V|.

Verified on SC cubic, Kelvin (BCC), C15 (Laves), Weaire-Phelan, and random Voronoi complexes. Mesh convergence at O(h^2). Extensions to dielectric contrast, Bloch cohomology, spectral pairing, and Hodge decomposition.

## Paper

Submitted to Journal of Computational Physics (Mar 2026) as JCOMP-D-26-00537.
Companion: JCOMP-D-26-00678 (st_voronoi_maxwell).

- `paper/bloch_exactness_jcp.tex` -- manuscript
- `paper/st_bloch_exactness_m5.pdf` -- compiled PDF

## Tests

20 files, 89 test functions. Each test file maps to a paper section.

See `tests/tests_map.md` for the complete inventory with per-test descriptions.

```
tests/
├── 1_test_core.py                (3 tests)   Pollution + Hodge + universality (Tables 1,2,4,5)
├── 2_test_convergence.py         (3 tests)   Mesh convergence O(h^2) (Table 3, Fig 1)
├── 3_test_robustness.py          (7 tests)   k->0, random dirs, distortion, BZ, gauge (Table 6)
├── 4_test_structure.py           (5 tests)   Random Voronoi, N-scaling, scalar, minimal
├── 5_test_r16_oscillations.py    (7 tests)   Standard oscillations, k-scaling, Hodge defect
├── 7_test_ker_theorem.py         (2 tests)   Kernel chain: rank(d0)=ker(d1)=ker(K)=V
├── 8_test_cor1_proof.py          (4 tests)   Corollary 1: uniqueness + intra-face contradictions
├── 9_test_d2_exactness.py        (3 tests)   d2*d1 exactness at level 2->3
├── 10_test_dielectric.py         (4 tests)   Scalar + tensor dielectric, mu != 1
├── 11_test_h_refinement.py       (3 tests)   h-refinement on Kelvin + random Voronoi
├── 12_test_mpb_comparison.py     (5 tests)   DEC vs MPB reference (requires meep/mpb)
├── 13_test_bloch_cohomology.py   (4 tests)   Betti numbers, BZ scan, rank identities
├── 14_test_acoustic_deformation.py (4 tests) Harmonic forms, Helmholtz splitting, acoustic speed
├── 15_test_hodge_star_interface.py (8 tests) Interface Hodge star formulas vs MPB
├── 16_test_voronoi_dielectric.py (7 tests)   Dielectric on random Voronoi meshes
├── 17_test_face_normal_statistics.py (5 tests) Orientation-weighted vs log mean formulas
├── 18_test_spectral_pairing.py   (5 tests)   Hodge Laplacian spectral pairing
├── 19_test_hodge_decomposition.py (5 tests)  SVD-based Hodge decomposition
├── 20_test_perturbation_stability.py (5 tests) Exactness under geometric perturbation
├── 6_make_figures.py             (2 figs)    Band structure + mesh wireframes (Figs 2,3)
└── tests_map.md                              Complete test inventory
```

## Source code

```
src/
├── physics/
│   ├── gauge_bloch.py        Exactness-preserving d1(k) construction
│   ├── hodge.py              Voronoi complex builders + Hodge stars
│   ├── bloch.py              Standard Bloch operators (for comparison)
│   └── constants.py          Physical constants
└── core_math/
    ├── operators/incidence.py    Topological d0, d1 (k=0)
    ├── builders/solids.py        Unit cell definitions
    ├── builders/solids_periodic.py   Periodic supercell builders
    └── spec/                     Mesh contract + constants
```

The core contribution is in `src/physics/gauge_bloch.py`: the function `build_d1_bloch_exact()` implements the recurrence construction. Compare with `build_d1_bloch_standard()` in `src/physics/bloch.py` which breaks exactness at k != 0.

## Running tests

```bash
# All tests except MPB comparison (~2 min)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest tests/ -v --ignore=tests/12_test_mpb_comparison.py

# MPB comparison (requires meep/mpb conda environment)
/Users/alextoader/miniconda3/envs/mpb_env/bin/python -m pytest tests/12_test_mpb_comparison.py -v
```

## Requirements

- Python 3.9+
- NumPy, SciPy
- Matplotlib (for figure generation only)
- meep/mpb (for test 12 only, via conda-forge)

## License

MIT
