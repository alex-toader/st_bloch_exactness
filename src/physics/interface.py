"""Shared helpers for dielectric interface averaging.

Used by tests 15, 16. Consolidates log_mean and build_inv_eps_face
which were previously copy-pasted with minor variations.
"""
import numpy as np

_FORMULA_ALIASES = {'logarithmic': 'log'}


def log_mean(a, b):
    """Logarithmic mean: L(a,b) = (b-a)/ln(b/a). L(a,a) = a."""
    if abs(a - b) < 1e-14 * max(abs(a), abs(b)):
        return a
    return (b - a) / (np.log(b) - np.log(a))


def build_inv_eps_face(F, ftc, eps_cells, formula='log'):
    """Build per-face 1/ε array using specified averaging formula.

    Interface faces detected by |ε_A - ε_B| > 1e-12.

    Parameters
    ----------
    F : list
        Face list (used only for length).
    ftc : dict or array-like
        face_to_cells mapping: ftc[fi] = (cell_a, cell_b).
    eps_cells : array
        Per-cell permittivity.
    formula : str
        'log'/'logarithmic', 'harmonic', 'geometric', or 'arithmetic'.

    Returns
    -------
    inv_eps : ndarray
        Per-face 1/ε values.
    n_iface : int
        Number of interface faces (where ε differs across the face).
    """
    formula = _FORMULA_ALIASES.get(formula, formula)
    nF = len(F)
    inv_eps = np.zeros(nF)
    n_iface = 0
    for fi in range(nF):
        ca, cb = ftc[fi]
        ea, eb = float(eps_cells[ca]), float(eps_cells[cb])
        if abs(ea - eb) < 1e-12:
            inv_eps[fi] = 1.0 / ea
        else:
            n_iface += 1
            if formula == 'log':
                inv_eps[fi] = 1.0 / log_mean(ea, eb)
            elif formula == 'harmonic':
                inv_eps[fi] = 0.5 * (1.0 / ea + 1.0 / eb)
            elif formula == 'geometric':
                inv_eps[fi] = 1.0 / np.sqrt(ea * eb)
            elif formula == 'arithmetic':
                inv_eps[fi] = 2.0 / (ea + eb)
            else:
                raise ValueError(f"Unknown formula: {formula}")
    return inv_eps, n_iface
