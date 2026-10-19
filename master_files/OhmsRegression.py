"""
Regional regression for the generic Ohm's law closure

    E_tilde^a = R_tilde J_tilde^a + alpha B_tilde^a + gamma Hall^a + residual^a.

resMHD_3D.fit_ohms_law_closure fits R_tilde, alpha, gamma independently at every meso
gridpoint. With all three terms that fit is an identity (J, B and J x B span the three
spatial directions), so its residual says nothing about the closure. Here the coefficients
are instead constrained to be the same over a whole region: they are fitted to all the
points in the region at once by least squares (minimising the summed Minkowski norm of the
residual), so the model can genuinely fail.

Pure numpy and independent of the rest of the repo, so it can be tested on its own. All
vectors are 4-vectors orthogonal to u_tilde, so the Minkowski product (-+++) is positive
on them.
"""
import itertools

import numpy as np

TERMS = ('R_tilde', 'alpha_dynamo', 'gamma_hall')


def mink_dot(a, b):
    """Minkowski product (-+++) over the last axis."""
    return -a[..., 0] * b[..., 0] + np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def fit_region(E, ops, names):
    """
    Least-squares fit of one set of constant coefficients to all points given.

    E: array (..., 4); ops: dict term name -> array (..., 4) of the same shape;
    names: the terms to include (a subset of TERMS).

    Returns a dict with the coefficients (NaN for terms not included), 'n' (points),
    'rss' (summed squared residual norm), 'R2' = 1 - rss / sum|E|^2, and 'EE'. R2 is
    uncentred: the model has no intercept, so 0 means no better than predicting E = 0.
    """
    E = np.asarray(E, dtype=float).reshape(-1, 4)
    names = tuple(names)
    EE = float(mink_dot(E, E).sum())
    coeffs = {t: np.nan for t in TERMS}
    if names:
        O = [np.asarray(ops[t], dtype=float).reshape(-1, 4) for t in names]
        G = np.array([[mink_dot(a, b).sum() for b in O] for a in O])
        r = np.array([mink_dot(a, E).sum() for a in O])
        # lstsq: G is singular when operators are linearly dependent over the region
        c, *_ = np.linalg.lstsq(G, r, rcond=None)
        rss = EE - 2.0 * c @ r + c @ G @ c
        for t, v in zip(names, c):
            coeffs[t] = float(v)
    else:
        rss = EE
    rss = max(float(rss), 0.0)
    out = dict(coeffs)
    out.update({'n': len(E), 'rss': rss, 'EE': EE, 'names': names,
                'R2': (1.0 - rss / EE) if EE > 0 else np.nan})
    return out


def all_subsets():
    """Every non-empty subset of the three terms, smallest first."""
    return [s for k in range(1, len(TERMS) + 1) for s in itertools.combinations(TERMS, k)]


def subset_table(E, ops):
    """Fit every non-empty subset of the terms to the whole region."""
    return [fit_region(E, ops, s) for s in all_subsets()]


def _edges(n, b):
    """Tile boundaries along an axis of length n; a short trailing tile is merged into the last."""
    b = max(1, int(b))
    starts = list(range(0, n, b))
    edges = [(s, min(s + b, n)) for s in starts]
    if len(edges) > 1 and (edges[-1][1] - edges[-1][0]) < max(1, b // 2):
        edges[-2] = (edges[-2][0], edges[-1][1])
        edges.pop()
    return edges


def tile_fits(E5, ops5, names, block, min_points=1):
    """
    Fit constant coefficients separately in each tile of a (Nt, Nx, Ny, Nz, 4) meso grid.

    block = (bx, by, bz) tile size in meso cells; every time slice is included in a tile.
    Tiles with fewer than min_points points are skipped. Returns a list of fit dicts, each
    with an added 'tile' = ((x0, x1), (y0, y1), (z0, z1)).
    """
    Nt, Nx, Ny, Nz = E5.shape[:4]
    fits = []
    for (x0, x1), (y0, y1), (z0, z1) in itertools.product(
            _edges(Nx, block[0]), _edges(Ny, block[1]), _edges(Nz, block[2])):
        sl = (slice(None), slice(x0, x1), slice(y0, y1), slice(z0, z1))
        if Nt * (x1 - x0) * (y1 - y0) * (z1 - z0) < min_points:
            continue
        fit = fit_region(E5[sl], {t: o[sl] for t, o in ops5.items()}, names)
        fit['tile'] = ((x0, x1), (y0, y1), (z0, z1))
        fits.append(fit)
    return fits


def permutation_null(E, ops, names, n_perm=50, seed=0):
    """
    R2 of the same fit when E is paired with operators from randomly chosen other points.
    This is the chance level for a constant-coefficient fit with these operators.
    """
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=float).reshape(-1, 4)
    ops = {t: np.asarray(o, dtype=float).reshape(-1, 4) for t, o in ops.items()}
    return np.array([fit_region(E[rng.permutation(len(E))], ops, names)['R2'] for _ in range(n_perm)])


def shift_null(E5, ops5, names, n_perm=100, min_shift=3, seed=0):
    """
    Chance level of the fit that respects spatial correlation.

    permutation_null shuffles individual points, which destroys the spatial correlation of
    the fields and so understates how well unrelated, smooth fields can fit by chance. Here
    E is instead circularly shifted, as a whole, by a random offset of at least min_shift
    meso cells along each spatial axis that is long enough (an axis shorter than
    2*min_shift + 1 is left unshifted), keeping every field's internal structure but
    breaking the alignment between E and the operators. Time is not shifted. The shift
    wraps around the box, which adds a seam at the crop faces: acceptable for a null.

    E5: (Nt, Nx, Ny, Nz, 4); ops5: dict of arrays of the same shape.
    """
    rng = np.random.default_rng(seed)
    E5 = np.asarray(E5, dtype=float)
    lengths = E5.shape[1:4]
    shiftable = [n >= 2 * min_shift + 1 for n in lengths]
    if not any(shiftable):
        raise ValueError("grid %s too small to shift by at least %d cells" % (lengths, min_shift))
    out = []
    for _ in range(n_perm):
        shifts = [int(rng.integers(min_shift, n - min_shift + 1)) if ok else 0
                  for n, ok in zip(lengths, shiftable)]
        Es = np.roll(E5, shift=shifts, axis=(1, 2, 3))
        out.append(fit_region(Es, ops5, names)['R2'])
    return np.array(out)


def summarize(values):
    """median, 16th and 84th percentiles, and fraction positive of a set of tile values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {'median': np.nan, 'p16': np.nan, 'p84': np.nan, 'frac_pos': np.nan, 'n': 0}
    p16, med, p84 = np.percentile(v, [16, 50, 84])
    return {'median': med, 'p16': p16, 'p84': p84, 'frac_pos': float(np.mean(v > 0)), 'n': len(v)}


# ---------------------------------------------------------------------------------------
# Local (sliding-window) fits, for maps of which term describes E_tilde where
# ---------------------------------------------------------------------------------------

def box_sum(a, window):
    """
    Sum of a 3D array over a (wx, wy, wz) box (odd sizes) centred on each point; the box is
    clipped at the array edges. Uses cumulative sums, so it is O(N) whatever the window.
    """
    out = np.asarray(a, dtype=float)
    for axis, w in enumerate(window):
        n = out.shape[axis]
        h = int(w) // 2
        csum = np.concatenate([np.zeros_like(np.take(out, [0], axis=axis)), np.cumsum(out, axis=axis)], axis=axis)
        idx = np.arange(n)
        hi = np.minimum(idx + h + 1, n)
        lo = np.maximum(idx - h, 0)
        out = np.take(csum, hi, axis=axis) - np.take(csum, lo, axis=axis)
    return out


def local_term_maps(E5, ops5, window, min_fraction=0.5):
    """
    Constant-coefficient fits in a sliding spatial window around every meso point (all time
    slices are included in the window sums).

    E5, ops5[term]: (Nt, Nx, Ny, Nz, 4); window: odd (wx, wy, wz) in meso cells.
    Returns a dict of (Nx, Ny, Nz) arrays: 'R2' (3, ...) the single-term R2 for each of
    TERMS, 'coeff' (3, ...) the single-term coefficients, 'R2_full' for all three terms,
    and 'valid' (windows with at least min_fraction of their full volume inside the grid).
    """
    names = TERMS
    E5 = np.asarray(E5, dtype=float)
    O = [np.asarray(ops5[t], dtype=float) for t in names]

    def tsum(x):
        return box_sum(x.sum(axis=0), window)

    EE = tsum(mink_dot(E5, E5))
    r = np.stack([tsum(mink_dot(o, E5)) for o in O])
    G = np.empty(EE.shape + (3, 3))
    for i in range(3):
        for j in range(i, 3):
            G[..., i, j] = G[..., j, i] = tsum(mink_dot(O[i], O[j]))
    gdiag = np.stack([G[..., k, k] for k in range(3)])
    with np.errstate(divide='ignore', invalid='ignore'):
        R2 = r ** 2 / (gdiag * EE)
        coeff = r / gdiag
        c_full = np.einsum('...ij,...j->...i', np.linalg.pinv(G), np.moveaxis(r, 0, -1))
        R2_full = np.sum(c_full * np.moveaxis(r, 0, -1), axis=-1) / EE
    R2 = np.where((gdiag > 0) & (EE > 0), R2, np.nan)
    count = box_sum(np.ones(EE.shape), window)
    valid = count >= min_fraction * np.prod(window)
    return {'R2': R2, 'coeff': coeff, 'R2_full': R2_full, 'valid': valid}


def winner_map(R2, threshold):
    """Index into TERMS of the best single term (0, 1, 2); -1 where none exceeds threshold or R2 is NaN."""
    R2 = np.asarray(R2)
    best = np.nanargmax(np.where(np.isnan(R2), -np.inf, R2), axis=0)
    top = np.take_along_axis(R2, best[None], axis=0)[0]
    return np.where(np.isfinite(top) & (top >= threshold), best, -1)


def local_null_threshold(E5, ops5, window, n_null=20, min_shift=None, q=95.0, seed=0):
    """
    Chance level for the best single-term local R2: the q-th percentile, over valid points,
    of the best single-term R2 when E is circularly shifted (as in shift_null) so that its
    alignment with the operators is broken but its spatial structure is kept.
    """
    rng = np.random.default_rng(seed)
    E5 = np.asarray(E5, dtype=float)
    min_shift = int(max(window)) if min_shift is None else min_shift
    lengths = E5.shape[1:4]
    shiftable = [n >= 2 * min_shift + 1 for n in lengths]
    if not any(shiftable):
        raise ValueError("grid %s too small to shift by at least %d cells" % (lengths, min_shift))
    best = []
    for _ in range(n_null):
        shifts = [int(rng.integers(min_shift, n - min_shift + 1)) if ok else 0
                  for n, ok in zip(lengths, shiftable)]
        m = local_term_maps(np.roll(E5, shift=shifts, axis=(1, 2, 3)), ops5, window)
        top = np.nanmax(np.where(np.isnan(m['R2']), -np.inf, m['R2']), axis=0)
        best.append(top[m['valid'] & np.isfinite(top)])
    return float(np.percentile(np.concatenate(best), q))
