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


def hall_operator(J, u, B, levi_upper, metric):
    """
    Hall^a = eps^{abcd} J_b u_c B_d (covariant: the vectors are lowered with the metric before the
    contraction). levi_upper = eps^{abcd}, e.g. Base.raise_LeviCivita4D(metric). The result is
    orthogonal to u exactly, and for the static observer u = (1,0,0,0) its spatial part is J x B.
    (Contracting eps^{abcd} with the UPPER-index components, as an earlier version did, gives a vector
    that is not orthogonal to a moving observer and so cannot represent a term of E~, which is.)
    """
    lower = lambda x: np.einsum('ab,...b->...a', metric, x)
    return np.einsum('abcd,...b,...c,...d->...a', levi_upper, lower(J), lower(u), lower(B))


def standard_operators(J, B, u, levi_upper, metric):
    """The three Ohm's-law operators {R_tilde: J, alpha_dynamo: B, gamma_hall: Hall}."""
    return {'R_tilde': J, 'alpha_dynamo': B, 'gamma_hall': hall_operator(J, u, B, levi_upper, metric)}


def mink_dot(a, b):
    """Minkowski product (-+++) over the last axis."""
    return -a[..., 0] * b[..., 0] + np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def _wdot(a, b, weights=None):
    """Sum over the last axis of w * a * b; Minkowski (-+++) when weights is None."""
    if weights is None:
        return mink_dot(a, b)
    return np.sum(np.asarray(weights) * a * b, axis=-1)


def _solve_normal(G, r, EE):
    """
    Least squares from the Gram matrix G = O_k.O_l and r = O_k.E. Columns are scaled to unit norm
    first, so operators of very different size (J~ is ~1e-6 B~ here) do not make the system badly
    conditioned; lstsq still handles linearly dependent or all-zero operators. Returns (c, rss).
    """
    d = np.sqrt(np.clip(np.diag(G), 0.0, None))
    d = np.where(d > 0, d, 1.0)
    c, *_ = np.linalg.lstsq(G / np.outer(d, d), r / d, rcond=None)
    c = c / d
    return c, max(float(EE - 2.0 * c @ r + c @ G @ c), 0.0)


def _fit_result(names, c, rss, EE, n):
    coeffs = {t: np.nan for t in TERMS}
    for t, v in zip(names, c):
        coeffs[t] = float(v)
    out = dict(coeffs)
    out.update({'n': n, 'rss': rss, 'EE': EE, 'names': tuple(names),
                'R2': (1.0 - rss / EE) if EE > 0 else np.nan})
    return out


def fit_region(E, ops, names, weights=None):
    """
    Least-squares fit of one set of constant coefficients to all points given.

    E: array (..., 4); ops: dict term name -> array (..., 4) of the same shape;
    names: the terms to include (a subset of TERMS).

    Returns a dict with the coefficients (NaN for terms not included), 'n' (points),
    'rss' (summed squared residual norm), 'R2' = 1 - rss / sum|E|^2, and 'EE'. R2 is
    uncentred: the model has no intercept, so 0 means no better than predicting E = 0.

    weights: optional length-k weights for the product (the default is the Minkowski product on 4-vectors);
    with weights the last axis can hold any k components, e.g. 16 flattened tensor components.
    """
    E = np.asarray(E, dtype=float)
    k = E.shape[-1]
    E = E.reshape(-1, k)
    names = tuple(names)
    EE = float(_wdot(E, E, weights).sum())
    if not names:
        return _fit_result(names, [], EE, EE, len(E))
    O = [np.asarray(ops[t], dtype=float).reshape(-1, k) for t in names]
    G = np.array([[_wdot(a, b, weights).sum() for b in O] for a in O])
    r = np.array([_wdot(a, E, weights).sum() for a in O])
    c, rss = _solve_normal(G, r, EE)
    return _fit_result(names, c, rss, EE, len(E))


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


def tile_fits(E5, ops5, names, block, min_points=1, weights=None):
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
        fit = fit_region(E5[sl], {t: o[sl] for t, o in ops5.items()}, names, weights)
        fit['tile'] = ((x0, x1), (y0, y1), (z0, z1))
        fits.append(fit)
    return fits


def _shift_setup(lengths, min_shift):
    """Which spatial axes are long enough to shift by at least min_shift cells (raises if none are)."""
    shiftable = [n >= 2 * min_shift + 1 for n in lengths]
    if not any(shiftable):
        raise ValueError("grid %s too small to shift by at least %d cells" % (tuple(lengths), min_shift))
    return shiftable


def _draw_shifts(rng, lengths, shiftable, min_shift):
    """One random circular shift per spatial axis (0 on axes that are too short)."""
    return [int(rng.integers(min_shift, n - min_shift + 1)) if ok else 0 for n, ok in zip(lengths, shiftable)]


def permutation_null(E, ops, names, n_perm=50, seed=0):
    """
    R2 of the same fit when E is paired with operators from randomly chosen other points.
    This is the chance level for a constant-coefficient fit with these operators.
    """
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=float).reshape(-1, 4)
    ops = {t: np.asarray(o, dtype=float).reshape(-1, 4) for t, o in ops.items()}
    return np.array([fit_region(E[rng.permutation(len(E))], ops, names)['R2'] for _ in range(n_perm)])


def shift_null(E5, ops5, names, n_perm=100, min_shift=3, seed=0, weights=None):
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
    shiftable = _shift_setup(lengths, min_shift)
    names = tuple(names)
    k = E5.shape[-1]
    if not names:
        return np.array([fit_region(E5, ops5, names, weights)['R2'] for _ in range(n_perm)])
    # The operators and the total |E|^2 do not change under a shift, so the Gram matrix is built once
    O = [np.asarray(ops5[t], dtype=float).reshape(-1, k) for t in names]
    G = np.array([[_wdot(a, b, weights).sum() for b in O] for a in O])
    EE = float(_wdot(E5.reshape(-1, k), E5.reshape(-1, k), weights).sum())
    out = []
    for _ in range(n_perm):
        Es = np.roll(E5, shift=_draw_shifts(rng, lengths, shiftable, min_shift), axis=(1, 2, 3)).reshape(-1, k)
        r = np.array([_wdot(a, Es, weights).sum() for a in O])
        _, rss = _solve_normal(G, r, EE)
        out.append((1.0 - rss / EE) if EE > 0 else np.nan)
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


def _local_operator_sums(ops5, window):
    """Windowed Gram matrix of the three operators, G (Nx,Ny,Nz,3,3), and its diagonal gdiag (3,Nx,Ny,Nz)."""
    O = [np.asarray(ops5[t], dtype=float) for t in TERMS]
    G = np.empty(O[0].shape[1:4] + (3, 3))
    for i in range(3):
        for j in range(i, 3):
            G[..., i, j] = G[..., j, i] = box_sum(mink_dot(O[i], O[j]).sum(axis=0), window)
    return O, G, np.stack([G[..., k, k] for k in range(3)])


def _local_E_sums(E5, O, window):
    """Windowed |E|^2 (Nx,Ny,Nz) and the windowed O_k.E, r (3,Nx,Ny,Nz)."""
    EE = box_sum(mink_dot(E5, E5).sum(axis=0), window)
    return EE, np.stack([box_sum(mink_dot(o, E5).sum(axis=0), window) for o in O])


def local_term_maps(E5, ops5, window, min_fraction=0.5):
    """
    Constant-coefficient fits in a sliding spatial window around every meso point (all time
    slices are included in the window sums).

    E5, ops5[term]: (Nt, Nx, Ny, Nz, 4); window: odd (wx, wy, wz) in meso cells.
    Returns a dict of (Nx, Ny, Nz) arrays: 'R2' (3, ...) the single-term R2 for each of
    TERMS, 'coeff' (3, ...) the single-term coefficients, 'R2_full' for all three terms,
    and 'valid' (windows with at least min_fraction of their full volume inside the grid).
    """
    E5 = np.asarray(E5, dtype=float)
    O, G, gdiag = _local_operator_sums(ops5, window)
    EE, r = _local_E_sums(E5, O, window)
    with np.errstate(divide='ignore', invalid='ignore'):
        R2 = r ** 2 / (gdiag * EE)
        coeff = r / gdiag
        c_full = np.einsum('...ij,...j->...i', np.linalg.pinv(G), np.moveaxis(r, 0, -1))
        R2_full = np.sum(c_full * np.moveaxis(r, 0, -1), axis=-1) / EE
    R2 = np.where((gdiag > 0) & (EE > 0), R2, np.nan)
    coeff = np.where(gdiag > 0, coeff, np.nan)           # same mask as R2: never inf/NaN-by-accident downstream
    R2_full = np.where(EE > 0, R2_full, np.nan)
    count = box_sum(np.ones(EE.shape), window)
    valid = count >= min_fraction * np.prod(window)
    return {'R2': R2, 'coeff': coeff, 'R2_full': R2_full, 'valid': valid}


def winner_map(R2, threshold):
    """Index into TERMS of the best single term (0, 1, 2); -1 where none exceeds threshold or R2 is NaN."""
    R2 = np.asarray(R2)
    best = np.nanargmax(np.where(np.isnan(R2), -np.inf, R2), axis=0)
    top = np.take_along_axis(R2, best[None], axis=0)[0]
    return np.where(np.isfinite(top) & (top >= threshold), best, -1)


def local_null_threshold(E5, ops5, window, n_null=20, min_shift=None, q=95.0, seed=0, valid=None):
    """
    Chance level for the best single-term local R2: the q-th percentile, over valid points,
    of the best single-term R2 when E is circularly shifted (as in shift_null) so that its
    alignment with the operators is broken but its spatial structure is kept.

    valid: optional boolean (Nx, Ny, Nz) mask of the points that count (default: windows mostly inside
    the grid, as local_term_maps; a mosaic passes the points that are covered by a crop).
    """
    rng = np.random.default_rng(seed)
    E5 = np.asarray(E5, dtype=float)
    min_shift = int(max(window)) if min_shift is None else min_shift
    lengths = E5.shape[1:4]
    shiftable = _shift_setup(lengths, min_shift)
    O, _, gdiag = _local_operator_sums(ops5, window)        # shift-invariant: built once
    if valid is None:
        valid = box_sum(np.ones(lengths), window) >= 0.5 * np.prod(window)
    best = []
    for _ in range(n_null):
        Es = np.roll(E5, shift=_draw_shifts(rng, lengths, shiftable, min_shift), axis=(1, 2, 3))
        EE, r = _local_E_sums(Es, O, window)
        with np.errstate(divide='ignore', invalid='ignore'):
            R2 = np.where((gdiag > 0) & (EE > 0), r ** 2 / (gdiag * EE), np.nan)
        top = np.nanmax(np.where(np.isnan(R2), -np.inf, R2), axis=0)
        best.append(top[valid & np.isfinite(top)])
    return float(np.percentile(np.concatenate(best), q))


# ---------------------------------------------------------------------------------------
# Merging meso grids from several crops into one mosaic
# ---------------------------------------------------------------------------------------

def merge_meso_tiles(tiles, spacing, tol=0.02):
    """
    Place the meso fields of several crops on one global lattice.

    tiles: list of dicts with 'X', 'Y', 'Z' (1D coordinate arrays of the tile's meso points)
    and 'fields' {name: array (Nt, Nx, Ny, Nz, ...)}; spacing = (Dx, Dy, Dz) of the meso grid.
    All tiles must sit on a common lattice (coordinates differing by whole multiples of the
    spacing) and have the same Nt. Where tiles overlap the later one wins. Returns a dict with
    the global axes 'X', 'Y', 'Z', the merged 'fields' (zero where no tile covers a point) and
    'present', a boolean (Nx, Ny, Nz) mask of covered points.
    """
    axes = []
    for k, key in enumerate('XYZ'):
        lo = min(float(np.min(t[key])) for t in tiles)
        hi = max(float(np.max(t[key])) for t in tiles)
        axes.append((lo, int(round((hi - lo) / spacing[k])) + 1))
    idx = []
    for t in tiles:
        ti = []
        for k, key in enumerate('XYZ'):
            f = (np.asarray(t[key], dtype=float) - axes[k][0]) / spacing[k]
            if np.max(np.abs(f - np.rint(f))) > tol:
                raise ValueError("tiles are not on a common lattice along %s" % key)
            ti.append(np.rint(f).astype(int))
        idx.append(ti)
    names = list(tiles[0]['fields'])
    Nt = tiles[0]['fields'][names[0]].shape[0]
    shape = tuple(n for _, n in axes)
    merged = {n: np.zeros((Nt,) + shape + tiles[0]['fields'][n].shape[4:]) for n in names}
    present = np.zeros(shape, dtype=bool)
    for t, (ix, iy, iz) in zip(tiles, idx):
        sel = np.ix_(ix, iy, iz)
        for n in names:
            if t['fields'][n].shape[0] != Nt:
                raise ValueError("tiles have different numbers of time slices")
            merged[n][(slice(None),) + sel] = t['fields'][n]
        present[sel] = True
    return {'X': axes[0][0] + spacing[0] * np.arange(shape[0]),
            'Y': axes[1][0] + spacing[1] * np.arange(shape[1]),
            'Z': axes[2][0] + spacing[2] * np.arange(shape[2]),
            'fields': merged, 'present': present}


# ---------------------------------------------------------------------------------------
# Gradient (second-order) model of E_tilde from the filter's own covariance
# ---------------------------------------------------------------------------------------
#
# Setup: micro fields obey ideal MHD, E = -v x B (lab frame), and F^{0i} = E^i. The meso
# observer is u~ = gamma (1, v~) with v~ = <D v>/<D> (Favre average, weight D = n u^0 = the
# lab-frame density, which is how the code defines u~). Write <F> as the ideal-MHD tensor built
# from v~ and <B>, plus a purely electric remainder dE = <E> + v~ x <B>. The ideal part has zero
# electric field for u~, so exactly
#
#     E~^a = u~_b <F^{ab}> = gamma (v~ . dE, dE),
#     dE   = -<dv x dB> - (<v> - v~) x <B>,    <v> - v~ = -cov(D, v)/<D>.
#
# The filter (Filters.box_filter_parallel) averages over a 3D SPATIAL box in the observer's rest space:
# 27 Gauss points at point + sum_i c_i e_i, with e_i the three spatial legs of the tetrad orthogonal to u
# (the time-like leg u is not averaged over). To second order in its width, the covariance of two
# fluctuating fields is
#
#     cov(a, b) = sigma^2 [ grad a . grad b - d_t a d_t b + (u.d a)(u.d b) ],
#
# with sigma^2 = W^2/12 per box direction (the 3-point quadrature reproduces the top-hat's moments through
# 5th order) and u.d = gamma (d_t + v.grad). This uses sum_i e_i e_i = g + u u, so it does not depend on how
# the spatial axes of the box are oriented. For a static observer it reduces to sigma^2 grad a . grad b.
# (An earlier version assumed a 4D box, giving g + 2uu, i.e. a factor 2 on the last term: that was wrong.)
# Hence
#
#     dE = -g1 + g2,   g1 = sigma^2 S(v, B) [cross products],   g2 = sigma^2 (S(D, v)/D) x <B>,
#
# where S(a, b) is the bracket above with the product replaced by the cross product (vector,
# vector) or scalar x vector. No free constant remains, so E~ ~= -O(g1) + O(g2), with
# O(g) = gamma (v~ . g, g), is a prediction. Regressing E~ on O(g1), O(g2) should give
# coefficients near -1 and +1 if the second-order model is the whole story.

def lab_derivatives(f, spacings, dt=None):
    """[d_t f, d_x f, d_y f, d_z f] for f of shape (Nt, Nx, Ny, Nz, ...). d_t is zero unless dt is given and Nt >= 3."""
    d0 = np.gradient(f, dt, axis=0) if (dt and f.shape[0] >= 3) else np.zeros_like(f)
    return [d0] + [np.gradient(f, spacings[l], axis=l + 1) for l in range(3)]


def _expand(x, like):
    return x.reshape(x.shape + (1,) * (like.ndim - x.ndim))


def covariance_bracket_parts(a, b, v, gamma, spacings, dt, product):
    """
    The four pieces of the covariance bracket, split by the kind of derivative:
      'ss' = sum_l d_l a (x) d_l b                        (spatial gradients, from the metric)
      'tt' = (gamma^2 - 1) d_t a (x) d_t b                (time derivatives)
      'ts' = gamma^2 [d_t a (x) (v.grad) b + (v.grad) a (x) d_t b]      (time-advective cross terms)
      'vv' = gamma^2 (v.grad) a (x) (v.grad) b            (advective)
    Their sum is  grad a . grad b - d_t a d_t b + gamma^2 (Dt a)(Dt b)  with Dt = d_t + v.grad
    (= grad a . grad b for a static observer).
    The time pieces are zero when dt is None or Nt < 3.
    """
    da, db = lab_derivatives(a, spacings, dt), lab_derivatives(b, spacings, dt)

    def advect(d):
        return sum(_expand(v[..., l], d[0]) * d[l + 1] for l in range(3))

    tt0 = product(da[0], db[0])
    g2 = _expand(gamma ** 2, tt0)
    return {'ss': sum(product(da[l + 1], db[l + 1]) for l in range(3)),
            'tt': (g2 - 1.0) * tt0,
            'ts': g2 * (product(da[0], advect(db)) + product(advect(da), db[0])),
            'vv': g2 * product(advect(da), advect(db))}


def covariance_bracket(a, b, v, gamma, spacings, dt, product):
    """
    sum_{mu nu} (g^{mu nu} + u^mu u^nu) d_mu a (x) d_nu b = the sum of covariance_bracket_parts.
    Multiply by sigma^2 to get the second-order covariance.
    """
    return sum(covariance_bracket_parts(a, b, v, gamma, spacings, dt, product).values())


def _cross(x, y):
    return np.cross(x, y)


def _scalar_times_vector(x, y):
    return x[..., None] * y


def observer_electric(g, v, gamma):
    """The 4-vector gamma (v.g, g): the electric field g as seen by an observer u = gamma (1, v)."""
    out = np.zeros(g.shape[:-1] + (4,))
    out[..., 0] = gamma * np.sum(v * g, axis=-1)
    out[..., 1:] = gamma[..., None] * g
    return out


def gradient_model_operators(v, B, D, u, spacings, sigma2, dt=None):
    """
    Second-order model of E_tilde (see the derivation above).

    v: (Nt,Nx,Ny,Nz,3) lab-frame meso velocity u~^i/u~^0; B: same shape, lab-frame filtered B;
    D: (Nt,Nx,Ny,Nz) filtered lab-frame density <n u^0>; u: (...,4) the meso observer u~;
    spacings: meso (Dx, Dy, Dz); sigma2 = W^2/12; dt: meso time spacing (time derivatives are
    set to zero if None or Nt < 3, which leaves the model incomplete).

    Returns {'model_vB': O(g1) (expected coefficient -1), 'model_dens': O(g2) (expected +1),
    'prediction': -O(g1) + O(g2)}, each shape (...,4).
    """
    gamma = u[..., 0]
    g1 = sigma2 * covariance_bracket(v, B, v, gamma, spacings, dt, _cross)
    T = covariance_bracket(D, v, v, gamma, spacings, dt, _scalar_times_vector)
    g2 = sigma2 * np.cross(T / D[..., None], B)
    O1, O2 = observer_electric(g1, v, gamma), observer_electric(g2, v, gamma)
    return {'model_vB': O1, 'model_dens': O2, 'prediction': -O1 + O2}


def gradient_model_parts(v, B, D, u, spacings, sigma2, dt=None):
    """
    The gradient-model operators split into the four covariance pieces ('ss', 'tt', 'ts', 'vv'):
    keys 'vB_<piece>' (each with expected coefficient -1) and 'dens_<piece>' (expected +1), each a 4-vector
    of shape (...,4). Summing the 'vB_' pieces gives gradient_model_operators()['model_vB'], and likewise
    for 'dens_' and 'model_dens'.
    """
    gamma = u[..., 0]
    pv = covariance_bracket_parts(v, B, v, gamma, spacings, dt, _cross)
    pd = covariance_bracket_parts(D, v, v, gamma, spacings, dt, _scalar_times_vector)
    out = {}
    for k, g in pv.items():
        out['vB_' + k] = observer_electric(sigma2 * g, v, gamma)
    for k, T in pd.items():
        out['dens_' + k] = observer_electric(sigma2 * np.cross(T / D[..., None], B), v, gamma)
    return out


def prediction_r2(E, pred, weights=None):
    """R2 of a parameter-free prediction: 1 - |E - pred|^2 / |E|^2 (can be negative). See fit_region for weights."""
    E = np.asarray(E, dtype=float)
    k = E.shape[-1]
    E = E.reshape(-1, k)
    pred = np.asarray(pred, dtype=float).reshape(-1, k)
    EE = _wdot(E, E, weights).sum()
    r = E - pred
    return float(1.0 - _wdot(r, r, weights).sum() / EE) if EE > 0 else float('nan')


def subgrid_B_energy(BSq_filt, B_lab_tilde):
    """<B^2> - |<B>|^2 from the filtered B^2 and the filtered lab-frame B (>= 0 for a non-negative filter)."""
    return np.asarray(BSq_filt) - np.sum(np.asarray(B_lab_tilde) ** 2, axis=-1)


def prediction_shift_null(E5, pred5, n_perm=50, min_shift=3, seed=0, weights=None):
    """
    Chance level for prediction_r2: R2 of the (parameter-free) prediction when E is circularly
    shifted relative to it, keeping spatial structure (as shift_null). Time is not shifted.
    """
    rng = np.random.default_rng(seed)
    E5 = np.asarray(E5, dtype=float)
    lengths = E5.shape[1:4]
    shiftable = _shift_setup(lengths, min_shift)
    return np.array([prediction_r2(np.roll(E5, shift=_draw_shifts(rng, lengths, shiftable, min_shift), axis=(1, 2, 3)),
                                   pred5, weights) for _ in range(n_perm)])


# ---------------------------------------------------------------------------------------
# Stress-energy tensor: second-order model of the sub-filter stress and heat flux
# ---------------------------------------------------------------------------------------
#
# Micro: T^{ab} = s n^a n^b + p g^{ab}, with s = w/n^2 (w = n h the enthalpy density) and n^a = n u^a.
# Filtering, to second order in the filter width:
#
#     <T^{ab}> = s_ n^a n^b + p_ g^{ab} + C^{ab},
#     C^{ab}   = s_ cov(n^a, n^b) + N^b cov(s, n^a) + N^a cov(s, n^b),      N^a = <n^a>,
#
# with cov(f, g) = sigma^2 * covariance_bracket(f, g) (the same filter covariance as for E~). Relative to the
# Favre observer u~ = N/|N| (so N^a = n~ u~^a), with h^{ab} = g^{ab} + u~^a u~^b and the usual definitions
#
#     eps~ = u_a u_b T^{ab},  p~ = h_ab T^{ab}/3,  q^a = -h^a_c u_d T^{cd},  pi^{ab} = h^a_c h^b_d T^{cd} - p~ h^{ab},
#
# the ideal terms drop out of q and pi (h N = 0, h g h = h), leaving
#
#     pi^{ab}_pred = s~ TF[h X h]^{ab},               X^{cd} = sigma^2 covariance_bracket(N, N; outer),
#     q^a_pred     = -s~ h^a_c X^{cd} u_d  +  n~ h^a_c Y^c,   Y^c = sigma^2 covariance_bracket(s, N),
#
# s~ = (eps~ + p~)/n~^2, TF = trace-free part with respect to h. No fitted constants: expected coefficient
# +1 for each of pi_pred, q1 (the first q term) and q2 (the second).

def _inverse_metric(metric):
    return np.linalg.inv(np.asarray(metric, dtype=float))


def decompose_set(T, u, metric):
    """
    Split T^{ab} (shape (...,4,4), upper indices) relative to the observer u^a (...,4) into
    eps, p, q^a (...,4), pi^{ab} (...,4,4), following the definitions above. Exact: for
    T = eps u u + p h + q u + u q + pi the inputs are recovered.
    """
    g = np.asarray(metric, dtype=float)
    ginv = _inverse_metric(g)
    ul = np.einsum('ab,...b->...a', g, u)
    hup = ginv + u[..., :, None] * u[..., None, :]                       # h^{ab}
    hmix = np.eye(4) + u[..., :, None] * ul[..., None, :]                # h^a_c
    hlow = g + ul[..., :, None] * ul[..., None, :]                       # h_{ab}
    eps = np.einsum('...a,...b,...ab->...', ul, ul, T)
    p = np.einsum('...ab,...ab->...', hlow, T) / 3.0
    q = -np.einsum('...ac,...d,...cd->...a', hmix, ul, T)
    pi = np.einsum('...ac,...bd,...cd->...ab', hmix, hmix, T) - p[..., None, None] * hup
    return {'eps': eps, 'p': p, 'q': q, 'pi': pi}


def stress_model(N, s, u, v, spacings, sigma2, metric, dt=None):
    """
    Second-order model of the sub-filter anisotropic stress and heat flux (see the derivation above).

    N: filtered baryon current <n^a> (Nt,Nx,Ny,Nz,4); s: filtered-field estimate (eps~ + p~)/n~^2
    (Nt,Nx,Ny,Nz); u: meso observer u~ (...,4); v = u~^i/u~^0 (...,3), the advection velocity; sigma2 = W^2/12;
    metric: 4x4. Returns {'pi': ..., 'q1': ..., 'q2': ...} (pi shape (...,4,4), q shape (...,4)), each with
    expected coefficient +1; the predicted heat flux is q1 + q2.
    """
    g = np.asarray(metric, dtype=float)
    ginv = _inverse_metric(g)
    gamma = u[..., 0]
    ul = np.einsum('ab,...b->...a', g, u)
    hup = ginv + u[..., :, None] * u[..., None, :]
    hmix = np.eye(4) + u[..., :, None] * ul[..., None, :]
    hlow = g + ul[..., :, None] * ul[..., None, :]
    ntil = np.sqrt(-np.einsum('...a,...a->...', N, np.einsum('ab,...b->...a', g, N)))

    outer = lambda x, y: x[..., :, None] * y[..., None, :]
    X = sigma2 * covariance_bracket(N, N, v, gamma, spacings, dt, outer)
    Y = sigma2 * covariance_bracket(s, N, v, gamma, spacings, dt, _scalar_times_vector)

    hXh = np.einsum('...ac,...bd,...cd->...ab', hmix, hmix, X)
    trace = np.einsum('...ab,...ab->...', hlow, hXh) / 3.0
    pi = s[..., None, None] * (hXh - trace[..., None, None] * hup)
    q1 = -s[..., None] * np.einsum('...ac,...cd,...d->...a', hmix, X, ul)
    q2 = ntil[..., None] * np.einsum('...ac,...c->...a', hmix, Y)
    return {'pi': pi, 'q1': q1, 'q2': q2}


def set_weights(metric):
    """Weights for the flattened (16) tensor product pi_ab pi^ab with a diagonal metric: sign(g_aa g_bb)."""
    d = np.sign(np.diag(np.asarray(metric, dtype=float)))
    return np.outer(d, d).reshape(16)


# ---------------------------------------------------------------------------------------
# Sub-filter Lorentz force F_closure = <f^b> - f~^b
# ---------------------------------------------------------------------------------------
#
# Micro f^b = -j_a F^{ab}. To second order, <f^b> = -j~_a F~^{ab} - C^b with
# C^b = sigma^2 covariance_bracket(j, F)_a^{ab}. The pipeline defines F_closure = <f^b> + J~_a F~^{ab}, with
# J~ the part of j~ orthogonal to u~; j~ = sigma~ u~ + J~ (sigma~ = -u~.j~), so exactly
#
#     F_closure^b = -C^b  -  sigma~ u~_a F~^{ab}.
#
# The second term is the force on the net charge, known from the filtered fields; the first is the prediction,
# with no fitted constant (expected coefficient +1). It involves the meso-grid gradient of j~, itself a
# derivative of F~, so it is the noisiest of the closure tests.

def lorentz_force_model(j, F, u, v, spacings, sigma2, metric, dt=None):
    """
    j: filtered current <j^a> (Nt,Nx,Ny,Nz,4); F: filtered <F^{ab}> (...,4,4); u: meso observer u~ (...,4);
    v = u~^i/u~^0 (...,3). Returns {'cov': -C^b (expected coefficient +1), 'charge': -sigma~ u~_a F~^{ab}
    (exact, coefficient +1)}, each (...,4); F_closure = cov + charge up to higher order.
    """
    g = np.asarray(metric, dtype=float)
    gamma = u[..., 0]

    def lorentz_contraction(x, y):                       # g_{ac} x^c y^{ab}
        return np.einsum('ac,...c,...ab->...b', g, x, y)

    C = sigma2 * covariance_bracket(j, F, v, gamma, spacings, dt, lorentz_contraction)
    sigma_t = -np.einsum('...a,ab,...b->...', u, g, j)
    charge = -sigma_t[..., None] * np.einsum('ac,...c,...ab->...b', g, u, F)
    return {'cov': -C, 'charge': charge}


# ---------------------------------------------------------------------------------------
# Sub-filter electromagnetic flux deficits for a coarse (LES) run: EMF, Maxwell stress, Poynting flux
# ---------------------------------------------------------------------------------------
#
# Lab frame, flat metric, ideal micro fields E = -v x B. The resolved fields are v (the lab-filtered velocity) and B; E_r = -v x B.
# To second order, with cov(a, b) = sigma^2 sum_l d_l a d_l b (spatial pieces only),
#
#   <E>      = E_r + EMF,                    EMF = -sigma^2 sum_l d_l v x d_l B,
#   <E x B>  = E_r x B + EMF x B + sigma^2 sum_l d_l E x d_l B,
#   <T^ij>   = T^ij(E_r, B) + Delta^ij,
#   Delta^ij = -(E_r^i EMF^j + EMF^i E_r^j) + delta^ij E_r.EMF + sigma^2 sum_l [-d_l E^i d_l E^j - d_l B^i d_l B^j
#                                                                             + (1/2) delta^ij (|d_l E|^2 + |d_l B|^2)],
#   <(E^2 + B^2)/2> = (E_r^2 + B^2)/2 + E_r.EMF + (sigma^2/2) sum_l (|d_l E|^2 + |d_l B|^2),
#
# with T^ij = -E^i E^j - B^i B^j + (1/2) delta^ij (E^2 + B^2) and d_l E = -d_l v x B - v x d_l B. The O(EMF^2) terms are dropped.
# The coarse run's momentum, energy and induction sources are -d_j Delta^ij, -d_j (Delta^0j), -curl EMF, with Delta^0j the
# Poynting-flux deficit above; the energy-density deficit (last line) is a state correction that is not used.

def em_flux_deficits(v, B, spacings, sigma2):
    """
    v, B: (Nt, Nx, Ny, Nz, 3) resolved lab-frame velocity and magnetic field; spacings (Dx, Dy, Dz); sigma2 the filter variance.
    Returns {'EMF': (...,3), 'Poynting': (...,3) the Poynting-flux deficit Delta^{0j}, 'stress': (...,3,3) Delta^{ij},
    'energy': (...) the energy-density deficit}; see the block comment above.
    """
    E = -np.cross(v, B)
    dv = [np.gradient(v, spacings[l], axis=l + 1) for l in range(3)]
    dB = [np.gradient(B, spacings[l], axis=l + 1) for l in range(3)]
    dE = [-np.cross(dv[l], B) - np.cross(v, dB[l]) for l in range(3)]
    emf = -sigma2 * sum(np.cross(dv[l], dB[l]) for l in range(3))
    poynting = np.cross(emf, B) + sigma2 * sum(np.cross(dE[l], dB[l]) for l in range(3))
    eye = np.eye(3)
    ee = np.sum(E * emf, axis=-1)
    quad = sum(-dE[l][..., :, None] * dE[l][..., None, :] - dB[l][..., :, None] * dB[l][..., None, :]
               + 0.5 * eye * (np.sum(dE[l] ** 2, axis=-1) + np.sum(dB[l] ** 2, axis=-1))[..., None, None] for l in range(3))
    stress = -(E[..., :, None] * emf[..., None, :] + emf[..., :, None] * E[..., None, :]) + eye * ee[..., None, None] + sigma2 * quad
    energy = ee + 0.5 * sigma2 * sum(np.sum(dE[l] ** 2, axis=-1) + np.sum(dB[l] ** 2, axis=-1) for l in range(3))
    return {'EMF': emf, 'Poynting': poynting, 'stress': stress, 'energy': energy}


# ---------------------------------------------------------------------------------------
# 4-covariant form of the electromagnetic closure
# ---------------------------------------------------------------------------------------
#
# Notation: mostly plus metric g, mu_0 = 1, the meso (Favre) observer u~^a = N^a / n~ with N^a = <n u^a> and n~ = sqrt(-N.N);
# h^{ab} = g^{ab} + u~^a u~^b projects onto its rest space; F^{ab} is the Faraday tensor and (pipeline convention)
# E^a = F^{ab} u_b, b^a = -u_b (*F)^{ab}, *F^{ab} = (1/2) eps^{abcd} F_{cd}. For ideal micro fields F^{ab} = s eps^{abcd} u_c b_d
# (s = +-1 fixes the sign convention of eps; the tests determine it).
#
# Covariance of two fields over the box filter (a spatial 3-ball in the observer's rest space):  cov(A, B) = sigma^2 h^{mu nu} grad_mu A grad_nu B
# (the sum over the three spatial tetrad legs, sum_i e_i^mu e_i^nu = h^{mu nu}). In the lab frame this is
# sigma^2 [grad A . grad B - d_t A d_t B + (u.d A)(u.d B)] with u the 4-velocity.
#
# (1) Fluid-frame sub-filter electric field, the pipeline's E~^a = u~_b <F^{ab}>  (exact to second order, given ideal micro fields):
#         E~^a = s eps^{abcd} u~_b  sigma^2 h^{mu nu} [ grad_mu u_c grad_nu b_d  -  zeta (grad_mu n / n~) grad_nu u_c  b~_d ],
#     where the second term (zeta = 1) is the Favre-versus-plain average difference <u_c> - u~_c = -cov(n, u_c) / n~ (the density term);
# (2) the electromagnetic stress-energy tensor T^{ab} = F^{ac} F^b_c - (1/4) g^{ab} F_{cd} F^{cd} is bilinear in F, so
#         <T^{ab}> = T^{ab}(F~) + sigma^2 h^{mu nu} Theta^{ab}(grad_mu F, grad_nu F),
#         Theta^{ab}(X, Y) = (1/2)(X^{ac} Y^b_c + Y^{ac} X^b_c) - (1/4) g^{ab} X_{cd} Y^{cd},
#     exact to second order with F~ = <F> (no ideal-MHD assumption);
# (3) the resolved run uses the ideal tensor F_r^{ab} = s eps^{abcd} u~_c b~_d, whose difference from F~ is the electric part:
#         F~ = F_r + u~ ^ E~  (F~^{ab} = F_r^{ab} + u~^a E~^b - u~^b E~^a), so the deficit relative to the resolved ideal field is
#         Delta^{ab} = <T^{ab}> - T^{ab}(F_r) = 2 Theta^{ab}(F_r, u~ ^ E~) + sigma^2 h^{mu nu} Theta^{ab}(grad_mu F_r, grad_nu F_r)  (+ O(E~^2)),
#     and the source of the conservation law  nabla_a T^{ab}_total = 0  is  -nabla_a Delta^{ab}  (its 0j components are the Poynting flux
#     deficit, the ij components the stress deficit, as in the lab-frame formulas above).

def levi_civita_up(metric):
    """eps^{abcd} for the given (diagonal +-1) metric, with eps_{0123} = +1 as in Base.LeviCivita4D."""
    eps = np.zeros((4, 4, 4, 4))
    for perm in itertools.permutations(range(4)):
        sign = 1
        for x in range(4):
            for y in range(x + 1, 4):
                if perm[x] > perm[y]:
                    sign = -sign
        eps[perm] = sign
    g = np.asarray(metric, dtype=float)
    return np.einsum('ae,bf,cg,dh,efgh->abcd', g, g, g, g, eps)


def faraday_ideal_sign(metric):
    """The sign s in F^{ab} = s eps^{abcd} u_c b_d for the pipeline's conventions (b^a = -u_b (*F)^{ab}, *F^{ab} = eps^{abcd} F_{cd} / 2)."""
    g = np.asarray(metric, dtype=float)
    eps = levi_civita_up(g)
    rng = np.random.default_rng(1)
    v = 0.3 * rng.normal(size=3)
    u = np.concatenate([[1.0], v]) / np.sqrt(1.0 - v @ v)
    ul = g @ u
    b = rng.normal(size=4)
    b = b + (b @ ul) * u                      # b . u = 0 (u.u = -1)
    for s in (1.0, -1.0):
        F = s * np.einsum('abcd,c,d->ab', eps, ul, g @ b)
        Fl = g @ F @ g
        dual = 0.5 * np.einsum('abcd,cd->ab', eps, Fl)
        if np.allclose(-np.einsum('ab,b->a', dual, ul), b):
            return s
    raise ValueError("no consistent sign")


def covariant_emf(u, b, n, spacings, sigma2, metric, dt=None, zeta=1.0, s=1.0):
    """
    Fluid-frame sub-filter electric field, formula (1) above. u: resolved 4-velocity u~^a (...,4); b: resolved comoving field b~^a (...,4);
    n: resolved rest density n~ (...); fields of shape (Nt,Nx,Ny,Nz,...). Derivatives: lab_derivatives (time derivative zero if dt is None).
    s: the sign in F^{ab} = s eps^{abcd} u_c b_d for the eps and conventions in use.
    """
    g = np.asarray(metric, dtype=float)
    ginv = np.linalg.inv(g)
    eps = levi_civita_up(g)
    ul = np.einsum('ab,...b->...a', g, u)
    bl = np.einsum('ab,...b->...a', g, b)
    h = ginv + u[..., :, None] * u[..., None, :]
    du = np.stack(lab_derivatives(ul, spacings, dt), axis=-2)         # (..., mu, c): d_mu u_c
    db = np.stack(lab_derivatives(bl, spacings, dt), axis=-2)
    dn = np.stack(lab_derivatives(n, spacings, dt), axis=-1)           # (..., mu)
    t1 = np.einsum('...mn,...mc,...nd->...cd', h, du, db)
    out = np.einsum('abcd,...b,...cd->...a', eps, ul, t1)
    t2 = np.einsum('...mn,...m,...nc->...c', h, dn / n[..., None], du)
    out = out - zeta * np.einsum('abcd,...b,...c,...d->...a', eps, ul, t2, bl)
    return s * sigma2 * out


def theta(X, Y, metric):
    """Theta^{ab}(X, Y) = (1/2)(X^{ac} Y^b_c + Y^{ac} X^b_c) - (1/4) g^{ab} X_{cd} Y^{cd} for X, Y of shape (...,4,4)."""
    g = np.asarray(metric, dtype=float)
    ginv = np.linalg.inv(g)
    Yl = np.einsum('...bd,dc->...bc', Y, g)                  # Y^b_c
    Xl = np.einsum('...bd,dc->...bc', X, g)
    sym = 0.5 * (np.einsum('...ac,...bc->...ab', X, Yl) + np.einsum('...ac,...bc->...ab', Y, Xl))
    XY = np.einsum('...ab,...cd,ac,bd->...', X, Y, g, g)
    return sym - 0.25 * ginv * XY[..., None, None]


def covariant_em_deficit_parts(F_r, u, E, spacings, sigma2, metric, dt=None):
    """
    The two parts of formula (3): lin = 2 Theta^{ab}(F_r, u ^ E) (linear in the fluid-frame EMF E~) and
    quad = sigma^2 h^{mu nu} Theta^{ab}(grad_mu F_r, grad_nu F_r). F_r: resolved ideal Faraday tensor (...,4,4); u: observer u~^a (...,4);
    E: fluid-frame electric field E~^a (...,4) (the data's, or covariant_emf's). Returns (lin, quad), each (...,4,4).
    """
    g = np.asarray(metric, dtype=float)
    ginv = np.linalg.inv(g)
    wedge = u[..., :, None] * E[..., None, :] - E[..., :, None] * u[..., None, :]
    h = ginv + u[..., :, None] * u[..., None, :]
    dF = lab_derivatives(F_r, spacings, dt)                   # list of 4 arrays (..., 4, 4)
    cov = 0.0
    for mu in range(4):
        for nu in range(4):
            cov = cov + h[..., mu, nu][..., None, None] * theta(dF[mu], dF[nu], g)
    return 2.0 * theta(F_r, wedge, g), sigma2 * cov


def covariant_em_deficit(F_r, u, E, spacings, sigma2, metric, dt=None):
    """
    Formula (3): Delta^{ab} = 2 Theta^{ab}(F_r, u ^ E) + sigma^2 h^{mu nu} Theta^{ab}(grad_mu F_r, grad_nu F_r), the sum of covariant_em_deficit_parts.
    The coarse run's source is -d_j Delta^{jb}.
    """
    lin, quad = covariant_em_deficit_parts(F_r, u, E, spacings, sigma2, metric, dt)
    return lin + quad
