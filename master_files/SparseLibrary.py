"""
Sparse regression of E_tilde on a library of candidate operators (SINDy-like, sequentially thresholded least squares).

The question: does the data pick out the second-order filter term  E~ = -sigma^2 sum_k d_k v x d_k B  (+ density term) from a larger family of candidate
terms, without the form being imposed? The library has three kinds of columns, all 4-vectors in the observer frame (orthogonal to u~):
  - the Ohm's-law terms  J~, B~, Hall  (OhmsRegression.standard_operators);
  - the density term of the second-order model (OhmsRegression.gradient_model_operators, 'model_dens');
  - the quadratic gradient forms below, sigma^2 times a vector built from d_k v_i and d_l B_j (lab frame, spatial gradients of the filtered fields).

Quadratic gradient forms. A vector bilinear in the gradients S_ki = d_k v_i and T_lj = d_l B_j, linear in an odd number of epsilons (B is axial, E polar), is a contraction of the
rank-5 tensor S_ki T_lj with eps_{mab} delta_{cd} for a choice of 3 of the 5 indices on eps and the other 2 on delta: 10 forms, plus w x J (w = curl v, J = curl B), which is a
combination of them. The isotropic rank-5 tensors of three dimensions span only 6 dimensions (Schouten identity), so the 11 forms are linearly dependent: **6 independent
forms for general gradients, 5 when div B = 0, 4 when also div v = 0** (checked numerically in the tests). `independent_columns` keeps a maximal independent subset in the order
given, so the physically preferred forms come first.

The regression (`stlsq`) works on the Gram matrix G_ik = sum <O_i, O_k>, r_i = sum <O_i, E> (Minkowski product over the 4-vector, summed over the points of the region, or over
windows), exactly as OhmsRegression.fit_region, so it needs no change for 4-vectors. Columns are scaled to unit norm, so the threshold is on a dimensionless quantity: the
fraction of |E| carried by the term. Windows (`window_sums`) integrate each operator and E over a box of the (t,x,y,z) meso grid with a test function (box or Hann) before the
regression, which gives rows that are volume integrals and a natural unit for the bootstrap (`bootstrap_inclusion`). Note that this is not the integration by parts of the weak form
of the PDE-SINDy literature: the derivatives in the library are still finite differences of the filtered fields.

Pure numpy; all vectors are 4-vectors with the Minkowski product (-+++), which is positive on the vectors orthogonal to u~.
"""
import itertools

import numpy as np

import OhmsRegression as OR


def levi3():
    e = np.zeros((3, 3, 3))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        e[i, j, k] = 1.0
        e[i, k, j] = -1.0
    return e


EPS3 = levi3()

# name -> einsum over (eps, S, T) with S[..., k, i] = d_k v_i, T[..., l, j] = d_l B_j; output [..., m]. 'w_x_J' is handled separately.
# The preferred order (the filter's own form first) decides which of the dependent forms is kept.
FORMS = {
    'dv_x_dB':        'mij,...ki,...kj->...m',    # sum_k d_k v x d_k B  (the covariance of the filter; expected coefficient -1)
    'w_x_J':          None,                       # (curl v) x (curl B)
    'divv_J':         'mlj,...kk,...lj->...m',    # (div v) (curl B)
    'J_grad_v':       'klj,...lj,...km->...m',    # (J . grad) v
    'w_grad_B':       'lki,...ki,...lm->...m',    # (w . grad) B
    'gradvT_J':       'ilj,...lj,...mi->...m',    # J_i d_m v_i
    'gradBT_w':       'jki,...ki,...mj->...m',    # w_j d_m B_j
    'gradv_i_x_gradB_i': 'mkl,...ki,...li->...m',  # sum_i grad v_i x grad B_i
    'mixed_vB':       'mkj,...ki,...ij->...m',    # eps_{mkj} d_k v_i d_i B_j
    'mixed_vB2':      'mil,...ki,...lk->...m',    # eps_{mil} d_k v_i d_l B_k
    'w_divB':         'mki,...ki,...jj->...m',    # (curl v)(div B): zero for solenoidal B
}


def gradient_tensor(f, spacings):
    """G[..., k, i] = d_k f_i for f of shape (Nt, Nx, Ny, Nz, 3) (central differences); k runs over the three spatial axes."""
    return np.stack([np.gradient(f, spacings[k], axis=k + 1) for k in range(3)], axis=-2)


def quadratic_forms(S, T, names=None):
    """The quadratic gradient forms (3-vectors, no sigma^2) of S[..., k, i] = d_k v_i and T[..., l, j] = d_l B_j, as a dict name -> (..., 3)."""
    names = list(FORMS) if names is None else list(names)
    out = {}
    for n in names:
        if n == 'w_x_J':
            w = np.einsum('lki,...ki->...l', EPS3, S)
            J = np.einsum('klj,...lj->...k', EPS3, T)
            out[n] = np.cross(w, J)
        else:
            out[n] = np.einsum(FORMS[n], EPS3, S, T)
    return out


def quadratic_library(vel, B, spacings, sigma2, gamma=None, names=None):
    """
    sigma^2 times the quadratic forms of the filtered lab-frame velocity vel (..., 3) and field B (..., 3), as 4-vectors in the observer frame
    gamma (1, vel) (OhmsRegression.observer_electric). gamma=None gives the spatial 4-vector (0, g), i.e. the static observer; pass gamma = u~^0 for the meso observer.
    """
    S, T = gradient_tensor(vel, spacings), gradient_tensor(B, spacings)
    forms = quadratic_forms(S, T, names)
    out = {}
    for n, g in forms.items():
        g = sigma2 * g
        if gamma is None:
            x = np.zeros(g.shape[:-1] + (4,))
            x[..., 1:] = g
            out[n] = x
        else:
            out[n] = OR.observer_electric(g, vel, gamma)
    return out


def density_form(vel, B, D, spacings, sigma2, gamma=None):
    """
    The density-weighting (Favre) term of the second-order model, g2 = sigma^2 (sum_k d_k D d_k v / D) x B (expected coefficient +1), as a 4-vector in the observer frame
    (static if gamma is None). Spatial gradients only, as the quadratic forms.
    """
    dD = np.stack([np.gradient(D, spacings[k], axis=k + 1) for k in range(3)], axis=-1)       # [..., k]
    S = gradient_tensor(vel, spacings)                                                        # [..., k, i]
    T = np.einsum('...k,...ki->...i', dD, S)
    g = sigma2 * np.cross(T / D[..., None], B)
    if gamma is None:
        x = np.zeros(g.shape[:-1] + (4,))
        x[..., 1:] = g
        return x
    return OR.observer_electric(g, vel, gamma)


def static_library(vel, B, D, spacings, sigma2, names=None, density=True, ohm=True):
    """
    The whole library for a static observer (4-vectors (0, g)): the quadratic forms, the density term (if `density`) and the Ohm's-law terms J = curl B, B, J x B
    (if `ohm`; named like OhmsRegression.TERMS). Used for synthetic data; for real data the Ohm terms come from the model (J~, B~, Hall in the meso observer frame).
    """
    lib = quadratic_library(vel, B, spacings, sigma2, None, names)
    if density:
        lib['model_dens'] = density_form(vel, B, D, spacings, sigma2)
    if ohm:
        J = np.einsum('klj,...lj->...k', EPS3, gradient_tensor(B, spacings))
        for n, x in zip(OR.TERMS, (J, B, np.cross(J, B))):
            a = np.zeros(x.shape[:-1] + (4,))
            a[..., 1:] = x
            lib[n] = a
    return lib


def spatial4(x):
    """3-vectors (..., 3) as 4-vectors (0, x)."""
    a = np.zeros(x.shape[:-1] + (4,))
    a[..., 1:] = x
    return a


# ---- windows and the Gram system ---------------------------------------------------------------------------------------------------------------------------------------

def _test_function(shape, kind):
    if kind == 'box':
        return np.ones(shape)
    if kind == 'hann':
        w = [np.hanning(n + 2)[1:-1] if n > 1 else np.ones(1) for n in shape]
        return np.einsum('a,b,c,d->abcd', *w)
    raise ValueError("test function must be 'box' or 'hann'")


def window_sums(arrays, window, stride=None, kind='box'):
    """
    Integrals of each array (Nt, Nx, Ny, Nz, ...) over boxes of the meso grid with the test function `kind`: (wt, wx, wy, wz) = window, stride (default the window: tiles)
    in each direction. Returns {name: (Nw, ...)}. The sum over the box is the spacetime volume integral up to the constant cell volume.
    """
    shape = next(iter(arrays.values())).shape[:4]
    window = tuple(min(int(w), n) for w, n in zip(window, shape))
    stride = window if stride is None else tuple(int(s) for s in stride)
    phi = _test_function(window, kind)
    starts = [range(0, n - w + 1, s) for n, w, s in zip(shape, window, stride)]
    out = {k: [] for k in arrays}
    for i0 in itertools.product(*starts):
        sl = tuple(slice(i, i + w) for i, w in zip(i0, window))
        for k, a in arrays.items():
            x = a[sl]
            out[k].append(np.tensordot(phi, x, axes=4))
    return {k: np.array(v) for k, v in out.items()}


def gram_system(ops, E, names):
    """(G, r, EE): G_ik = sum <O_i, O_k>, r_i = sum <O_i, E>, EE = sum <E, E> over all leading indices (Minkowski product on the last axis of 4 components)."""
    O = [np.asarray(ops[n], dtype=float).reshape(-1, 4) for n in names]
    E = np.asarray(E, dtype=float).reshape(-1, 4)
    G = np.array([[OR.mink_dot(a, b).sum() for b in O] for a in O])
    r = np.array([OR.mink_dot(a, E).sum() for a in O])
    return G, r, float(OR.mink_dot(E, E).sum())


def independent_columns(G, tol=1e-3, order=None, min_norm=1e-6):
    """
    Indices of a maximal linearly independent subset of the columns, taken in `order` (default 0, 1, 2, ...): a column is dropped when the part of it not in the span of
    the columns kept so far has a norm below tol times its own (Gram-Schmidt in the Gram metric), or when its own norm is below min_norm times the largest (a column that is
    zero to round-off, such as (div v) J in an incompressible flow, would otherwise be an arbitrary unit vector after scaling). Returns (kept, dropped).
    """
    order = list(range(len(G))) if order is None else list(order)
    kept, dropped = [], []
    gmax = float(np.max(np.diag(G))) if len(G) else 0.0
    for i in order:
        if G[i, i] <= min_norm ** 2 * gmax:
            dropped.append(i)
            continue
        if kept:
            Gk = G[np.ix_(kept, kept)]
            c = np.linalg.solve(Gk + 1e-14 * np.trace(Gk) * np.eye(len(kept)), G[kept, i])
            resid = G[i, i] - c @ G[kept, i]
        else:
            resid = G[i, i]
        (kept if resid > tol ** 2 * G[i, i] else dropped).append(i)
    return kept, dropped


def stlsq(G, r, EE, threshold=0.05, ridge=1e-10, max_iter=20):
    """
    Sequentially thresholded least squares on the Gram system. Columns are scaled to unit norm, c_i is the coefficient of the unit-norm column, and `threshold` is on
    |c_i| / sqrt(EE): the fraction of the rms of E carried by the term. Terms below it are removed, the rest refitted, until the support does not change.
    Returns (coefficients in the units of the original columns, 0 for removed terms; the boolean support; R2 = 1 - rss/EE).
    """
    n = len(r)
    d = np.sqrt(np.maximum(np.diag(G), 1e-300))
    Gn, rn = G / np.outer(d, d), r / d
    active = np.ones(n, dtype=bool)
    c = np.zeros(n)
    for _ in range(max_iter):
        idx = np.where(active)[0]
        if len(idx) == 0:
            c = np.zeros(n)
            break
        A = Gn[np.ix_(idx, idx)] + ridge * np.eye(len(idx))
        c = np.zeros(n)
        c[idx] = np.linalg.solve(A, rn[idx])
        small = active & (np.abs(c) < threshold * np.sqrt(EE))
        if not small.any():
            break
        active &= ~small
        c[small] = 0.0
    rss = EE - 2.0 * c @ rn + c @ Gn @ c
    return c / d, active, float(1.0 - rss / EE) if EE > 0 else float('nan')


def stlsq_path(G, r, EE, thresholds):
    """The STLSQ solution at each threshold: list of (threshold, coefficients, support, R2)."""
    return [(t,) + stlsq(G, r, EE, t) for t in thresholds]


def bootstrap_inclusion(ops_w, E_w, names, threshold=0.05, n_boot=200, seed=0):
    """
    Stability of the selection: STLSQ on resamples (with replacement) of the windows. ops_w: {name: (Nw, 4)}, E_w: (Nw, 4).
    Returns {name: (inclusion frequency, median coefficient, 16th, 84th percentile of the coefficient over the resamples in which it is included)}.
    """
    rng = np.random.default_rng(seed)
    Nw = len(E_w)
    coefs = np.zeros((n_boot, len(names)))
    incl = np.zeros((n_boot, len(names)), dtype=bool)
    for b in range(n_boot):
        ix = rng.integers(0, Nw, Nw)
        G, r, EE = gram_system({n: ops_w[n][ix] for n in names}, E_w[ix], names)
        c, act, _ = stlsq(G, r, EE, threshold)
        coefs[b], incl[b] = c, act
    out = {}
    for j, n in enumerate(names):
        sel = coefs[incl[:, j], j]
        out[n] = (float(incl[:, j].mean()),
                  float(np.median(sel)) if len(sel) else float('nan'),
                  float(np.percentile(sel, 16)) if len(sel) else float('nan'),
                  float(np.percentile(sel, 84)) if len(sel) else float('nan'))
    return out


# ---- synthetic fields ----------------------------------------------------------------------------------------------------------------------------------------------

def _random_periodic(shape, rng, kmax=3, amp=1.0):
    """A smooth periodic scalar field on a cube: random Fourier modes with |k_i| <= kmax, amplitude falling as 1/(1+|k|^2)."""
    n = shape[0]
    x = np.arange(n) * 2 * np.pi / n
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    f = np.zeros(shape)
    for kx, ky, kz in itertools.product(range(-kmax, kmax + 1), repeat=3):
        if (kx, ky, kz) <= (0, 0, 0):                       # one of each +-k pair
            continue
        k2 = kx * kx + ky * ky + kz * kz
        if k2 == 0 or k2 > kmax * kmax:
            continue
        a, b = rng.normal(size=2) / (1.0 + k2)
        ph = kx * X + ky * Y + kz * Z
        f += amp * (a * np.cos(ph) + b * np.sin(ph))
    return f


def _curl_periodic(A, h):
    """Spectral curl of a periodic vector field (n, n, n, 3), grid spacing h."""
    n = A.shape[0]
    k = np.fft.fftfreq(n, d=h) * 2 * np.pi
    K = np.meshgrid(k, k, k, indexing='ij')
    Ah = [np.fft.fftn(A[..., i]) for i in range(3)]
    c = [1j * (K[(i + 1) % 3] * Ah[(i + 2) % 3] - K[(i + 2) % 3] * Ah[(i + 1) % 3]) for i in range(3)]
    return np.stack([np.fft.ifftn(x).real for x in c], axis=-1)


def box_average(f, w):
    """Periodic running mean over w cells along each of the three spatial axes (axes 0, 1, 2 of f)."""
    out = f
    for ax in range(3):
        out = sum(np.roll(out, s - (w - 1) // 2, axis=ax) for s in range(w)) / w
    return out


def synthetic_meso(n_fine=48, width=4, stride=2, seed=0, kmax=3, vamp=0.3, compressible=True, density_amp=0.2):
    """
    A fine periodic grid with smooth random v (|v| ~ vamp), solenoidal B = curl A and density D; box-averaged over `width` fine cells and sampled every `stride` cells.
    Returns a dict with the meso fields {'v', 'B', 'D'} (each with a leading time axis of length 1: shape (1, N, N, N, ...)), the meso spacing, sigma^2 of the discrete box
    filter ((width^2 - 1)/12 h^2, h = fine grid spacing), 'E_exact': the exact EMF deficit  E~ = -(<v x B> - <v>x<B>)  (plain average velocity, 'v_plain'), and 'E_favre':
    E~ = -<v x B> + vf x <B> with the density-weighted velocity vf (= 'v'), the quantity of the real model (u~ is the Favre velocity), which the second-order model
    with its density term describes: coefficients -1 (dv_x_dB) and +1 (density).
    """
    rng = np.random.default_rng(seed)
    shp = (n_fine,) * 3
    h = 2 * np.pi / n_fine
    v = np.stack([_random_periodic(shp, rng, kmax, vamp) for _ in range(3)], axis=-1)
    if not compressible:                                    # remove the compressive part spectrally
        k = np.fft.fftfreq(n_fine, d=h) * 2 * np.pi
        K = np.stack(np.meshgrid(k, k, k, indexing='ij'), axis=-1)
        vh = np.stack([np.fft.fftn(v[..., i]) for i in range(3)], axis=-1)
        k2 = np.sum(K ** 2, axis=-1)
        k2[0, 0, 0] = 1.0
        proj = vh - K * (np.sum(K * vh, axis=-1) / k2)[..., None]
        v = np.stack([np.fft.ifftn(proj[..., i]).real for i in range(3)], axis=-1)
    A = np.stack([_random_periodic(shp, rng, kmax, 1.0) for _ in range(3)], axis=-1)
    B = _curl_periodic(A, h) + np.array([0.0, 0.0, 1.0])
    D = 1.0 + density_amp * _random_periodic(shp, rng, kmax, 1.0) if compressible else np.ones(shp)
    avg = lambda f: box_average(f, width)
    sub = (slice(0, None, stride),) * 3
    vm, Bm, Dm = avg(v), avg(B), avg(D)
    E_exact = -(avg(np.cross(v, B)) - np.cross(vm, Bm))
    Dv = avg(D[..., None] * v)
    vf = Dv / Dm[..., None]                                 # Favre velocity
    E_favre = -avg(np.cross(v, B)) + np.cross(vf, Bm)       # the filtered E minus the ideal field of the Favre velocity: -<dv x dB> - (<v> - vf) x <B>
    cell = lambda f: f[sub][None]
    return {'v': cell(vf), 'B': cell(Bm), 'D': cell(Dm), 'v_plain': cell(vm), 'E_exact': cell(E_exact), 'E_favre': cell(E_favre),
            'spacing': (stride * h,) * 3, 'sigma2': (width ** 2 - 1) / 12.0 * h * h}
